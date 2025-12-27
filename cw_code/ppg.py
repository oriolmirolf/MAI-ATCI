import copy
from collections import deque
from typing import Deque, Dict, Optional
import gym
import gc
import subprocess

import itertools

from torch.optim.lr_scheduler import ReduceLROnPlateau

# GOD: https://github.com/MineDojo/MineDojo/issues/85

import numpy as np
import torch as th
from helpers import (
    calculate_gae,
    clipped_value_loss,
    create_agent,
    load_model_parameters,
    normalize,
    update_network_,
    minerl_to_minedojo,
    add_text_to_frame,
    obs_to_device,
    hidden_to_device,
    kill_java_processes,
    freeze_policy_layers,
)
from memory_2 import AuxMemory, Episode, Memory, create_dataloader
from torch.optim import Adam
import os
import cv2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from steve1.VPT.lib.tree_util import tree_map
from steve1.config import PRIOR_INFO
from steve1.data.text_alignment.vae import load_vae_model
from steve1.utils.mineclip_agent_env_utils import load_mineclip_wconfig
from steve1.utils.embed_utils import get_prior_embed
from steve1.VPT.lib.scaled_mse_head import ScaledMSEHead
import minedojo

import time

from torch.profiler import profile, record_function, ProfilerActivity
import psutil

def get_ram_info():
    ram = psutil.virtual_memory()
    total_ram = ram.total
    available_ram = ram.available
    used_ram = ram.used
    free_ram = ram.free
    percent_ram = ram.percent
    
    print(f"Total RAM: {total_ram}")
    print(f"Available RAM: {available_ram}")
    print(f"Used RAM: {used_ram}")
    print(f"Free RAM: {free_ram}")
    print(f"Percent RAM: {percent_ram}")


class PPG:
    def __init__(
        self,
        env_name,
        model,
        weights,
        out_weights,
        device,
        epochs_wake,
        epochs_sleep,
        minibatch_size,
        lr,
        max_grad_norm,
        betas,
        gamma,
        lam,
        clip,
        value_clip,
        kl_beta,
        p_decay,
        beta_clone,
        Toff,
        Tfreq,
        Nenv,
        M,
        Kpi,
        Kv,
        beta_v,
        freeze_layers,
    ):
        self.epochs_wake = epochs_wake
        self.epochs_sleep = epochs_sleep
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.betas = betas
        self.kl_beta = kl_beta
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.value_clip = value_clip
        self.env_name = env_name
        self.device = device
        self.out_weights = out_weights
        self.beta_clone = beta_clone
        self.Toff = Toff
        self.Tfreq = Tfreq
        self.Nenv = Nenv
        self.M = M
        self.Kpi = Kpi
        self.Kv = Kv
        self.beta_v = beta_v
        
        self.episode_count = 0
        
        # Load model parameters and create agents
        agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(model)
        
        self.cond_scale = 6.0
        
        # Initialize agents
        self.agent1 = create_agent(
            agent_policy_kwargs, agent_pi_head_kwargs, weights, cond_scale=self.cond_scale
        )
        self.agent2 = copy.deepcopy(self.agent1)
        
        # we need 2 because they are different! assymetric!
        self.initial_agent1 = self._make_policy_snapshot(self.agent1)
        self.initial_agent2 = self._make_policy_snapshot(self.agent2)
        self.kl_beta = kl_beta
        self.p_decay = p_decay

        # Initialize value head weights and biases to zero
        with th.no_grad():
            self.agent1.policy.value_head.linear.weight.zero_()
            self.agent1.policy.value_head.linear.bias.zero_()
            self.agent2.policy.value_head.linear.weight.zero_()
            self.agent2.policy.value_head.linear.bias.zero_()
        
        if freeze_layers:
            trainable_layers1 = [
                self.agent1.policy.pi_head,
                self.agent1.policy.net.lastlayer,
                self.agent1.policy.value_head,
            ]
            trainable_layers2 = [
                self.agent2.policy.pi_head,
                self.agent2.policy.net.lastlayer,
                self.agent2.policy.value_head,
            ]
            self.trainable_parameters1 = freeze_policy_layers(
                self.agent1.policy, trainable_layers1
            )
            self.trainable_parameters2 = freeze_policy_layers(
                self.agent2.policy, trainable_layers2
            )
        else:  # FIX: ensure .trainable_parameters* defined even when not freezing
            self.trainable_parameters1 = list(self.agent1.policy.parameters())
            self.trainable_parameters2 = list(self.agent2.policy.parameters())

        self.optimizer1 = Adam(self.trainable_parameters1, lr=self.lr, betas=self.betas)
        self.optimizer2 = Adam(self.trainable_parameters2, lr=self.lr, betas=self.betas)

        # encoder param lists (to avoid interferfence of critic on policy phase!)
        self._enc_params1 = list(self.agent1.policy.net.parameters()) # backbone only
        self._enc_params2 = list(self.agent2.policy.net.parameters())
        
        self.scheduler1 = ReduceLROnPlateau(self.optimizer1, mode='min', factor=0.1, patience=20)
        self.scheduler2 = ReduceLROnPlateau(self.optimizer2, mode='min', factor=0.1, patience=20)
        
        # prepare for saving info
        self.base_name = f"training-{env_name}-{int(time.time())}"
        os.makedirs(f'multiagent_training/data/training/{self.base_name}', exist_ok=True)
        
        # Prompt embeddings
        mineclip = load_mineclip_wconfig()
        prior = load_vae_model(PRIOR_INFO)
        self.prompt_embed_agent1 = get_prior_embed("locate the dirt block in the room and break it, find the block of dirt and mine it, search for a dirt block in this room and destroy it, look around the room for a dirt block and punch it", mineclip, prior, device)
        self.prompt_embed_agent2 = get_prior_embed("find and punch the hostile mobs, locate and attack the zombies and skeletons, search for creepers and other mobs to hit, look for hostile creatures in the area and fight them", mineclip, prior, device)
        del mineclip
        del prior
        gc.collect()
        
        # Tensorboard logging
        self.writer = SummaryWriter()
        self.wake_updates = 0
        self.sleep_updates = 0
        
        # Initialize off-policy buffer
        self.off_policy_buffer1 = deque(maxlen=self.M)
        self.off_policy_buffer2 = deque(maxlen=self.M)
        
        self.num_fast_reset = 0
        
    def _make_policy_snapshot(self, agent):
        """Deep‑copies `agent` and freezes all params (no grads)."""
        snap = copy.deepcopy(agent)
        snap.eval()
        for p in snap.policy.parameters():
            p.requires_grad_(False)
        return snap.to(self.device)
        
    def _store_in_replay(self, episode, buffer):
        for mem in episode:
            if np.random.rand() < 1.0 / self.Toff:
                buffer.append(mem)

        
    def save(self):
        """
        Saves the model weights for agent and value functions to the specified path
        """
        
        weight_base_path = f"multiagent_training/data/training/{self.base_name}/weights/"
        os.makedirs(weight_base_path, exist_ok=True)
        
        collector_policy_path   = weight_base_path + f"Collector_{self.episode_count}.pt"
        fighter_policy_path     = weight_base_path + f"Fighter_{self.episode_count}.pt"

        th.save(self.agent1.policy.state_dict(), collector_policy_path)
        th.save(self.agent2.policy.state_dict(), fighter_policy_path)

        print(f"Saved Collector model weights to {collector_policy_path}")
        print(f"Saved Fighter model weights to {fighter_policy_path}")
    
    
    def collect_episodes(self, num_episodes_to_collect, env, record_video):
        
        Episode1 = []
        Episode2 = []
        
        if record_video:
            # Initialize video writers for each agent
            weight_video_path = f"multiagent_training/data/training/{self.base_name}/videos/"

            os.makedirs(weight_video_path, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
            out = cv2.VideoWriter(f'{weight_video_path}ep_{self.episode_count}_both_agents_video.mp4', fourcc, 20.0, (1280, 360))
        
        
        for _ in range(num_episodes_to_collect):

            while True:
                try:
                    
                    if self.num_fast_reset == 27:
                        env.close()
                        del env
                        gc.collect()
                        kill_java_processes()
                        env = minedojo.make(
                            task_id=self.env_name,
                            image_size=(360, 640),
                        )
                        self.num_fast_reset = 0
                    
                    obs = env.reset()
                    self.agent1.reset(cond_scale=self.cond_scale)
                    self.agent2.reset(cond_scale=self.cond_scale)
                    self.num_fast_reset += 1
                    break
                except:
                    env.close()
                    del env
                    gc.collect()
                    kill_java_processes()
                    env = minedojo.make(
                        task_id=self.env_name,
                        image_size=(360, 640),
                    )
                    
            dummy_first = th.from_numpy(np.array((False,))).to(self.device)
            hidden_state1 = self.agent1.policy.initial_state(1)
            hidden_state2 = self.agent2.policy.initial_state(1)
            
            # this is hell: dataloader things.
            with th.no_grad():
                for i in range(len(hidden_state1)):
                    hidden_state1[i] = list(hidden_state1[i])
                    hidden_state1[i][0] = th.from_numpy(np.full(
                        (1, 1, 128), False)).to(device)
                
                for i in range(len(hidden_state2)):
                    hidden_state2[i] = list(hidden_state2[i])
                    hidden_state2[i][0] = th.from_numpy(np.full(
                        (1, 1, 128), False)).to(device)
            
            done = False
            episode_reward1 = 0
            episode_reward2 = 0
            
            steps = 0
            while not done:
                # print(f"steps: {steps}")
                steps += 1
                
                # Preprocess the observation
                with th.no_grad():
                    agent_obs1 = self.agent1._env_obs_to_agent(obs[0], self.prompt_embed_agent1)
                    agent_obs2 = self.agent2._env_obs_to_agent(obs[1], self.prompt_embed_agent2)
                    with th.cuda.amp.autocast():
                        # AGENT 1
                        action1, new_state1, result1 = self.agent1.policy.act(
                            agent_obs1, dummy_first, hidden_state1, return_pd=True, cond_scale=None
                        )
                        value1 = result1["vpred"]
                        # AGENT 2
                        action2, new_state2, result2 = self.agent2.policy.act(
                            agent_obs2, dummy_first, hidden_state2, return_pd=True, cond_scale=None
                        )
                        value2 = result2["vpred"]
                        
                        # Take the action in the environment
                        # agent 1
                        minerl_action1 = self.agent1._agent_action_to_env(action1)
                        minerl_action1["ESC"] = 0
                        minedojo_action1 = minerl_to_minedojo(minerl_action1)
                        
                        # agent 2
                        minerl_action2 = self.agent2._agent_action_to_env(action2)
                        minerl_action2["ESC"] = 0
                        minedojo_action2 = minerl_to_minedojo(minerl_action2)
                        next_obs, reward, next_done, _ = env.step([minedojo_action1, minedojo_action2])
                        episode_reward1 += reward[0]
                        episode_reward2 += reward[1]
                
                if next_done:
                    with th.cuda.amp.autocast():
                        # Compute the value for the next observation
                        # agent1
                        with th.no_grad():
                            next_value1 = self.agent1.policy.v(
                                obs     = self.agent1._env_obs_to_agent(next_obs[0], self.prompt_embed_agent1),
                                first   = dummy_first,
                                state_in= new_state1
                            )
                            # agent2
                            next_value2 = self.agent2.policy.v(
                                obs     = self.agent2._env_obs_to_agent(next_obs[1], self.prompt_embed_agent2),
                                first   = dummy_first,
                                state_in= new_state2
                            )
                    # Add the next_value to the last memory of the episode
                    memory1 = Memory(
                        agent_obs1,
                        obs[0]["rgb"],
                        action1,
                        result1["log_prob"],
                        reward[0],
                        next_done,
                        value1,
                        next_value1,
                        hidden_state1,
                        result1["pd"],
                        None,          # targ_return placeholder
                    )
                    memory2 = Memory(
                        agent_obs2,
                        obs[1]["rgb"],
                        action2,
                        result2["log_prob"],
                        reward[1],
                        next_done,
                        value2,
                        next_value2,
                        hidden_state2,
                        result2["pd"],
                        None,          # targ_return placeholder
                    )
                    Episode1.append(memory1)
                    Episode2.append(memory2)
                    break
                memory1 = Memory(
                    agent_obs1,
                    obs[0]["rgb"],
                    action1,
                    result1["log_prob"],
                    reward[0],
                    next_done,
                    value1,
                    None,
                    hidden_state1,
                    result1["pd"],
                    None
                )
                memory2 = Memory(
                    agent_obs2,
                    obs[1]["rgb"],
                    action2,
                    result2["log_prob"],
                    reward[1],
                    next_done,
                    value2,
                    None,
                    hidden_state2,
                    result2["pd"],
                    None
                )
                                
                Episode1.append(memory1)
                Episode2.append(memory2)
                # update the hidden state for the next timestep
                hidden_state1 = new_state1
                hidden_state2 = new_state2
                obs = next_obs
                done = next_done
                
                del new_state1
                del new_state2
                del next_obs
                # if steps % 20 == 0: # My gpu is shit so we must do this :(
                #     th.cuda.empty_cache()
                #     gc.collect()
                
                if record_video:
                    frame0 = np.transpose(obs[0]["rgb"], (1, 2, 0))
                    frame1 = np.transpose(obs[1]["rgb"], (1, 2, 0))
                    combined_frame = np.concatenate((frame0, frame1), axis=1)
                    
                    combined_frame = add_text_to_frame(combined_frame, 'Collector', (50, 50))
                    combined_frame = add_text_to_frame(combined_frame, 'Fighter', (690, 50))
                    
                    combined_frame = add_text_to_frame(combined_frame, f"Reward: {episode_reward1:.3f}", (400, 50))
                    combined_frame = add_text_to_frame(combined_frame, f"Reward: {episode_reward2:.3f}", (1040, 50))
                    
                    # Add episode number at the bottom middle
                    combined_frame = add_text_to_frame(combined_frame, f'Episode {self.episode_count}', (590, 340))
                    
                    out.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
            
            self.writer.add_scalar("Episode Reward/Agent 1", episode_reward1, self.episode_count)
            self.writer.add_scalar("Episode Reward/Agent 2", episode_reward2, self.episode_count)
            self.writer.add_scalar("Episode Reward/Cumulative", episode_reward1 + episode_reward2, self.episode_count)
            self.episode_count += 1
            
            del hidden_state1
            del hidden_state2
            del dummy_first
            
        if record_video:
                out.release()
        
        # cache GAE targets in the episode
        # NOTE:
        #   This is SUPER IMPORTANT for the GAE to work properly, because we only save 1/Toff of each episode the buffers
        #   are UNORDERED! Therefore we can NOT compute the GAE later (e.g., when we'll use it), as most implementations do.
        rewards_e1 = [m.reward for m in Episode1]
        values_e1  = [m.value for m in Episode1] + [Episode1[-1].next_value]
        masks_e1   = [1.0 - float(m.done) for m in Episode1]
        returns_e1 = calculate_gae(rewards_e1, values_e1, masks_e1, self.gamma, self.lam)

        rewards_e2 = [m.reward for m in Episode2]
        values_e2  = [m.value for m in Episode2] + [Episode2[-1].next_value]
        masks_e2   = [1.0 - float(m.done) for m in Episode2]
        returns_e2 = calculate_gae(rewards_e2, values_e2, masks_e2, self.gamma, self.lam)

        for idx, R in enumerate(returns_e1):
            Episode1[idx] = Episode1[idx]._replace(targ_return=R)
        for idx, R in enumerate(returns_e2):
            Episode2[idx] = Episode2[idx]._replace(targ_return=R)
        
        print(
            f"[INFO] Finished collecting episodes | memories: {len(Episode1)}"
        )
        return Episode1, Episode2, env
    
    def train(
        self,
        n_iterations,
        n_rollouts,
    ):
        env = minedojo.make(
            task_id=self.env_name,
            image_size=(360, 640),
        )
        
        for iteration in tqdm(range(n_iterations), desc="iterations"):
            # Data collection
            record_video = False
            if iteration % 100 == 0:
                record_video = True
            episode1, episode2, env = self.collect_episodes(
                n_rollouts, env, record_video=record_video
            )
            self._store_in_replay(episode1, self.off_policy_buffer1)
            self._store_in_replay(episode2, self.off_policy_buffer2)

            # policy phase
            self.initial_agent1 = self._make_policy_snapshot(self.agent1) # update the initial agent
            self.initial_agent2 = self._make_policy_snapshot(self.agent2)
            for _ in range(self.Kpi):
                self.optimize_policy_value(episode1, episode2)
                
            # aux phase
            if (iteration+1) % self.Tfreq == 0: # as paper said, we wait at least Tfreq iters. first
                self.initial_agent1 = self._make_policy_snapshot(self.agent1) # idem ^^
                self.initial_agent2 = self._make_policy_snapshot(self.agent2)
                for _ in range(self.Kv):
                    self.optimize_value_function()

            # update the KL divergence coefficient
            self.kl_beta *= self.p_decay
            

            del episode1, episode2
            gc.collect()
            if iteration % 250 == 0 and iteration != 0:
                self.save()
        
        self.save()
        env.close()
    
    def sample_minibatch(self, memories: Deque[Memory]):
        if len(memories) == 0:
            raise ValueError("Off‑policy buffer empty!")
        size = min(self.minibatch_size, len(memories))  # FIX
        idx = np.random.choice(len(memories), size=size, replace=False)
        return [memories[i] for i in idx]
    
    
    def optimize_policy_value(self, batch1, batch2):
        # on-policy; batch1 and batch2 are the just-collected episodes
        
        # Prepare the data
        agent_observations1 = [memory.agent_obs for memory in batch1]
        actions1            = [memory.action for memory in batch1]
        old_log_probs1      = [memory.action_log_prob for memory in batch1]
        rewards1            = [memory.reward for memory in batch1]
        masks1              = [1 - float(memory.done) for memory in batch1]
        values1             = [memory.value for memory in batch1]
        hidden_states1      = [memory.hidden_state for memory in batch1]
        
        agent_observations2 = [memory.agent_obs for memory in batch2]
        actions2            = [memory.action for memory in batch2]
        old_log_probs2      = [memory.action_log_prob for memory in batch2]
        rewards2            = [memory.reward for memory in batch2]
        masks2              = [1 - float(memory.done) for memory in batch2]
        values2             = [memory.value for memory in batch2]
        hidden_states2      = [memory.hidden_state for memory in batch2]
        
        # Use the next_value from the last memory in the batch
        next_value1 = batch1[-1].next_value
        values1 = values1 + [next_value1]
        
        next_value2 = batch2[-1].next_value
        values2 = values2 + [next_value2]

        # pre-computed GAEs!!
        returns1 = [m.targ_return for m in batch1]
        returns2 = [m.targ_return for m in batch2]
        
        # Convert to tensors and move to device
        old_values1     = th.tensor(values1[:-1]).to(self.device)
        old_log_probs1  = th.tensor(old_log_probs1).to(self.device)
        returns1        = th.tensor(returns1).float().to(self.device)
                
        old_values2     = th.tensor(values2[:-1]).to(self.device)
        old_log_probs2  = th.tensor(old_log_probs2).to(self.device)
        returns2        = th.tensor(returns2).float().to(self.device)
                
        
        # Prepare the data for the dataloader
        data = [
            agent_observations1, agent_observations2,
            actions1, actions2,
            old_log_probs1, old_log_probs2,
            rewards1, rewards2,
            returns1, returns2,
            old_values1, old_values2,
            hidden_states1, hidden_states2,
        ]
        
        # Create the dataloader
        dl = create_dataloader(data, batch_size=self.minibatch_size)
        
        # Optimize policy and value function
        for (
            agent_obs_batch1, agent_obs_batch2,
            actions_batch1, actions_batch2,
            old_log_probs_batch1, old_log_probs_batch2,
            rewards_batch1, rewards_batch2,
            returns_batch1, returns_batch2,
            old_values_batch1, old_values_batch2,
            hidden_states_batch1, hidden_states_batch2,
        ) in tqdm(dl, desc="Policy phase"):

            # avoid interferences!
            self.agent1.reset(cond_scale=self.cond_scale)
            self.agent2.reset(cond_scale=self.cond_scale)
            self.initial_agent1.reset(cond_scale=self.cond_scale)
            self.initial_agent2.reset(cond_scale=self.cond_scale)

            
            # Prepare the hidden states
            for i in range(len(hidden_states_batch1)):
                hidden_states_batch1[i][0] = hidden_states_batch1[i][0].squeeze(1)
                hidden_states_batch1[i][1][0] = hidden_states_batch1[i][1][0].squeeze(1)
                hidden_states_batch1[i][1][1] = hidden_states_batch1[i][1][1].squeeze(1)
            
            for i in range(len(hidden_states_batch2)):
                hidden_states_batch2[i][0] = hidden_states_batch2[i][0].squeeze(1)
                hidden_states_batch2[i][1][0] = hidden_states_batch2[i][1][0].squeeze(1)
                hidden_states_batch2[i][1][1] = hidden_states_batch2[i][1][1].squeeze(1)
                
            
            agent_obs_batch1 = tree_map(lambda x: x.squeeze(1), agent_obs_batch1)
            agent_obs_batch2 = tree_map(lambda x: x.squeeze(1), agent_obs_batch2)
            
            
            dummy_firsts = th.from_numpy(np.full(len(agent_obs_batch1["img"]), False)).to(self.device)
            
            
            # Run the policy to get the ACTUAL distribution and value function
            (
                pi_distribution1, values1, _
            ) = self.agent1.policy.get_output_for_observation(
                agent_obs_batch1, hidden_states_batch1, dummy_firsts
            )
            (
                pi_distribution2, values2, _
            ) = self.agent2.policy.get_output_for_observation(
                agent_obs_batch2, hidden_states_batch2, dummy_firsts
            )
            
            # Calculate the log probabilities of old actions w new policy
            log_probs1 = self.agent1.policy.get_logprob_of_action(
                pd      = pi_distribution1, 
                action  = actions_batch1
            )
            log_probs2 = self.agent2.policy.get_logprob_of_action(
                pd      = pi_distribution2, 
                action  = actions_batch2
            )

            # Calculate the KL divergence
            with th.no_grad():
                with th.cuda.amp.autocast():
                    initial_pi_distribution1, _, _ = self.initial_agent1.policy.get_output_for_observation(
                        agent_obs_batch1, hidden_states_batch1, dummy_firsts
                    )
                    initial_pi_distribution2, _, _ = self.initial_agent2.policy.get_output_for_observation(
                        agent_obs_batch2, hidden_states_batch2, dummy_firsts
                    )

            kl_div1 = self.agent1.policy.get_kl_of_action_dists(
                initial_pi_distribution1,
                pi_distribution1,
            )
            kl_div2 = self.agent2.policy.get_kl_of_action_dists(
                initial_pi_distribution2,
                pi_distribution2,
            )
            
            
            
            # Calculate the ratios and advantages
            ratios1 = (log_probs1 - old_log_probs_batch1).exp().to(self.device)
            ratios2 = (log_probs2 - old_log_probs_batch2).exp().to(self.device)
            
            
            advantages_batch1 = (returns_batch1 - old_values_batch1.detach()).to(self.device)
            advantages_batch2 = (returns_batch2 - old_values_batch2.detach()).to(self.device)
             
            # Calculate the policy losses
            surr11 = ratios1 * advantages_batch1
            surr12 = ratios1.clamp(1 - self.clip, 1 + self.clip) * advantages_batch1
            surr21 = ratios2 * advantages_batch2
            surr22 = ratios2.clamp(1 - self.clip, 1 + self.clip) * advantages_batch2


            # Calculate the policy losses with KL divergence
            policy_loss1 = -th.min(surr11, surr12).mean() + self.kl_beta * kl_div1.mean()
            policy_loss2 = -th.min(surr21, surr22).mean() + self.kl_beta * kl_div2.mean()
            
            
            # Calculate the regularization terms (L_v but whatever)
            value_loss1 = self.beta_v * (values1.squeeze(-1) - returns_batch1).pow(2).mean()
            value_loss2 = self.beta_v * (values2.squeeze(-1) - returns_batch2).pow(2).mean()
            

            # Optimize agent 1
            self.optimizer1.zero_grad()

                # (a) actor first,  encoder receives grad
            policy_loss1.backward(retain_graph=True)

                # (b) temporarily freeze encoder
            for p in self._enc_params1:
                p.requires_grad_(False)
            value_loss1.backward()
            for p in self._enc_params1:
                p.requires_grad_(True)
                
            th.nn.utils.clip_grad_norm_(self.trainable_parameters1, self.max_grad_norm)
            self.optimizer1.step()

            # Optimize agent 2
            self.optimizer2.zero_grad()
            policy_loss2.backward(retain_graph=True)
            for p in self._enc_params2:
                p.requires_grad_(False)
            value_loss2.backward()
            for p in self._enc_params2:
                p.requires_grad_(True)
            th.nn.utils.clip_grad_norm_(self.trainable_parameters2, self.max_grad_norm)
            self.optimizer2.step()
            
           
            total_loss1 = policy_loss1 + value_loss1
            total_loss2 = policy_loss2 + value_loss2
            # After each iteration, update the learning rate schedulers
            self.scheduler1.step(total_loss1.mean())
            self.scheduler2.step(total_loss2.mean())
            
            
            # Tensorboard logging
            self.writer.add_scalar(
                "[Awake] Policy Loss/Agent 1",
                policy_loss1.mean().item(),
                self.wake_updates,
            )
            self.writer.add_scalar(
                "[Awake] Policy Loss/Agent 2",
                policy_loss2.mean().item(),
                self.wake_updates,
            )
            self.writer.add_scalar(
                "[Awake] Total Loss/Agent 1",
                total_loss1.mean().item(),
                self.wake_updates,
            )
            self.writer.add_scalar(
                "[Awake] Total Loss/Agent 2",
                total_loss2.mean().item(),
                self.wake_updates,
            )
            
            self.wake_updates += 1
            
            
            del agent_obs_batch1, agent_obs_batch2
            del actions_batch1, actions_batch2
            del old_log_probs_batch1, old_log_probs_batch2
            del rewards_batch1, rewards_batch2
            del old_values_batch1, old_values_batch2
            del hidden_states_batch1, hidden_states_batch2

        del dl
        del hidden_states1  
        del hidden_states2
        del agent_observations1
        del agent_observations2
        del actions1
        del actions2
        del old_log_probs1
        del old_log_probs2
        del rewards1
        del rewards2
        del masks1
        del masks2
        del values1
        del values2
        del advantages_batch1
        del advantages_batch2
        del ratios1
        del ratios2
        del log_probs1
        del log_probs2
        # del entropy1
        # del entropy2
        del kl_div1
        del kl_div2
        del old_values1
        del old_values2
        del pi_distribution1
        del pi_distribution2
        
        gc.collect()
        
            
    def optimize_value_function(self):
        
        # off-policy
        batch1 = self.sample_minibatch(self.off_policy_buffer1)
        batch2 = self.sample_minibatch(self.off_policy_buffer2)
    
        # Prepare the data
        agent_observations1 = [memory.agent_obs for memory in batch1]
        rewards1            = [memory.reward for memory in batch1]
        old_values1         = [memory.value for memory in batch1]
        hidden_states1      = [memory.hidden_state for memory in batch1]
        pi_distributions1   = [memory.pi_distrib for memory in batch1]
        masks1              = [1 - float(memory.done) for memory in batch1]
        
        
        agent_observations2 = [memory.agent_obs for memory in batch2]
        rewards2            = [memory.reward for memory in batch2]
        old_values2         = [memory.value for memory in batch2]
        hidden_states2      = [memory.hidden_state for memory in batch2]
        pi_distributions2   = [memory.pi_distrib for memory in batch2]
        masks2              = [1 - float(memory.done) for memory in batch2]
        
        next_value1 = batch1[-1].next_value
        old_values1 = old_values1 + [next_value1]
        
        next_value2 = batch2[-1].next_value
        old_values2 = old_values2 + [next_value2]
        
        # pre-computed GAEs!!
        returns1 = [m.targ_return for m in batch1]
        returns2 = [m.targ_return for m in batch2]
        
        # Convert to tensors and move to device
        rewards1 = th.tensor(rewards1).float().to(self.device)
        old_values1 = th.tensor(old_values1[:-1]).to(self.device)
        
        rewards2 = th.tensor(rewards2).float().to(self.device)
        old_values2 = th.tensor(old_values2[:-1]).to(self.device)     
        
        
        returns1 = th.tensor(returns1).to(self.device)
        returns2 = th.tensor(returns2).to(self.device)
        
        
        # Prepare the data for the dataloader
        data = [
            agent_observations1, agent_observations2,
            rewards1, rewards2,
            old_values1, old_values2,
            hidden_states1, hidden_states2,
            pi_distributions1, pi_distributions2,
            returns1, returns2
        ]
        
        # Create the dataloader
        dl = create_dataloader(data, batch_size=self.minibatch_size)
        
        # Optimize value function
        for (
            agent_obs_batch1, agent_obs_batch2,
            rewards_batch1, rewards_batch2,
            old_values_batch1, old_values_batch2,
            hidden_states_batch1, hidden_states_batch2,
            pi_distribution_batch1, pi_distribution_batch2,
            returns_batch1, returns_batch2,
        ) in tqdm(dl, desc="Distillation phase"):

            # avoid interferences!
            self.agent1.reset(cond_scale=self.cond_scale)
            self.agent2.reset(cond_scale=self.cond_scale)
            self.initial_agent1.reset(cond_scale=self.cond_scale)
            self.initial_agent2.reset(cond_scale=self.cond_scale)
            
            # Prepare the hidden states
            for i in range(len(hidden_states_batch1)):
                hidden_states_batch1[i][0] = hidden_states_batch1[i][0].squeeze(1)
                hidden_states_batch1[i][1][0] = hidden_states_batch1[i][1][0].squeeze(1)
                hidden_states_batch1[i][1][1] = hidden_states_batch1[i][1][1].squeeze(1)
            
            for i in range(len(hidden_states_batch2)):
                hidden_states_batch2[i][0] = hidden_states_batch2[i][0].squeeze(1)
                hidden_states_batch2[i][1][0] = hidden_states_batch2[i][1][0].squeeze(1)
                hidden_states_batch2[i][1][1] = hidden_states_batch2[i][1][1].squeeze(1)

            # Prepare the observations
            agent_obs_batch1 = tree_map(lambda x: x.squeeze(1), agent_obs_batch1)
            agent_obs_batch2 = tree_map(lambda x: x.squeeze(1), agent_obs_batch2)
            
            dummy_firsts = th.from_numpy(np.full(len(agent_obs_batch1["img"]), False)).to(self.device)
            
            
            # Run the value function
            with th.cuda.amp.autocast():
                values1, _ = self.agent1.policy.v(
                    obs     = agent_obs_batch1, 
                    state_in= hidden_states_batch1, 
                    first   = dummy_firsts
                )
                values2, _ = self.agent2.policy.v(
                    obs     = agent_obs_batch2, 
                    state_in= hidden_states_batch2, 
                    first   = dummy_firsts
                )
            
            # Calculate the value losses
            value_loss1 = clipped_value_loss(
                values1,
                returns_batch1,
                old_values_batch1,
                self.value_clip,
            )
            value_loss2 = clipped_value_loss(
                values2,
                returns_batch2,
                old_values_batch2,
                self.value_clip,
            )
            
            # Calculate the KL divergence regularization terms
            with th.no_grad():
                pd1_old, _, _ = self.initial_agent1.policy.get_output_for_observation(
                    agent_obs_batch1, hidden_states_batch1, dummy_firsts)
                pd2_old, _, _ = self.initial_agent2.policy.get_output_for_observation(
                    agent_obs_batch2, hidden_states_batch2, dummy_firsts)

            pd1_new, _, _ = self.agent1.policy.get_output_for_observation(
                agent_obs_batch1, hidden_states_batch1, dummy_firsts)
            pd2_new, _, _ = self.agent2.policy.get_output_for_observation(
                agent_obs_batch2, hidden_states_batch2, dummy_firsts)

            Rpi1 = self.agent1.policy.get_kl_of_action_dists(pd1_old, pd1_new)
            Rpi2 = self.agent2.policy.get_kl_of_action_dists(pd2_old, pd2_new)

            total_loss1 = value_loss1 + self.beta_clone * Rpi1
            total_loss2 = value_loss2 + self.beta_clone * Rpi2
            # Optimize agent 1
            self.optimizer1.zero_grad()
            total_loss1.mean().backward()
            th.nn.utils.clip_grad_norm_(
                self.trainable_parameters1,
                self.max_grad_norm,
            )
            self.optimizer1.step()

            # Optimize agent 2
            self.optimizer2.zero_grad()
            total_loss2.mean().backward()
            th.nn.utils.clip_grad_norm_(
                self.trainable_parameters2,
                self.max_grad_norm,
            )
            self.optimizer2.step()
            
            # After each iteration, update the learning rate schedulers
            self.scheduler1.step(total_loss1.mean())
            self.scheduler2.step(total_loss2.mean())
            
            # Tensorboard logging
            self.writer.add_scalar(
                "[Sleep] Value Loss/Agent 1",
                value_loss1.mean().item(),
                self.sleep_updates,
            )
            self.writer.add_scalar(
                "[Sleep] Value Loss/Agent 2",
                value_loss2.mean().item(),
                self.sleep_updates,
            )
            
            self.writer.add_scalar(
                "[Sleep] Total Loss/Agent 1",
                total_loss1.mean().item(),
                self.sleep_updates,
            )
            self.writer.add_scalar(
                "[Sleep] Total Loss/Agent 2",
                total_loss2.mean().item(),
                self.sleep_updates,
            )
            
            self.sleep_updates += 1
            
            del agent_obs_batch1, agent_obs_batch2
            del rewards_batch1, rewards_batch2
            del old_values_batch1, old_values_batch2
            del hidden_states_batch1, hidden_states_batch2
            del pi_distribution_batch1, pi_distribution_batch2
            
        del dl
        del hidden_states1
        del hidden_states2
        del agent_observations1
        del agent_observations2
        del rewards1
        del rewards2
        del old_values1
        del old_values2
        del pi_distributions1
        del pi_distributions2
                    

if __name__ == "__main__":
    VPT_MODEL_PATH = "multiagent_training/models/base_2x.model"
    VPT_MODEL_WEIGHTS = "multiagent_training/weights/base_steve1.weights"
    ENV = "multiagent:treasurehunt"
    DEVICE = "cuda"
    device = th.device(DEVICE if th.cuda.is_available() else "cpu")
    
    ppg = PPG(
        ENV,
        VPT_MODEL_PATH,
        VPT_MODEL_WEIGHTS,
        f"multiagent_training/weights/{ENV}-ppg.weights",
        device,
        epochs_wake     = 1,
        epochs_sleep    = 2,
        minibatch_size  = 32,
        max_grad_norm   = 5,
        lr              =   1e-5,
        betas           = (0.9, 0.999),
        gamma           = 0.99,
        lam             = 0.95,
        clip            = 0.2,
        value_clip      = 0.2,
        kl_beta         = 0.5,
        p_decay         = 0.9995,
        beta_clone      = 1.0,
        Toff            = 5,
        Tfreq           = 8,
        Nenv            = 1,
        M               = 400,
        Kpi             = 4,
        Kv              = 1,
        beta_v          = 1,
        freeze_layers   = True,
    )
    ppg.train(
        n_iterations=500,
        n_rollouts=1,
    )