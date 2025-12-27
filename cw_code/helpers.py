import pickle

import gym
import torch
from steve1.MineRLConditionalAgent import MineRLConditionalAgent
import cv2

import subprocess

def load_model_parameters(path_to_model_file):
    with open(path_to_model_file, "rb") as f:
        agent_parameters = pickle.load(f)
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def create_agent(
    policy_kwargs, pi_head_kwargs, in_weights, cond_scale, DEVICE="cuda"
):
    agent = MineRLConditionalAgent(
        device=DEVICE,
        policy_kwargs=policy_kwargs,
        pi_head_kwargs=pi_head_kwargs,
    )
    agent.load_weights(in_weights)
    
    agent.reset(cond_scale=cond_scale)
    return agent


def freeze_policy_layers(policy, trainable_layers):
    for param in policy.parameters():
        param.requires_grad = False
    for layer in trainable_layers:
        for param in layer.parameters():
            param.requires_grad = True
    return [param for layer in trainable_layers for param in layer.parameters()]


def exists(val):
    return val is not None


def normalize(t, eps=1e-5):
    return (t - t.mean()) / (t.std() + eps)


def update_network_(loss, optimizer, policy, max_grad_norm=0.5, retain_graph=True):
    optimizer.zero_grad()
    loss.mean().backward(retain_graph=retain_graph)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()


def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.squeeze() - rewards) ** 2
    value_loss_2 = (values.squeeze() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))


def calculate_gae(rewards, values, masks, gamma, lam):
    """
    Calculate the generalized advantage estimate
    """
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        next_value = values[step + 1] if step < len(rewards) - 1 else 0
        delta = rewards[step] + gamma * next_value * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def minerl_to_minedojo(minerl_action):
    # Initialize the MineDojo action array according to the MultiDiscrete structure
    minedojo_action = [0] * 8  # 8 dimensions as per MultiDiscrete structure

    # Map basic movement actions (forward, back, left, right)
    minedojo_action[0] = 1 if minerl_action['forward'] == 1 else 2 if minerl_action['back'] == 1 else 0
    minedojo_action[1] = 1 if minerl_action['left'] == 1 else 2 if minerl_action['right'] == 1 else 0

    # Map jump, sneak, and sprint
    if minerl_action['jump'] == 1:
        minedojo_action[2] = 1
    elif minerl_action['sneak'] == 1:
        minedojo_action[2] = 2
    elif minerl_action['sprint'] == 1:
        minedojo_action[2] = 3

    # Map camera movement
    camera_pitch, camera_yaw = minerl_action['camera'][0]
    # minedojo_action[3] = convert_camera_value(camera_pitch, 'pitch')
    # minedojo_action[4] = convert_camera_value(camera_yaw, 'yaw')

    # Test: modify OG minedojo code to make it continous!
    minedojo_action[3] = camera_pitch
    minedojo_action[4] = camera_yaw
    
    
    # Map functional actions like use, drop, and attack
    if minerl_action['use'] == 1:
        minedojo_action[5] = 1
    elif minerl_action['drop'] == 1:
        minedojo_action[5] = 2
    elif minerl_action['attack'] == 1:
        minedojo_action[5] = 3
    # Additional functional actions can be mapped here

    # Placeholder for craft argument and equip/place/destroy argument
    # These would require more complex logic based on the specific MineRL action
    minedojo_action[6] = 0  # Placeholder for craft argument
    minedojo_action[7] = 0  # Placeholder for equip/place/destroy argument

    return minedojo_action

def add_text_to_frame(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_color=(255, 255, 255), line_type=cv2.LINE_AA, thickness=2):
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, line_type)
    return frame

def obs_to_device(agent_obs, device):
    """
    Moves all tensors in the agent_obs dictionary to the specified device.
    
    Args:
    agent_obs (dict): Dictionary containing tensors.
    device (torch.device): Target device ('cuda' or 'cpu').
    
    Returns:
    dict: Updated dictionary with tensors moved to the specified device.
    """
       
    for key in agent_obs:
        agent_obs[key] = agent_obs[key].to(device)
    return agent_obs

def hidden_to_device(item, device):
    """
    Recursively moves tensors in a complex nested structure to the specified device.
    
    Args:
    item: A nested structure containing tensors, lists, and tuples.
    device (torch.device): Target device ('cuda' or 'cpu').
    
    Returns:
    The updated structure with all tensors moved to the specified device.
    """
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, tuple):
        return tuple(hidden_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [hidden_to_device(x, device) for x in item]
    else:
        return item  # Return non-tensor item unchanged
    
def kill_java_processes():
    try:
        subprocess.run(['pkill', '-9', 'java'], check=True)
        print("Successfully killed Java processes.")
    except subprocess.CalledProcessError:
        print("Failed to kill Java processes. No Java process may be running.")
    except Exception as e:
        print(f"An error occurred: {e}")

def freeze_policy_layers(policy, trainable_layers):
    for param in policy.parameters():
        param.requires_grad = False
    for layer in trainable_layers:
        for param in layer.parameters():
            param.requires_grad = True
    return [param for layer in trainable_layers for param in layer.parameters()]

def detach_hidden_states(hidden_states):
    """
    Detach the hidden states from the computation graph
    """
    for i in range(len(hidden_states)):
        hidden_states[i] = list(hidden_states[i])
        hidden_states[i][1] = list(hidden_states[i][1])
        hidden_states[i][1][0] = hidden_states[i][1][0].detach()
        hidden_states[i][1][1] = hidden_states[i][1][1].detach()