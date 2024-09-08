"""
Here, we wrap the original environment to make it easier
to use. When a game is finished, instead of mannualy reseting
the environment, we do it automatically.
"""
import numpy as np
import torch

def _format_observation(obs, device):
    """
    A utility function to process observations and
    move them to CUDA.
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    x_batch = torch.from_numpy(obs['x_batch']).to(device)
    z_batch = torch.from_numpy(obs['z_batch']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'])
    z = torch.from_numpy(obs['z'])
    obs = {'x_batch': x_batch,
           'z_batch': z_batch,
           'legal_actions': obs['legal_actions'],
           }
    return position, obs, x_no_action, z

class Environment:
    def __init__(self, env, device):
        """ Initialzie this environment wrapper
        """
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self, model, device, flags=None):
        obs = self.env.reset(model, device, flags=flags)
        initial_position, initial_obs, x_no_action, z = _format_observation(obs, self.device)
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)
        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
        )

    def step(self, action, model, device, flags=None):
        obs, reward, done, draw, _ = self.env.step(action)

        self.episode_return = reward
        episode_return = self.episode_return
        buf = None
        if done:
            obs = self.env.reset(model, device, flags=flags)
            self.episode_return = torch.zeros(1, 1)
        if draw:
            obs = self.env.reset(model, device, flags=flags)
            self.episode_return = torch.zeros(1, 1)
        position, obs, x_no_action, z = _format_observation(obs, self.device)
        # reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        draw = torch.tensor(draw).view(1, 1)

        if buf is None:
            return position, obs, dict(
                draw=draw,
                done=done,
                episode_return=episode_return,
                obs_x_no_action=x_no_action,
                obs_z=z,
            )
        else:
            return position, obs, dict(
                draw=draw,
                done=done,
                episode_return=episode_return,
                obs_x_no_action=x_no_action,
                obs_z=z,
                begin_buf=buf
            )

    def close(self):
        self.env.close()
