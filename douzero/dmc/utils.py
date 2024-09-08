import typing
import logging
import traceback
import numpy as np
from collections import Counter
import torch
from .env_utils import Environment
from douzero.env import Env

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_env(flags):
    return Env(flags.objective)


def get_batch(b_queues, position, flags, lock):
    b_queue = b_queues
    buffer = []
    while len(buffer) < flags.batch_size:
        buffer.append(b_queue.get())
    batch = {
        key: torch.stack([m[key] for m in buffer], dim=1)
        for key in ["done", "episode_return", "target_adp", "target_wp",
                    "target_wp_bid", "obs_z", "obs_x_batch"]
    }
    del buffer
    return batch


def create_optimizers(flags, learner_model):
    positions = ['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RAdam(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            eps=flags.epsilon)
        optimizers[position] = optimizer
    return optimizers


def act(i, device, batch_queues, model, flags):
    positions = ['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_adp_buf = {p: [] for p in positions}
        target_wp_buf = {p: [] for p in positions}
        target_wp_bid_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}
        obs_x_batch_buf = {p: [] for p in positions}

        position, obs, env_output = env.initial(model, device, flags=flags)

        while True:
            while True:

                if len(obs['legal_actions']) > 1:
                    with torch.no_grad():
                        agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
                    _action_idx = int(agent_output['action'].cpu().detach().numpy())
                    action = obs['legal_actions'][_action_idx]

                    if position in ['first', 'second', 'third']:
                        obs_z_buf[position].append(
                            torch.vstack((torch.full((1, 54), action[0]), env_output['obs_z'])).float())
                    else:
                        obs_z_buf[position].append(
                            torch.vstack((_cards2tensor(action).unsqueeze(0), env_output['obs_z'])).float())
                else:
                    action = obs['legal_actions'][0]
                    if position in ['first', 'second', 'third']:
                        obs_z_buf[position].append(
                            torch.vstack((torch.full((1, 54), action[0]), env_output['obs_z'])).float())
                    else:
                        obs_z_buf[position].append(
                            torch.vstack((_cards2tensor(action).unsqueeze(0), env_output['obs_z'])).float())

                x_batch = env_output['obs_x_no_action'].float()
                obs_x_batch_buf[position].append(x_batch)
                size[position] += 1
                position, obs, env_output = env.step(action, model, device, flags=flags)

                if env_output['done'] or env_output['draw']:
                    for p in positions:
                        diff = size[p] - len(target_adp_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff - 1)])
                            done_buf[p].append(True)
                            if env_output['draw']:
                                episode_return = 0.
                                wp_return = 0.
                                wp_bid = [0, 0, 1]
                            else:
                                episode_return = env_output['episode_return']["play"][p]
                                wp_return = 1. if episode_return > 0. else -1.
                                wp_bid = [1, 0, 0] if episode_return > 0. else [0, 1, 0]
                            episode_return_buf[p].extend([0.0 for _ in range(diff - 1)])
                            episode_return_buf[p].append(episode_return)
                            target_adp_buf[p].extend([episode_return for _ in range(diff)])
                            target_wp_buf[p].extend([wp_return for _ in range(diff)])
                            target_wp_bid_buf[p].extend([torch.tensor(wp_bid) for _ in range(diff)])
                    break
            for p in positions:
                if size[p] > T:
                    batch_queues[p].put({
                        "done": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in done_buf[p][:T]]),
                        "episode_return": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in episode_return_buf[p][:T]]),
                        "target_adp": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in target_adp_buf[p][:T]]),
                        "target_wp": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in target_wp_buf[p][:T]]),
                        "target_wp_bid": torch.stack(
                            [ndarr.clone().detach() for ndarr in target_wp_bid_buf[p][:T]]),
                        "obs_z": torch.stack([ndarr.clone().detach() for ndarr in obs_z_buf[p][:T]]),
                        "obs_x_batch": torch.stack(
                            [ndarr.clone().detach() for ndarr in obs_x_batch_buf[p][:T]]),
                    })
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_adp_buf[p] = target_adp_buf[p][T:]
                    target_wp_buf[p] = target_wp_buf[p][T:]
                    target_wp_bid_buf[p] = target_wp_bid_buf[p][T:]
                    obs_x_batch_buf[p] = obs_x_batch_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


def _cards2tensor(list_cards):
    """
    Convert a list of integers to the tensor
    representation
    See Figure 2 in https://arxiv.org/pdf/2106.06135.pdf
    """
    if len(list_cards) == 0:
        return torch.zeros(54, dtype=torch.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    matrix = np.concatenate((matrix.flatten('F'), jokers))
    matrix = torch.from_numpy(matrix)
    return matrix
