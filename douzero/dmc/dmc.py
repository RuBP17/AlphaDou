import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model

from .utils import get_batch, log, create_env, create_optimizers, act

mean_episode_return_buf = {p: deque(maxlen=50) for p in
                           ['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down']}


def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss


def compute_loss_(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2)
    return loss


def compute_loss_bid(logits, targets):
    loss = ((logits - targets) ** 2).mean()
    return loss


def learn(position, actor_models, model, batch, optimizer, flags, lock):
    """Performs a learning (optimization) step."""
    print("Learn", position)
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')
    obs_x = batch["obs_x_batch"]
    obs_x = torch.flatten(obs_x, 0, 1).to(device)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target_adp = torch.flatten(batch['target_adp'].to(device), 0, 1)
    target_wp = torch.flatten(batch['target_wp'].to(device), 0, 1)
    target_wp_bid = torch.flatten(batch['target_wp_bid'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    if len(episode_returns) > 0:
        mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))
    with lock:
        win_rate, win, lose = model.forward(obs_z, obs_x, return_value=True)['values']

        if position in ["landlord", "landlord_up", "landlord_down"]:
            loss1 = compute_loss(win_rate, target_wp)
            l_w = compute_loss_(win, target_adp) * (1. + target_wp) / 2.
            l_l = compute_loss_(lose, target_adp) * (1. - target_wp) / 2.
            loss2 = l_w.mean() + l_l.mean()
            loss = loss1 + loss2
        else:
            loss1 = compute_loss_bid(win_rate, target_wp_bid)
            l_w = compute_loss_(win, target_adp) * torch.abs(target_wp) * (1. + target_wp) / 2.
            l_l = compute_loss_(lose, target_adp) * torch.abs(target_wp) * (1. - target_wp) / 2.
            loss2 = l_w.mean() + l_l.mean()
            loss = loss1 + loss2

        stats = {
            'mean_episode_return_' + position: torch.mean(
                torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_' + position: loss.item(),
        }
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats


def train(flags):
    """
    This is the main funtion for training. It will first
    initilize everything, such as buffers, optimizers, etc.
    Then it will start subprocesses as actors. Then, it will call
    learning function with  multiple threads.
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(
            flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # Initialize actor models
    models = {}
    for device in device_iterator:
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    batch_queues = {}
    for device in device_iterator:
        batch_queue = {"first": ctx.SimpleQueue(), "second": ctx.SimpleQueue(), "third": ctx.SimpleQueue(),
                       "landlord": ctx.SimpleQueue(), "landlord_up": ctx.SimpleQueue(),
                       "landlord_down": ctx.SimpleQueue()}
        batch_queues[device] = batch_queue

    # Learner model for training
    learner_model = Model(device=flags.training_device)

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)

    # Stat Keys
    stat_keys = [
        'mean_episode_return_first',
        'loss_first',
        'mean_episode_return_second',
        'loss_second',
        'mean_episode_return_third',
        'loss_third',
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'first': 0, 'second': 0, 'third': 0, 'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath,
            map_location=("cuda:" + str(flags.training_device) if flags.training_device != "cpu" else "cpu")
        )

        for k in ['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down']:
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            for device in device_iterator:
                models[device].get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(batch_queues[device][position], position, flags, local_lock)
            _stats = learn(position, models, learner_model.get_model(position), batch,
                           optimizers[position], flags, position_lock)
            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {'first': threading.Lock(), 'second': threading.Lock(), 'third': threading.Lock(),
                         'landlord': threading.Lock(), 'landlord_up': threading.Lock(),
                         'landlord_down': threading.Lock()}
    position_locks = {'first': threading.Lock(), 'second': threading.Lock(), 'third': threading.Lock(),
                      'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down']:
                thread = threading.Thread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i,
                    args=(i, device, position, locks[device][position], position_locks[position]))
                thread.start()
                threads.append(thread)

    # Starting actor processes
    for device in device_iterator:
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, batch_queues[device], models[device], flags))
            actor.start()
            actor_processes.append(actor)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # Save the weights for evaluation purpose
        for position in ['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position + '_' + str(frames) + '.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {k: (position_frames[k] - position_start_frames[k]) / (end_time - start_time) for k in
                            position_frames}
            log.info('After %i (L:%i U:%i D:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                     frames,
                     position_frames['landlord'],
                     position_frames['landlord_up'],
                     position_frames['landlord_down'],
                     fps,
                     fps_avg,
                     position_fps['landlord'],
                     position_fps['landlord_up'],
                     position_fps['landlord_down'],
                     pprint.pformat(stats))

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
