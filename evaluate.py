import os
import argparse
from douzero.evaluation.simulation import evaluate
import random
import numpy as np
import torch

# 设置 random 模块的随机数种子
random.seed(78419)
# 设置 NumPy 的随机数种子
np.random.seed(78419)
# 设置 PyTorch 的随机数种子
torch.manual_seed(78419)
# 如果你的代码也将在CUDA设备上运行，还需要为所有GPU设置随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(78419)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dou Dizhu Evaluation')

    parser.add_argument('--player_1_bid', type=str, default='Supervised')
    parser.add_argument('--player_2_bid', type=str, default='douzero_checkpoints/douzero_II/second_0.ckpt')
    parser.add_argument('--player_3_bid', type=str, default='random')

    parser.add_argument('--player_1_playcard', type=str, default='baseline/test/landlord.ckpt')
    parser.add_argument('--player_2_playcard', type=str, default='baseline/best/landlord_down.ckpt')
    parser.add_argument('--player_3_playcard', type=str, default='douzero_checkpoints/douzero_II/landlord_up_0.ckpt')

    parser.add_argument('--eval_data', type=str, default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    use_random_bid = False
    use_random_playcard = False

    if use_random_bid:
        args.player_1_bid = 'random'
        args.player_2_bid = 'random'
        args.player_3_bid = 'random'

    if use_random_playcard:
        args.player_1_playcard = 'random'
        args.player_2_playcard = 'random'
        args.player_3_playcard = 'random'

    evaluate(args.player_1_bid,
             args.player_2_bid,
             args.player_3_bid,
             args.player_1_playcard,
             args.player_2_playcard,
             args.player_3_playcard,
             args.eval_data,
             args.num_workers)
