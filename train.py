import os

from douzero.dmc import parser
from douzero.dmc.dmc import train

if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train(flags)
