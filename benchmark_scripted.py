import argparse
import time

import torch
from tqdm import tqdm

from utils import configure_cudnn


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='path to model named `scriped_*.pth`')
    parser.add_argument('--n_batch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shape', type=int, nargs=2, default=[224, 224])
    parser.add_argument('--device', default=None)
    return parser.parse_args()


def main():
    args = parse_arg()

    configure_cudnn(deterministic=True, benchmark=False)

    device = torch.device('cpu')  # cpu mode is required to run quantized model
    if args.device is not None:
        device = torch.device(args.device)
    print(f'Device: {device}')

    print('Preparing model...')
    model = torch.jit.load(args.model_path)
    model.to(device)

    batch_shape = [args.batch_size, 3, *args.shape]

    print('Warming up...')
    t = torch.zeros(batch_shape, device=device)
    for _ in tqdm(range(10)):
        model(t)

    print('Running benchmark...')
    start = time.time()
    for _ in tqdm(range(args.n_batch)):
        model(t)
    end = time.time()
    print('latency [sec]: %.4f' % (end - start))
    print('latency [ms/batch]: %.4f' % (end - start / args.n_batch * 1000))

if __name__ == '__main__':
    main()
