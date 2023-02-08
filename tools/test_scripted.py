import argparse

import torch
from tqdm import tqdm

from lib.utils import configure_cudnn, prepare_dataloaders, test


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to model named `scriped_*.pth`")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_arg()

    configure_cudnn(deterministic=True, benchmark=False)

    device = torch.device("cpu")  # cpu mode is required to run quantized model
    if args.device is not None:
        device = torch.device(args.device)
    print(f"device: {device}")

    print("Preparing dataset...")
    _, test_dataloader = prepare_dataloaders(args.batch_size)

    print("Preparing model...")
    model = torch.jit.load(args.model_path)
    model.to(device)

    print("Warming up...")
    t = torch.zeros([1, 3, 32, 32], device=device)
    for _ in tqdm(range(10)):
        model(t)

    print("Running evaluation...")
    accuracy = test(model, device, test_dataloader)
    print("accuracy: %.4f" % accuracy)


if __name__ == "__main__":
    main()
