import argparse
import os

import torch
from torch.quantization import quantize_fx
from tqdm import tqdm

from mobilenetv2 import mobilenet_v2
from utils import calibrate, configure_cudnn, prepare_dataloaders, replace_relu, set_seed, test


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", type=int)
    parser.add_argument(
        "--mode",
        choices=["normal", "ptq"],
        # normal: evaluation w/o quantization
        # ptq: post-training-quantization
        default="normal",
    )
    parser.add_argument("--quantization_backend", choices=["qnnpack", "fbgemm"], default="fbgemm")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_calib_batch", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--model_dir", default="models")
    return parser.parse_args()


def main():
    args = parse_arg()

    torch.backends.quantized.engine = args.quantization_backend

    # fix random seed
    set_seed(args.seed)
    configure_cudnn(deterministic=True, benchmark=False)

    exp_id = f"exp_{args.exp_id:04d}"
    exp_dir = os.path.join(args.model_dir, exp_id)

    if args.mode == "ptq":
        assert args.device == "cpu", "cuda is not supported in quantization with PyTorch"
    device = torch.device(args.device)
    print(f"device: {device}")

    print("Preparing dataset...")
    train_dataloader, test_dataloader = prepare_dataloaders(args.batch_size)

    print("Preparing model...")
    model = mobilenet_v2()
    model_path = os.path.join(exp_dir, "best_model.pth")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    if args.mode == "ptq":
        # replace ReLU6 with ReLU so that we can "fuse" Conv+BN+ReLU modules later
        replace_relu(model)
        # prepare model for ptq
        qconfig_mapping = {"": torch.quantization.get_default_qconfig(args.quantization_backend)}
        example_inputs = (torch.randn(1, 3, 32, 32),)
        model = quantize_fx.prepare_fx(model.eval(), qconfig_mapping, example_inputs)
        # calibrate and convert
        calibrate(model, train_dataloader, args.n_calib_batch)
        model = quantize_fx.convert_fx(model.eval())
    model.to(device)

    print("Warming up...")
    t = torch.zeros([1, 3, 32, 32], device=device)
    for _ in tqdm(range(10)):
        model(t)

    print("Running evaluation...")
    accuracy = test(model, device, test_dataloader)
    print("accuracy: %.4f" % accuracy)

    # save jit scripted model to see how much quantization reduces the model size
    # you can evaluate this scripted model with `test_scripted.py`
    # the accuracy shoule be the same
    scripted_model_path = os.path.join(exp_dir, f"scripted_model_{args.mode}.pth")
    model.to(torch.device("cpu"))
    torch.jit.save(torch.jit.script(model), scripted_model_path)
    print(f"Saved scripted model to {scripted_model_path}")


if __name__ == "__main__":
    main()
