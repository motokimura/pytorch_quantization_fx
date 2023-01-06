import argparse
import os

import torch
from tqdm import tqdm

from utils import calibrate_for_ptq, configure_cudnn, get_model, prepare_dataloaders, set_seed, test


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", type=int)
    parser.add_argument("--model", choices=["mobilenetv2"], default="mobilenetv2")
    parser.add_argument(
        "--mode",
        choices=["normal", "ptq", "qat"],
        # normal: evaluation w/o any quantization (requires pretrained model w/ mode='normal')
        # ptq: post-training-quantization (requires pretrained model w/ mode='normal')
        # qat: quantization-aware-training (requires pretrained model w/ mode='qat')
        default="normal",
    )
    parser.add_argument("--replace_relu", action="store_true")
    parser.add_argument("--fuse_model", action="store_true")
    parser.add_argument("--quantization_backend", choices=["qnnpack", "fbgemm"], default="fbgemm")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_calib_batch", type=int, default=32)
    parser.add_argument("--device", default=None)
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

    device = torch.device("cpu")  # cpu mode is required to run quantized model
    if args.device is not None:
        device = torch.device(args.device)
    print(f"device: {device}")

    print("Preparing dataset...")
    train_dataloader, test_dataloader = prepare_dataloaders(args.batch_size)

    print("Preparing model...")
    model_path = os.path.join(exp_dir, "best_model.pth")
    if args.mode == "ptq":
        model = get_model(
            args.model,
            pretrained=model_path,  # load weight here in ptq mode
            replace_relu=args.replace_relu,
            fuse_model=args.fuse_model,
            eval_before_fuse=True,
        )
        # XXX: arg `pretrained` assumes the model pretrained w/ mode='normal' & fuse_model=False
        model.qconfig = torch.ao.quantization.get_default_qconfig(args.quantization_backend)
        torch.ao.quantization.prepare(model, inplace=True)
        # calibrate
        calibrate_for_ptq(model, train_dataloader, args.n_calib_batch)
        torch.ao.quantization.convert(model.eval(), inplace=True)
    else:
        state_dict = torch.load(model_path)
        model = get_model(
            args.model,
            pretrained=None,
            replace_relu=args.replace_relu,
            fuse_model=args.fuse_model,
            eval_before_fuse=False,
        )
        if args.mode == "normal":
            model.load_state_dict(state_dict)
        elif args.mode == "qat":
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig(args.quantization_backend)
            torch.ao.quantization.prepare_qat(model, inplace=True)
            model.load_state_dict(state_dict)
            torch.ao.quantization.convert(model.eval(), inplace=True)
        else:
            raise ValueError(f"{args.mode} is not supported.")

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
    strs = []
    if args.replace_relu:
        strs.append("-relu")
    if args.fuse_model:
        strs.append("-fused")
    options = "".join(strs)
    scripted_model_path = os.path.join(exp_dir, f"scripted_model_{args.mode}{options}.pth")
    model.to(torch.device("cpu"))
    torch.jit.save(torch.jit.script(model), scripted_model_path)
    print(f"Saved scripted model to {scripted_model_path}")


if __name__ == "__main__":
    main()
