# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import argparse
import copy
import datetime
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.quantization import get_default_qat_qconfig, quantize_fx

from lib.mobilenetv2 import mobilenet_v2
from lib.utils import (
    configure_cudnn,
    configure_wandb,
    load_checkpoint,
    prepare_dataloaders,
    replace_relu,
    save_checkpoint,
    set_seed,
    test,
    train,
)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", type=int)
    parser.add_argument("--qat", action="store_true")
    parser.add_argument("--backend", choices=["qnnpack", "fbgemm"], default="fbgemm")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr_drop_epochs", type=int, nargs="+", default=[210, 270])
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--model_dir", default="models")
    return parser.parse_args()


def main():
    args = parse_arg()

    # fix random seed
    set_seed(args.seed)
    configure_cudnn(deterministic=True, benchmark=False)

    exp_id = f"exp_{args.exp_id:04d}"
    exp_dir = os.path.join(args.model_dir, exp_id)

    # prepare directory to save model
    if args.resume:
        assert os.path.exists(
            os.path.join(exp_dir, "checkpoint_latest.pth")
        ), "Failed to resume training. Cannot find checkpoint file."
    else:
        os.makedirs(exp_dir, exist_ok=False)

    # dump config
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S")
    with open(os.path.join(exp_dir, f"{time_str}.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    device = torch.device(args.device)
    print(f"device: {device}")

    print("Preparing dataset...")
    train_dataloader, test_dataloader = prepare_dataloaders(args.batch_size)

    print("Preparing model...")
    model = mobilenet_v2()
    if args.qat:
        # replace ReLU6 with ReLU so that we can "fuse" Conv+BN+ReLU modules later
        replace_relu(model)
        # prepare model for qat
        qconfig = {"": get_default_qat_qconfig(args.backend)}
        example_inputs = (torch.randn(1, 3, 32, 32),)
        model = quantize_fx.prepare_qat_fx(model.train(), qconfig, example_inputs)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_epochs, gamma=0.1)
    start_epoch = 0
    best_accuracy = -1
    epoch_at_best_accuracy = -1

    if args.resume:
        model, optimizer, scheduler, start_epoch, best_accuracy, epoch_at_best_accuracy = load_checkpoint(
            os.path.join(exp_dir, "checkpoint_latest.pth"),
            model,
            optimizer,
            scheduler,
            start_epoch,
            best_accuracy,
            epoch_at_best_accuracy,
        )
        start_epoch += 1

    configure_wandb(project="pytorch_quantization_fx", group=exp_id, config=args)

    # train loop
    for epoch in range(start_epoch, args.epochs):
        logs = {"epoch": epoch}

        lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch: {epoch} / {args.epochs}, lr: {lr:.9f}")
        logs["lr"] = lr

        # train
        loss_epoch = train(model, optimizer, scheduler, criterion, device, train_dataloader)
        print("loss: %.8f" % loss_epoch)
        logs["train/loss"] = loss_epoch

        # test
        accuracy = test(model, device, test_dataloader)
        print("accuracy: %.4f" % accuracy)
        logs["test/accuracy"] = accuracy

        if args.qat:
            # test with quantized model
            print("Evaluating quantized model...")
            model_quantized = copy.deepcopy(model)
            model_quantized.to(torch.device("cpu"))
            model_quantized = quantize_fx.convert_fx(model_quantized.eval())
            accuracy = test(model_quantized, torch.device("cpu"), test_dataloader)
            print("accuracy (quantized): %.4f" % accuracy)
            logs["test/accuracy"] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epoch_at_best_accuracy = epoch

            print("Best accuracy updated. Saving model...")
            model_path = os.path.join(exp_dir, "best_model.pth")
            model.to(torch.device("cpu"))
            torch.save(model.state_dict(), model_path)

            model.to(device)  # back model from cpu to `device`

        # save checkpoint
        save_checkpoint(
            os.path.join(exp_dir, "checkpoint_latest.pth"),
            model,
            optimizer,
            scheduler,
            epoch,
            best_accuracy,
            epoch_at_best_accuracy,
        )

        logs["test/best_accuracy"] = best_accuracy
        logs["test/epoch_at_best_accuracy"] = epoch_at_best_accuracy

        wandb.log(logs)

    print("Reached best accuracy %.4f at epoch %d" % (best_accuracy, epoch_at_best_accuracy))

    wandb.finish()


if __name__ == "__main__":
    main()
