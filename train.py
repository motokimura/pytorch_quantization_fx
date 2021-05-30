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
from utils import (configure_cudnn, configure_wandb, get_model,
                   load_checkpoint, prepare_dataloaders, save_checkpoint,
                   set_seed, test, train)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_id', type=int)
    parser.add_argument('--model',
                        choices=['mobilenetv2'],
                        default='mobilenetv2')
    parser.add_argument(
        '--mode',
        choices=['normal', 'qat'],
        # normal: training w/o quantization
        # qat: quantization-aware-training
        default='normal')

    parser.add_argument('--replace_relu', action='store_true')
    parser.add_argument('--fuse_model', action='store_true')
    parser.add_argument('--quantization_backend',
                        choices=['qnnpack', 'fbgemm'],
                        default='fbgemm')
    parser.add_argument('--pretrained', default='imagenet')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr_drop_epochs',
                        type=int,
                        nargs='+',
                        default=[210, 270])
    parser.add_argument('--lr', type=float, default=0.005)  # lower lr (e.g., 5e-4) is sutable for qat
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--model_dir', default='models')
    return parser.parse_args()


def main():
    args = parse_arg()

    torch.backends.quantized.engine = args.quantization_backend

    enable_qat = args.mode == 'qat'

    # fix random seed
    set_seed(args.seed)
    configure_cudnn(deterministic=True, benchmark=False)

    exp_id = f'exp_{args.exp_id:04d}'
    exp_dir = os.path.join(args.model_dir, exp_id)

    # prepare directory to save model
    if args.resume:
        assert os.path.exists(
            os.path.join(exp_dir, 'checkpoint_latest.pth')
        ), 'Failed to resume training. Cannot find checkpoint file.'
    else:
        os.makedirs(exp_dir, exist_ok=False)

    # dump config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
    with open(os.path.join(exp_dir, f'{time_str}.json'), mode='w') as f:
        json.dump(args.__dict__, f, indent=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.device is not None:
        device = torch.device(args.device)
    print(f'device: {device}')

    print('Preparing dataset...')
    train_dataloader, test_dataloader = prepare_dataloaders(args.batch_size)

    print('Preparing model...')
    model = get_model(args.model,
                      pretrained=args.pretrained,
                      replace_relu=args.replace_relu,
                      fuse_model=args.fuse_model)
    if enable_qat:
        model.qconfig = torch.quantization.get_default_qat_qconfig(
            args.quantization_backend)
        torch.quantization.prepare_qat(model, inplace=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.lr_drop_epochs,
                                               gamma=0.1)
    start_epoch = 0
    best_accuracy = -1
    best_accuracy_epoch = -1

    if args.resume:
        model, optimizer, scheduler, start_epoch, best_accuracy, best_accuracy_epoch = load_checkpoint(
            os.path.join(exp_dir, 'checkpoint_latest.pth'), model, optimizer,
            scheduler, start_epoch, best_accuracy, best_accuracy_epoch)
        start_epoch += 1

    configure_wandb(project='pytorch_model_quantization',
                    group=exp_id,
                    config=args)

    # train loop
    for epoch in range(start_epoch,
                       args.epochs):  # loop over the dataset multiple times
        logs = {'epoch': epoch}

        lr = scheduler.get_last_lr()[0]
        print(f'\nEpoch: {epoch} / {args.epochs}, lr: {lr:.9f}')
        logs['lr'] = lr

        # train
        loss_epoch = train(model, optimizer, scheduler, criterion, device,
                           train_dataloader)
        print('loss: %.3f' % loss_epoch)
        logs['train/loss'] = loss_epoch

        # test
        accuracy = test(model, device, test_dataloader)
        print('accuracy: %.4f' % accuracy)
        logs['test/accuracy'] = accuracy

        if enable_qat:
            # test with quantized model
            print('Evaluating quantized model...')
            model_quantized = copy.deepcopy(model)
            model_quantized.to(torch.device('cpu'))
            model_quantized = torch.quantization.convert(
                model_quantized.eval(), inplace=False)
            accuracy = test(model_quantized, torch.device('cpu'),
                            test_dataloader)
            print('accuracy (quantized): %.4f' % accuracy)
            logs['test/accuracy'] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_epoch = epoch

            print('Best accuracy updated. Saving models...')
            model_path = os.path.join(exp_dir, 'best_model.pth')
            model.to(torch.device('cpu'))
            torch.save(model.state_dict(), model_path)

            model.to(device)  # back model from cpu to `device`

        # save checkpoint
        save_checkpoint(os.path.join(exp_dir, 'checkpoint_latest.pth'), model,
                        optimizer, scheduler, epoch, best_accuracy,
                        best_accuracy_epoch)

        logs['test/best_accuracy'] = best_accuracy
        logs['test/best_accuracy_epoch'] = best_accuracy_epoch

        wandb.log(logs)

    wandb.finish()

    print('Reached best accuract %.4f at epoch %d' %
          (best_accuracy, best_accuracy_epoch))


if __name__ == '__main__':
    main()
