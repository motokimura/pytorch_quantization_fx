import random

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from mobilenetv2 import mobilenet_v2


def set_seed(seed):
    """Set seed for randome number generator.
    Args:
        seed (int): seed for randome number generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def configure_cudnn(deterministic=True, benchmark=False):
    """configure cuDNN.
    Args:
        deterministic (bool) : make cuDNN behavior deterministic if True.
        benchmark (bool) : use cuDNN benchmark function if True.
    """
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch,
                    best_accuracy, best_accuracy_epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_accuracy': best_accuracy,
        'best_accuracy_epoch': best_accuracy_epoch
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch,
                    best_accuracy, best_accuracy_epoch):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    best_accuracy_epoch = checkpoint['best_accuracy_epoch']

    return model, optimizer, scheduler, epoch, best_accuracy, best_accuracy_epoch


def configure_wandb(project=None, group=None, config=None):
    from os import environ

    from dotenv import load_dotenv

    import wandb

    load_dotenv()  # load WANDB_API_KEY from .env file
    assert "WANDB_API_KEY" in environ, '"WANDB_API_KEY" is empty. Create ".env" file with your W&B API key. See ".env.sample" for the file format'

    wandb.init(project=project, group=group, config=config)


def prepare_dataloaders(batch_size):
    # prepara train dataloader
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CIFAR10(root='./data',
                            train=True,
                            download=True,
                            transform=train_transform)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2)

    # prepare test dataloader
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = CIFAR10(root='./data',
                           train=False,
                           download=True,
                           transform=test_transform)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=2)

    return train_dataloader, test_dataloader


def get_model(model_name,
              pretrained='imagenet',
              replace_relu=False,
              fuse_model=False,
              eval_before_fuse=True):
    if model_name == 'mobilenetv2':
        model = mobilenet_v2(pretrained=pretrained,
                             replace_relu=replace_relu,
                             fuse_model=fuse_model,
                             eval_before_fuse=eval_before_fuse)
    # TODO: other models
    else:
        raise ValueError(f'model {model_name} is not supported.')

    return model.eval()


def train(model, optimizer, scheduler, criterion, device, train_dataloader):
    model.train()
    loss_epoch = 0.0
    num_samples = 0
    for data in tqdm(train_dataloader,
                     total=len(train_dataloader),
                     desc='train'):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        loss_epoch += loss.item() * len(data)
        num_samples += len(data)
    # update lr
    scheduler.step()
    loss_epoch /= num_samples
    return loss_epoch


def test(model, device, test_dataloader):
    model.eval()
    num_correct = 0
    num_samples = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(test_dataloader,
                         total=len(test_dataloader),
                         desc='test'):
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            num_samples += labels.size(0)
            num_correct += (predicted == labels).sum().item()
    accuracy = num_correct / num_samples
    return accuracy


def calibrate_for_ptq(model, train_dataloader, n_calib_batch):
    model.eval()
    model.to(torch.device('cpu'))  # quantization is not yet supported for cuda
    batch_count = 0
    with torch.no_grad():
        for data in tqdm(train_dataloader, total=n_calib_batch, desc='calib'):
            inputs = data[0].to(torch.device('cpu'))
            model(inputs)

            batch_count += 1
            if batch_count > n_calib_batch:
                break
