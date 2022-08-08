from __future__ import print_function

import argparse
import json
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from HAND.tasks.resnet18 import ResNet18
from HAND.tasks.resnet14 import ResNet14
from HAND.tasks.simple_net import SimpleNet
from HAND.tasks.vgg8 import VGG8
from HAND.models.regularization import CosineSmoothness, L2Smoothness


def get_dataloaders(test_kwargs, train_kwargs):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, **train_kwargs)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)

    return test_loader, train_loader


def train(args, model, device, train_loader, optimizer, epoch, smoothness):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if smoothness is None:
            smoothness_loss = 0
        else:
            smoothness_loss = - smoothness(model)

        classification_loss = loss_fn(output, target)
        loss = classification_loss + args.smoothness_factor * smoothness_loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            if smoothness is None:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                        loss.item()))
            else:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClassification Loss: {:.6f}\tSmoothness: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                        loss.item(), classification_loss.item(), smoothness_loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Name of the experiment. Will be the name of output model file.')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--num-hidden', type=int, default=None, metavar='N', nargs='+',
                        help='List of hidden channels sizes in SimpleNet')
    parser.add_argument('--num-layers', type=int, default=3, metavar='N',
                        help='number of layers in SimpleNet')
    parser.add_argument('--kernel-sizes', type=int, default=None, metavar='N', nargs='+',
                        help='List of kernel sizes in SimpleNet')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--smoothness-type', type=str, default=None,
                        help='Smoothness regularization, can be Cosine/L2')
    parser.add_argument('--smoothness-factor', type=float, default=1e-2,
                        help='Factor for the smoothness regularization term')
    parser.add_argument('--model_arch', type=str, default="SimpleNet",
                        help='The model architecture, can be SimpleNet/VGG8/ResNet18/ResNet14')

    args = parser.parse_args()
    if args.num_hidden is not None and len(args.num_hidden) != args.num_layers:
        raise ValueError(f"Got num layers = {args.num_layers}, but {len(args.num_hiddens)} hidden sizes")

    if args.kernel_sizes is not None and len(args.kernel_sizes) != args.num_layers:
        raise ValueError(f"Got num layers = {args.num_layers}, but {len(args.kernel_sizes)} kernel sizes")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    test_loader, train_loader = get_dataloaders(test_kwargs, train_kwargs)

    if args.model_arch == "SimpleNet":
        model_kwargs = dict(input_size=32, num_hidden=args.num_hidden, input_channels=3, num_layers=args.num_layers,
                            num_classes=10, kernel_sizes=args.kernel_sizes)
        model = SimpleNet(**model_kwargs).to(device)
    elif args.model_arch == "VGG8":
        model_kwargs = dict(input_size=32, input_channels=3, num_classes=10)
        model = VGG8(**model_kwargs).to(device)
    elif args.model_arch == "ResNet18":
        model_kwargs = dict(num_classes=10)
        model = ResNet18(**model_kwargs).to(device)
    elif args.model_arch == "ResNet14":
        model_kwargs = dict(num_classes=10)
        model = ResNet14(**model_kwargs).to(device)
    else:
        raise ValueError(f"Unknown model architecture {args.model_arch}")
    model_kwargs.update({
        "smoothness_type": args.smoothness_type,
        "smoothness_factor": args.smoothness_factor
    })

    for p in model.parameters():
        if len(p.shape) >= 2:
            torch.nn.init.xavier_normal_(p)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if args.smoothness_type is None:
        smoothness = None
    elif args.smoothness_type == "Cosine":
        smoothness = CosineSmoothness()
    elif args.smoothness_type == "L2":
        smoothness = L2Smoothness()
    else:
        raise ValueError("Unexpected smoothness type")

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, smoothness)
        test(model, device, test_loader)
        scheduler.step()

        if args.save_model:
            save_dir = '../../trained_models/original_tasks/mnist'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), save_dir + "/" + args.exp_name + ".pt")
            with open(save_dir + '/' + args.exp_name + '.json', 'w') as model_save_path:
                json.dump(model_kwargs, model_save_path)


if __name__ == '__main__':
    main()
