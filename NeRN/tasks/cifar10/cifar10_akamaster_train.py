import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from NeRN.models.regularization import CosineSmoothness, L2Smoothness
from NeRN.tasks.cifar10 import cifar_resnet

model_names = sorted(name for name in cifar_resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(cifar_resnet.__dict__[name]))


parser = argparse.ArgumentParser(description='Proper ResNets for CIFAR10 in pytorch')
parser.add_argument('--exp-name', type=str, required=True,
                    help='Name of the experiment. Will be the name of output model file.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--smoothness-type', type=str, default="Cosine",
                    help='Smoothness regularization, can be Cosine/L2')
parser.add_argument('--cosine-smoothness-factor', type=float, default=1e-4,
                    help='Factor for the cosine smoothness regularization term')
parser.add_argument('--l2-smoothness-factor', type=float, default=None,
                    help='Factor for the l2 smoothness regularization term. If none, taking the value of the cosine smoothness factor')
parser.add_argument('--basic-block-option', type=str, default='A',
                    help='Basic block option, can be A/cifar10')
parser.add_argument('--cifar_100', action='store_true',
                    help='Train CIFAR100 instead of CIFAR10')
best_prec1 = 0


class MyDataParallel(torch.nn.DataParallel):
    def __init__(self, module):
        super(MyDataParallel, self).__init__(module)
        self.module = module

    def get_learnable_weights(self):
        return self.module.get_learnable_weights()


def main():
    global args, best_prec1
    args = parser.parse_args()

    num_classes = 100 if args.cifar_100 else 10
    model = MyDataParallel(cifar_resnet.__dict__[args.arch](basic_block_option=args.basic_block_option,
                                                            num_classes=num_classes))
    model.cuda()

    if args.smoothness_type is None:
        smoothness = None
    elif args.smoothness_type == "Cosine":
        smoothness = CosineSmoothness()
    elif args.smoothness_type == "L2":
        smoothness = L2Smoothness()
    else:
        raise ValueError("Unexpected smoothness type")

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.module.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}".format(args.evaluate))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    get_dataloaders_fn = get_cifar100_dataloaders if args.cifar_100 else get_dataloaders
    print(f"Loading {'cifar10' if not args.cifar_100 else 'cifar100'} data")
    train_loader, val_loader = get_dataloaders_fn({'batch_size': args.batch_size, 'num_workers': args.workers}, {'num_workers': args.workers}, use_workers=True)


    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print(f"At epoch {epoch} of experiment {args.exp_name}")
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, smoothness)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if is_best:
            save(model, 'best')

    save(model, 'last')


def save(model, suffix=""):
    model_kwargs = dict(arch=args.arch)
    model_kwargs.update({
        "smoothness_type": args.smoothness_type,
        "cosine_smoothness_factor": args.cosine_smoothness_factor,
        "l2_smoothness_factor": args.l2_smoothness_factor,
        "basic_block_option": args.basic_block_option,
        "num_classes": 100 if args.cifar_100 else 10
    })
    if args.save_model:
        save_dir = 'trained_models/original_tasks/cifar'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.module.state_dict(), save_dir + "/" + args.exp_name + ('.pt' if not suffix else f'_{suffix}.pt'))
        with open(save_dir + '/' + args.exp_name + ('.json' if not suffix else f'_{suffix}.json'), 'w') as model_save_path:
            json.dump(model_kwargs, model_save_path)


def get_dataloaders(train_kwargs, test_kwargs, use_workers=True, **kwargs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=train_kwargs["batch_size"], shuffle=True,
        num_workers=train_kwargs["num_workers"] if use_workers else 0, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=test_kwargs["num_workers"] if use_workers else 0, pin_memory=True)
    return train_loader, val_loader


def get_cifar100_dataloaders(train_kwargs, test_kwargs, use_workers=True, **kwargs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=train_kwargs["batch_size"], shuffle=True,
        num_workers=train_kwargs["num_workers"] if use_workers else 0, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=test_kwargs["num_workers"] if use_workers else 0, pin_memory=True)
    return train_loader, val_loader


def train(train_loader, model, criterion, optimizer, epoch, smoothness):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cosine_smoothness_losses = AverageMeter()
    l2_smoothness_losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        if smoothness is None:
            l2_smoothness_loss = 0
            cosine_smoothness_loss = 0
        else:
            cosine_smoothness, l2_smoothness = smoothness(model)
            cosine_smoothness_loss = -cosine_smoothness
            l2_smoothness_loss = -l2_smoothness

        # compute output
        output = model(input_var)
        l2_smoothness_factor = args.l2_smoothness_factor if args.l2_smoothness_factor is not None else args.cosine_smoothness_factor
        loss = criterion(output,
                         target_var) + l2_smoothness_factor * l2_smoothness_loss + args.cosine_smoothness_factor * cosine_smoothness_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        cosine_smoothness_loss = cosine_smoothness_loss.float()
        l2_smoothness_loss = l2_smoothness_loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        cosine_smoothness_losses.update(cosine_smoothness_loss.item(), input.size(0))
        l2_smoothness_losses.update(l2_smoothness_loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            print('L2 Smoothness {:.4f}, Cosine Smoothness {:.4f}'.format(l2_smoothness_losses.val,
                                                                          cosine_smoothness_losses.val))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
