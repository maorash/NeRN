from collections import OrderedDict
import time
from contextlib import suppress

import torch

from timm import utils
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_dataset
from timm.data.loader import create_loader
from timm.data.mixup import FastCollateMixup

eval_args = {
    "batch_size": 128,
    "prefetcher": True,
    "num_workers": 4,
    "pin_mem": False,
    "distributed": False,  # If turned on, set world_size & local_rank accordingly
    "local_rank": 0,
    "world_size": 1,
    "channels_last": False,
    "tta": 0,
    "log_interval": 40
}

train_args = {
    "batch_size": 128,
    "prefetcher": True,
    "no_aug": False,
    "reprob": 0.0,
    "remode": "pixel",
    "recount": 1,
    "resplit": False,
    "scale": [0.08, 1.0],
    "ratio": [3. / 4., 4. / 3.],
    "hflip": 0.5,
    "vflip": 0,
    "color_jitter": 0.4,
    "aa": "rand-m7-mstd0.5-inc1",
    "aug_repeats": 0,
    "num_aug_splits": 0,
    "num_workers": 4,
    "distributed": False,  # If turned on, set aug_repeats to 3 and world_size & locaL_rank accordingly
    "local_rank": 0,
    "world_size": 1,
    "pin_mem": False,
    "use_multi_epochs_loader": False,
    "worker_seeding": "all",
    "channels_last": False
}

mixup_args = dict(
    mixup_alpha=0.1, cutmix_alpha=1, cutmix_minmax=None,
    prob=1, switch_prob=0.5, mode='batch',
    label_smoothing=0, num_classes=1000)


def get_dataloaders(train_kwargs, test_kwargs, task_cfg):
    print("Creating ImageNet train dataloader")
    dataset_train = create_dataset(
        '', root=task_cfg.imagenet_path, split='train', is_training=True,
        class_map='', download=False,
        batch_size=train_kwargs['batch_size'], repeats=0)

    print("Creating ImageNet val dataloader")
    dataset_eval = create_dataset(
        '', root=task_cfg.imagenet_path, split='val', is_training=False,
        class_map='', download=False,
        batch_size=test_kwargs['batch_size'])

    if task_cfg.original_model_name in ['ResNet18', 'ResNet14']:
        data_config = {
            'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
            'crop_pct': 0.875, 'interpolation': 'bilinear',
            'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
            'first_conv': 'conv1', 'classifier': 'fc'
        }
    else:
        raise ValueError(f"No data config for model {task_cfg.original_model_name}")

    collate_fn = FastCollateMixup(**mixup_args)

    if test_kwargs is not None:
        eval_args.update(test_kwargs)
    if train_kwargs is not None:
        train_args.update(train_kwargs)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config["input_size"],
        batch_size=eval_args["batch_size"],
        is_training=False,
        use_prefetcher=eval_args["prefetcher"],
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=eval_args["num_workers"],
        distributed=eval_args["distributed"],
        crop_pct=data_config["crop_pct"],
        pin_memory=eval_args["pin_mem"],
    )

    loader_train = create_loader(
        dataset_train,
        input_size=data_config["input_size"],
        batch_size=train_args["batch_size"],
        is_training=True,
        use_prefetcher=train_args["prefetcher"],
        no_aug=train_args["no_aug"],
        re_prob=train_args["reprob"],
        re_mode=train_args["remode"],
        re_count=train_args["recount"],
        re_split=train_args["resplit"],
        scale=train_args["scale"],
        ratio=train_args["ratio"],
        hflip=train_args["hflip"],
        vflip=train_args["vflip"],
        color_jitter=train_args["color_jitter"],
        auto_augment=train_args["aa"],
        num_aug_repeats=train_args["aug_repeats"],
        num_aug_splits=train_args["num_aug_splits"],
        interpolation="random",
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=train_args["num_workers"],
        distributed=train_args["distributed"],
        collate_fn=collate_fn,
        pin_memory=train_args["pin_mem"],
        use_multi_epochs_loader=train_args["use_multi_epochs_loader"],
        worker_seeding=train_args["worker_seeding"],
    )
    return loader_train, loader_eval


def validate(model, loader, loss_fn, amp_autocast=suppress, log_suffix=''):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not eval_args["prefetcher"]:
                input = input.cuda()
                target = target.cuda()
            if eval_args["channels_last"]:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = eval_args["tta"]
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if eval_args["distributed"]:
                reduced_loss = utils.reduce_tensor(loss.data, eval_args["world_size"])
                acc1 = utils.reduce_tensor(acc1, eval_args["world_size"])
                acc5 = utils.reduce_tensor(acc5, eval_args["world_size"])
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if eval_args["local_rank"] == 0 and (last_batch or batch_idx % eval_args["log_interval"] == 0):
                log_name = 'Test' + log_suffix
                print(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('eval_loss', losses_m.avg), ('top1_accuracy', top1_m.avg), ('top5_accuracy', top5_m.avg)])
    print(f"Results - Loss: {losses_m.avg:.5f} Top1: {top1_m.avg:.5f} Top5: {top5_m.avg:.5f}")

    return metrics
