from torch import nn
import torch.optim as optim
import os
import time
import torch
import numpy as np
from skimage.transform import rescale, rotate 
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from dataset import BrainSegmentationDataset
import argparse
import utils
import sys

import torch.distributed as dist
import habana_frameworks.torch.core as htcore
# os.environ["PT_HPU_LAZY_MODE"]="1"

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
    
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

def transforms_net(scale=0.995, angle=5):
    transform_list = []

    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))

    return Compose(transform_list)

class Scale(object): 

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample

        img_size = image.shape[0]

        image = rescale(
            image,
            (self.scale, self.scale),
            channel_axis=-1,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        mask = rescale(
            mask,
            (self.scale, self.scale),
            order=0,
            channel_axis=-1,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        diff = (img_size - image.shape[0]) / 2.0
        padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
        image = np.pad(image, padding, mode="constant", constant_values=0)
        mask = np.pad(mask, padding, mode="constant", constant_values=0)
        return image, mask

class Rotate(object):

    def __init__(self, angle):
        self.angle = angle
    
    def __call__(self, sample):
        image, mask = sample

        image = rotate(image, self.angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(
            mask, self.angle, resize=False, order=0, preserve_range=True, mode="constant"
        )
        return image, mask

def dataset_net(args):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    train = BrainSegmentationDataset(
        images_dir=os.path.join(dir_path, f"data/{args.data_path}"),
        subset="train",
        image_size=args.image_size,
        transform=transforms_net(scale=args.aug_scale, angle=args.aug_angle),
        from_cache=args.im_cache,
        cache_dir=os.path.join(dir_path, f"data/{args.cache_path}")
    )
    valid = BrainSegmentationDataset(
        images_dir=os.path.join(dir_path, f"data/{args.data_path}"),
        subset="validation",
        image_size=args.image_size,
        from_cache=args.im_cache,
        cache_dir=os.path.join(dir_path, f"data/{args.cache_path}")
    )
    return train, valid

def data_loaders(args):
    dataset_train, dataset_valid = dataset_net(args)

    def worker_init(worker_id):
        np.random.seed(args.seed)
        np.random.seed(42 + worker_id)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_valid)
    
        loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            sampler = train_sampler,
            shuffle=False,
            drop_last=True,
            num_workers=args.world_size,
            pin_memory=True
            )
        loader_valid = DataLoader(
            dataset_valid,
            batch_size=args.batch_size,
            sampler = valid_sampler,
            drop_last=True,
            num_workers=args.world_size,
            pin_memory=True
            #worker_init_fn=worker_init,
            )
        return loader_train, loader_valid, train_sampler, valid_sampler
    else:
        loader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            # num_workers=args.workers,
            worker_init_fn=worker_init,
        )
        loader_valid = DataLoader(
            dataset_valid,
            batch_size=args.batch_size,
            drop_last=False,
            # num_workers=args.workers,
            worker_init_fn=worker_init,
        )
        return loader_train, loader_valid

def log_loss_summary(loss, step, prefix=""):
    return f"epoch {step} | {prefix}loss: {np.mean(loss)}"

def log_scalar_summary(tag, value, step):
    return f"epoch {step} | {tag}: {value}"

def dsc(y_pred, y_true):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list

def train(args, model, loss_fn, optimizer, loader_train, device, epoch):
    model.train()
    train_time = time.time()
    loss_train = []

    for i, data in enumerate(loader_train):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)
        if args.use_lazy_mode:
            htcore.mark_step() 

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            if args.use_lazy_mode:
                htcore.mark_step() 
            optimizer.step()
            if args.use_lazy_mode:
                htcore.mark_step() 
            loss_train.append(loss.item())
    if args.distributed:
        log(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
            f"performance/{args.run_name}.txt"), 
            f"train,{args.rank},{epoch},{time.time() - train_time},{np.mean(loss_train)},"
        )
    else:
        log(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
            f"performance/{args.run_name}.txt"), 
            f"train,{epoch},{time.time() - train_time},{np.mean(loss_train)},"
        )
    print(log_loss_summary(loss_train, epoch))

def eval(args, model, loss_fn, testloader, device, epoch):
    model.eval()
    eval_time = time.time()

    loss_valid = []
    validation_pred = []
    validation_true = []
    for i, data in enumerate(testloader):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)
        if args.use_lazy_mode:
            htcore.mark_step()

        # with torch.set_grad_enabled(False):
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)

        loss_valid.append(loss.item())
        y_pred_np = y_pred.detach().cpu().numpy()
        validation_pred.extend(
            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
        )
        y_true_np = y_true.detach().cpu().numpy()
        validation_true.extend(
            [y_true_np[s] for s in range(y_true_np.shape[0])]
        )

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"performance/{args.run_name}.txt")
    print(log_loss_summary(loss_valid, epoch, prefix="val_"))
    mean_dsc = np.mean(
        dsc_per_volume(
            validation_pred,
            validation_true,
            testloader.dataset.patient_slice_index,
        )
    )
    if args.distributed:
        log(save_path, f"eval,{args.rank},{epoch},{time.time() - eval_time},{np.mean(loss_valid)}, {mean_dsc}")
    else:
        log(save_path, f"eval,{epoch},{time.time() - eval_time},{np.mean(loss_valid)}, {mean_dsc}")
    print(log_scalar_summary("val_dsc", mean_dsc, epoch))
    # log(save_path, log_scalar_summary("val_dsc", mean_dsc, epoch))
    return mean_dsc

def log(save_path, line):
    with open(save_path, "a") as file:
        file.write(line +"\n")

def main():

    # Get parser arguments, prepare the logging file.
    args = add_parser()
    dir_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(dir_path, f"performance/{args.run_name}.txt")
    log(save_path, f"Command: {sys.argv}")
    log(save_path, f"Begin CSV")
    if args.distributed:
        log(save_path, f"type,card,epoch,time,loss,dsc")
    else:
        log(save_path, f"type,epoch,time,loss,dsc")

    # Set the HPU settings
    if args.use_lazy_mode:
        os.environ["PT_HPU_LAZY_MODE"]="1"
    else:
        os.environ["PT_HPU_LAZY_MODE"]="2"   
    torch.manual_seed(args.seed)
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device("hpu")
    torch.cuda.current_device = lambda: None
    torch.cuda.set_device = lambda x: None
    print(device)
    import habana_frameworks.torch.distributed.hccl
    utils.init_distributed_mode(args)
    
    # Get the data loaders and time their init (useful for cache vs non cache)
    st_timer = time.time()
    if args.distributed:
        loader_train, loader_valid, train_sampler, valid_sampler = data_loaders(args)
        log(save_path, f"loaders_init,,,{time.time() - st_timer},,")
    else:
        loader_train, loader_valid = data_loaders(args)
        log(save_path, f"loaders_init,,{time.time() - st_timer},,")

    # Block here is to allow other versions of the unet. Essentially unused, though.
    if args.layers == 4:
        from models.unet_base import UNet
    model = UNet(in_channels=3, out_channels=1)
    model.to(device)
    
    # Set other hyperparameters and hyper...functions?
    loss_fn = DiceLoss()
    best_validation_dsc = 0.0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    total_train_time = 0
    total_eval_time = 0
    
    # One of the spots the distributed training may be going wrong.
    if args.distributed and args.hpu:
        model = torch.nn.parallel.DistributedDataParallel(model, bucket_cap_mb=100, gradient_as_bucket_view=True)

    # Train and test loop.
    # Almost everywhere epoch is used is log output, hence index by 1
    for epoch in range(1, args.epochs+1):
        print(f"Beginning epoch {epoch}")
        st_timer = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch-1)
        train(args, model, loss_fn, optimizer, loader_train, device, epoch)
        total_train_time += (time.time() - st_timer)

        st_timer = time.time()
        dsc = eval(args, model, loss_fn, loader_valid, device, epoch)
        total_eval_time += (time.time() - st_timer)

        if dsc > best_validation_dsc:
            best_validation_dsc = dsc
            torch.save(model.state_dict(), os.path.join(dir_path, f"weights/{args.weights_file}"))

    # Log all the info not already logged in train/eval
    if not args.distributed:
        print(f"total_train_time = {total_train_time} for {args.epochs} epochs")
        print(f"total_eval_time = {total_eval_time} for {args.epochs} epochs")
        print(f"\nBest validation mean DSC: {best_validation_dsc}\n")
        log(save_path, f"total_train_time,{args.epochs},{total_train_time},,")
        log(save_path, f"total_eval_time,{args.epochs},{total_eval_time},,")
        log(save_path, f"\nBest validation mean DSC: {best_validation_dsc}\n")
        log(save_path, f"End CSV")
    elif args.rank == 0: # Don't both with dsc cause it's broken in distributed atm
        print(f"total_train_time = {total_train_time} for {args.epochs} epochs")
        print(f"total_eval_time = {total_eval_time} for {args.epochs} epochs")
        log(save_path, f"total_train_time,,{args.epochs},{total_train_time},,")
        log(save_path, f"total_eval_time,,{args.epochs},{total_eval_time},,")
        log(save_path, f"End CSV")

def add_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch unet bmri')
    parser.add_argument('--run-name', type=str, default=None,
                        help='name of the run to give the perf file'),
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=2001, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--layers', type=int, default=4, help='number of layers in the UNet')
    parser.add_argument('--hpu', action='store_true', default=False,
                        help='use hpu device')
    parser.add_argument('--use_lazy_mode', action='store_true', default=False,
                        help='enable lazy mode on hpu device, default eager mode')
    parser.add_argument('--data-path', type=str, default='kaggle_3m', metavar='STR',
                        help='input data path for train and test')
    parser.add_argument('--im-cache', action='store_false', default=True, help='use pre-processed and cached images'),
    parser.add_argument('--cache-path', type=str, default='kaggle_cache', help='place to store/load cached images'),
    parser.add_argument('--weights-file', type=str, default='unet.pt', metavar='STR',
                        help='file path to save and load PyTorch weights.')
    parser.add_argument('--image-size', type=int, default=32, help='image size to resize data to')
    parser.add_argument('--aug-angle', type=int, default=5, help='image angle for transforms')
    parser.add_argument('--aug-scale', type=float, default=0.995, help='image scale for transforms')
    parser.add_argument('--world-size', default=1, type=int, metavar='N',
                        help='number of total workers (default: 1)')
    parser.add_argument('--distributed', action='store_true', help='whether to enable distributed mode and run on multiple devices')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, help='rank of the card. Overwritten')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
 