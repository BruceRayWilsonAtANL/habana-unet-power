
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
    dataset_train, dataset_valid = dataset_net(args) #TODO add dataset as an arg

    def worker_init(worker_id):
        np.random.seed(2021)
        np.random.seed(42 + worker_id)
    
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

def train(args, model, loss_fn, optimizer, trainloader, device, epoch):
    model.train()
    train_time = time.time()
    loss_train = []

    for i, data in enumerate(trainloader):
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
    log(
        save_path, f"eval,{epoch},{time.time() - eval_time},{np.mean(loss_valid)}, {mean_dsc}"
    )
    print(log_scalar_summary("val_dsc", mean_dsc, epoch))
    # log(save_path, log_scalar_summary("val_dsc", mean_dsc, epoch))
    return mean_dsc

def log(save_path, line):
    with open(save_path, "a") as file:
        file.write(line +"\n")

def main():
    args = add_parser()
    dir_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(dir_path, f"performance/{args.run_name}.txt")
    log(save_path, f"Command: {sys.argv}")
    log(save_path, f"Begin CSV")
    log(save_path, f"type,epoch,time,loss,dsc")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.multiprocessing.set_start_method('spawn')


    # loaders = {"train": loader_train, "valid": loader_valid}
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_lazy_mode:
        os.environ["PT_HPU_LAZY_MODE"]="1"
    else:
        os.environ["PT_HPU_LAZY_MODE"]="2"   
    device = torch.device("hpu")
    torch.cuda.current_device = lambda: None
    torch.cuda.set_device = lambda x: None
    print(device)

    if args.is_hmp:
        from habana_frameworks.torch.hpex import hmp
        hmp.convert(opt_level=args.hmp_opt_level, bf16_file_path=args.hmp_bf16,
                    fp32_file_path=args.hmp_fp32, isVerbose=args.hmp_verbose)

    utils.init_distributed_mode(args)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    st_timer = time.time()
    trainloader, validloader = data_loaders(args)
    log(save_path, f"loaders_init,,{time.time() - st_timer},,")

    if args.layers == 3:
        from models.unet_three_layer import UNet
    elif args.layers == 4:
        from models.unet_base import UNet
    elif args.layers == 5:
        from models.unet_base import UNet

    model = UNet(in_channels=3, out_channels=1)
    model.to(device)
    
    loss_fn = DiceLoss()
    best_validation_dsc = 0.0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    total_train_time = 0
    total_eval_time = 0

    for epoch in range(1, args.epochs+1):
        print(f"Beginning epoch {epoch}")
        st_timer = time.time()
        train(args, model, loss_fn, optimizer, trainloader, device, epoch)
        end_timer = time.time()
        total_train_time += (end_timer - st_timer)

        st_timer = time.time()
        dsc = eval(args, model, loss_fn, validloader, device, epoch)
        end_timer = time.time()
        total_eval_time += (end_timer - st_timer)

        if dsc > best_validation_dsc:
            best_validation_dsc = dsc
            torch.save(model.state_dict(), os.path.join(dir_path, f"weights/{args.weights_file}"))
    log(save_path, f"total_train_time,{args.epochs},{total_train_time},,")
    print(f"total_train_time = {total_train_time} for {args.epochs} epochs")
    log(save_path, f"total_eval_time,{args.epochs},{total_eval_time},,")
    print(f"total_eval_time = {total_eval_time} for {args.epochs} epochs")
    log(save_path, f"End CSV")
    print(f"\nBest validation mean DSC: {best_validation_dsc}\n")
    log(save_path, f"\nBest validation mean DSC: {best_validation_dsc}\n")

def add_parser():
    # Training settings
    #TODO clean these up
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
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='for Saving the current Model')
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
    parser.add_argument('--hmp', dest='is_hmp', action='store_true', help='enable hmp mode')
    parser.add_argument('--hmp-bf16', default='ops_bf16_mnist.txt', help='path to bf16 ops list in hmp O1 mode')
    parser.add_argument('--hmp-fp32', default='ops_fp32_mnist.txt', help='path to fp32 ops list in hmp O1 mode')
    parser.add_argument('--hmp-opt-level', default='O1', help='choose optimization level for hmp')
    parser.add_argument('--hmp-verbose', action='store_true', help='enable verbose mode for hmp')
    parser.add_argument('--dl-worker-type', default='MP', type=lambda x: x.upper(),
                        choices = ["MT", "MP"], help='select multithreading or multiprocessing')
    parser.add_argument('--world_size', default=1, type=int, metavar='N',
                        help='number of total workers (default: 1)')
    parser.add_argument('--process-per-node', default=8, type=int, metavar='N',
                        help='Number of process per node')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N', help='starting epoch number, default 0')
    parser.add_argument('--checkpoint', default='', help='resume from checkpoint')
    parser.add_argument('--distributed', action='store_true', help='whether to enable distributed mode and run on multiple devices')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
 