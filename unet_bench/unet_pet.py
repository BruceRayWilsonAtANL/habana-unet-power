from collections import OrderedDict
from unittest import loader
from torch import nn
import torch.optim as optim
import os
import time
import torch
import argparse
import numpy as np
from skimage.transform import rescale, rotate 
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
from torchvision import datasets, transforms
from dataset import BrainSegmentationDataset
import utils

import habana_frameworks.torch.core as htcore
# os.environ["PT_HPU_LAZY_MODE"]="1"
# os.environ["PT_HPU_LAZY_MODE"]="2"

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

class UNet(nn.Module):
    #TODO should be 480, way too slow on my machine
    def __init__(self, in_channels=3, out_channels=1, initial_features=48):
        super(UNet, self).__init__()

        features = initial_features
        self.enc1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.dec4 = UNet._block(features * 16, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = UNet._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = UNet._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # print(x.shape)
        enc1 = self.enc1(x)
        # print(enc1.shape)
        enc2 = self.enc2(self.pool1(enc1))
        # print(enc2.shape)
        enc3 = self.enc3(self.pool2(enc2))
        # print(enc3.shape)
        enc4 = self.enc4(self.pool3(enc3))
        # print(enc4.shape)

        bottleneck = self.bottleneck(self.pool4(enc4))
        # print(bottleneck.shape)

        dec4 = self.upconv4(bottleneck)
        # print(dec4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d( 
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

def datasets_net():
    tforms = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])
    train = datasets.OxfordIIITPet(
        "data", 
        transform=tforms, 
        target_transform=tforms, 
        split="trainval", 
        target_types="segmentation", 
        download=True
    )
    valid = datasets.OxfordIIITPet(
        "data", 
        transform=tforms, 
        target_transform=tforms, 
        split="test", 
        target_types="segmentation", 
        download=True
    )
    return train, valid

def data_loaders(args, workers):
    dataset_train, dataset_valid = datasets_net()

    def worker_init(worker_id):
        np.random.seed(2021)
        np.random.seed(42 + worker_id)
    
    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        # num_workers=12,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    # print(len(loader_train))
    loader_valid = DataLoader(
        dataset_valid,
        # num_workers=12,
        pin_memory=True,
        drop_last=True,
        batch_size=args.batch_size,
    )
    # print(len(loader_valid))
    return loader_train, loader_valid

def log_loss_summary(loss, step, prefix=""):
    print("epoch {} | {}: {}".format(step, prefix + "loss", np.mean(loss)))

def log_scalar_summary(tag, value, step):
    print("epoch {} | {}: {}".format(step, tag, value))

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
    loss_train = []
    step = 0

    for i, data in enumerate(trainloader):
        step += 1
        if step > 1000:
            log_loss_summary(loss_train, epoch)
            return
        img, target = data
        img = img.to(device)
        if args.use_lazy_mode:
            htcore.mark_step() 

        optimizer.zero_grad()
        st_timer = time.time()

        with torch.set_grad_enabled(True):
            pred = model(img)
            pred = pred.squeeze(1)
            target = target.squeeze(1)
            loss = loss_fn(pred, target)
            loss.backward()
            if args.use_lazy_mode:
                htcore.mark_step()
            optimizer.step()
            if args.use_lazy_mode:
                htcore.mark_step()
            end_timer = time.time()
            loss_train.append(loss.item())
        if args.batch_size > 1:
            log_loss_summary(loss_train, epoch)
            loss_train=[]

def eval(args, model, loss_fn, testloader, device, epoch):
    model.eval()

    loss_valid = []
    validation_pred = []
    validation_true = []
    step = 0
    for i, data in enumerate(testloader):
        if step > 100:
            log_loss_summary(loss_valid, epoch, prefix="val_")
            # mean_dsc = np.mean(
            #     dsc_per_volume(
            #         validation_pred,
            #         validation_true,
            #         testloader.dataset.patient_slice_index, #TODO figure this out tomorrow
            #     )
            # )
            # log_scalar_summary("val_dsc", mean_dsc, epoch)
            return np.mean(loss_valid)
        img, target = data
        img = img.to(device)
        if args.use_lazy_mode:
            htcore.mark_step() 

        pred = model(img)
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        loss = loss_fn(pred, target)
        # htcore.mark_step()

        loss_valid.append(loss.item())
        y_pred_np = pred.detach().cpu().numpy()
        validation_pred.extend(
            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
        )
        y_true_np = pred.detach().cpu().numpy()
        validation_true.extend(
            [y_true_np[s] for s in range(y_true_np.shape[0])]
        )

    # log_loss_summary(loss_valid, epoch, prefix="val_")
    # mean_dsc = np.mean(
    #     dsc_per_volume(
    #         validation_pred,
    #         validation_true,
    #         testloader.dataset.patient_slice_index, #TODO see above
    #     )
    # )
    # log_scalar_summary("val_dsc", mean_dsc, epoch)
    return np.mean(loss_valid)

def main():
    args = add_parser()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.multiprocessing.set_start_method('spawn')
   
    #batch_size = 16
    batch_size = 1
    epochs = 25
    # epochs = 1
    lr = 0.0001
    workers = 0
    weights = "./"

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

    trainloader, validloader = data_loaders(args, workers)


    model = UNet(in_channels=3, out_channels=1)
    model.to(device)
    
    loss_fn = DiceLoss()
    best_validation_dsc = 0.0
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, args.epochs+1):
        print(f"Beginning epoch {epoch}")
        train(args, model, loss_fn, optimizer, trainloader, device, epoch)
        dsc = eval(args, model, loss_fn, validloader, device, epoch)

        if dsc > best_validation_dsc:
            best_validation_dsc = dsc
            torch.save(model.state_dict(), os.path.join(weights, "pet.pt"))

def add_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch unet bmri')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
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
    parser.add_argument('--hpu', action='store_true', default=False,
                        help='Use hpu device')
    parser.add_argument('--use_lazy_mode', action='store_true', default=False,
                        help='Enable lazy mode on hpu device, default eager mode')
    parser.add_argument('--data-path', type=str, default='./lgg-mri-segmentation/kaggle_3m', metavar='STR',
                        help='input data path for train and test')
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