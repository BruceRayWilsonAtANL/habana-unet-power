
from collections import OrderedDict
from torch import nn
import torch.optim as optim
import os
import torch
import time
import numpy as np
from skimage.transform import rescale, rotate
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from dataset import BrainSegmentationDataset

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

    def __init__(self, in_channels=3, out_channels=1, initial_features=32):
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
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
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

def transforms(scale=0.995, angle=5):
    transform_list = []

    #TODO set default scale angle values that match normal rotation
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

def datasets(images, image_size, aug_scale, aug_angle):
    train = BrainSegmentationDataset(
        images_dir=images,
        subset="train",
        image_size=image_size,
        transform=transforms(scale=aug_scale, angle=aug_angle),
        from_cache=True,
    )
    valid = BrainSegmentationDataset(
        images_dir=images,
        subset="validation",
        image_size=image_size,
        from_cache=True,
    )
    # print(len(train))
    # print(len(valid))
    return train, valid

def data_loaders(batch_size, workers, image_size, aug_scale, aug_angle):
    dataset_train, dataset_valid = datasets("kaggle_3m", image_size, aug_scale, aug_angle)

    def worker_init(worker_id):
        np.random.seed(2021)
        np.random.seed(42 + worker_id)
    
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid

def log_loss_summary(loss, step, prefix=""):
    print("epoch {} | {}: {}".format(step + 1, prefix + "loss", np.mean(loss)))

def log_scalar_summary(tag, value, step):
    print("epoch {} | {}: {}".format(step + 1, tag, value))

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

def train(model, loss_fn, optimizer, trainloader, device, epoch):
    model.train()
    loss_train = []

    for i, data in enumerate(trainloader):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()
            loss_train.append(loss.item())
    log_loss_summary(loss_train, epoch)

def eval(model, loss_fn, testloader, device, epoch):
    model.eval()

    loss_valid = []
    validation_pred = []
    validation_true = []
    for i, data in enumerate(testloader):
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)

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
    log_loss_summary(loss_valid, epoch, prefix="val_")
    mean_dsc = np.mean(
        dsc_per_volume(
            validation_pred,
            validation_true,
            testloader.dataset.patient_slice_index,
        )
    )
    log_scalar_summary("val_dsc", mean_dsc, epoch)
    return mean_dsc

def main():
   
    #batch_size = 16
    batch_size = 1
    epochs = 25
    # epochs = 1
    lr = 0.0001
    workers = 0
    weights = "./"
    image_size = 32
    #image_size = 64
    #image_size = 224
    aug_scale = 0.05
    aug_angle = 15

    trainloader, validloader = data_loaders(batch_size, workers, image_size, aug_scale, aug_angle)
    # loaders = {"train": loader_train, "valid": loader_valid}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=BrainSegmentationDataset.in_channels, out_channels=BrainSegmentationDataset.out_channels)
    model.to(device)
    
    loss_fn = DiceLoss()
    best_validation_dsc = 0.0
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        print(f"Beginning epoch {epoch}.")
        train(model, loss_fn, optimizer, trainloader, device, epoch)
        dsc = eval(model, loss_fn, validloader, device, epoch)

        if dsc > best_validation_dsc:
            best_validation_dsc = dsc
            torch.save(model.state_dict(), os.path.join(weights, "unet.pt"))

if __name__ == '__main__':
    main()
 