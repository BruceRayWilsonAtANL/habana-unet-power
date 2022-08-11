import os
import torch
import numpy as np
from skimage.io import imread, imsave
from torch.utils.data import Dataset
from skimage.transform import resize
from skimage.exposure import rescale_intensity

def crop_sample(x):
    volume, mask = x

    volume[volume < np.max(volume) * 0.1] = 0
    z_proj = np.max(np.max(np.max(volume, axis=-1), axis=-1), axis=-1)
    z_nonzero = np.nonzero(z_proj)
    z_min = np.min(z_nonzero)
    z_max = np.max(z_nonzero) + 1
    y_proj = np.max(np.max(np.max(volume, axis=0), axis=-1), axis=-1)
    y_nonzero = np.nonzero(y_proj)
    y_min = np.min(y_nonzero)
    y_max = np.max(y_nonzero) + 1
    x_proj = np.max(np.max(np.max(volume, axis=0), axis=0), axis=-1)
    x_nonzero = np.nonzero(x_proj)
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero) + 1

    return (
        volume[z_min:z_max, y_min:y_max, x_min:x_max],
        mask[z_min:z_max, y_min:y_max, x_min:x_max]
    )


def pad_sample(x):
    volume, mask = x
    a = volume.shape[1]
    b = volume.shape[2]
    if a == b:
        return volume, mask
    diff = (max(a, b) - min(a, b)) / 2.0
    if a > b: 
        padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
    mask = np.pad(mask, padding, mode="constant", constant_values=0)
    padding = padding + ((0, 0),)
    volume = np.pad(volume, padding, mode="constant", constant_values=0)
    return volume, mask


def resize_sample(x, size=256):
    volume, mask = x
    v_shape = volume.shape
    out_shape = (v_shape[0], size, size)
    mask = resize(
        mask,
        output_shape=out_shape,
        order=0,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    out_shape = out_shape + (v_shape[3],)
    volume = resize(
        volume,
        output_shape=out_shape,
        order=2,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    return volume, mask


def normalize_volume(volume):
    volume = rescale_intensity(volume, in_range=(np.percentile(volume, 10), np.percentile(volume, 99)))
    volume = (volume - np.mean(volume, axis=(0, 1, 2))) / np.std(volume, axis=(0, 1, 2))
    return volume


def normalize_mask(mask):
    mask = mask / (np.max(mask))
    return mask


class BrainSegmentationDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 3 # Constants
    out_channels = 1

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        from_cache=False,
        cache_dir="kaggle_cache"
    ):
        assert subset in ["all", "train", "validation"]
        volumes = {}
        masks = {}
        
        if from_cache:
            for patient in sorted(os.listdir(cache_dir)):
                if "mask.npy" in patient:
                    masks[patient.replace("_mask", "")] = np.load(f"{cache_dir}/{patient}")
                elif ".npy" in patient:
                    volumes[patient] = np.load(f"{cache_dir}/{patient}")
            self.patients = sorted(volumes)
            
            if not subset == "all":
                
                validation_patients = self.patients[-10:]
                
                if subset == "validation":
                    self.patients = validation_patients
                else:
                    self.patients = sorted(
                        list(set(self.patients).difference(validation_patients))
                    )
            self.volumes = [(volumes[k], masks[k]) for k in self.patients]
            print("loaded {} cached dataset.".format(subset))

        else:
            print("reading {} images...".format(subset))
            step = 0
            
            for (dirpath, dirnames, filenames) in os.walk(images_dir):
                image_slices = []
                mask_slices = []
                for filename in sorted( 
                    filter(lambda f: ".tif" in f, filenames),
                    key=lambda x: int(x.split(".")[-2].split("_")[4]),
                ):
                    filepath = os.path.join(dirpath, filename)
                    if "mask" in filename:
                        mask_slices.append(imread(filepath, as_gray=True))
                    else:
                        image_slices.append(imread(filepath))
                    
                if len(image_slices) > 0:
                    patient_id = dirpath.split("/")[-1]
                    volumes[patient_id] = np.array(image_slices[1:-1])
                    masks[patient_id] = np.array(mask_slices[1:-1])

            self.patients = sorted(volumes)

            if not subset == "all":
                
                validation_patients = self.patients[-10:]
                
                if subset == "validation":
                    self.patients = validation_patients
                else:
                    self.patients = sorted(
                        list(set(self.patients).difference(validation_patients))
                    )
            print(f"start: {volumes[self.patients[0]].shape}")

            print("preprocessing {} volumes...".format(subset))
            self.volumes = [(volumes[k], masks[k]) for k in self.patients]

            print("cropping {} volumes...".format(subset))
            self.volumes = [crop_sample(v) for v in self.volumes]
            print(f"crop: {self.volumes[0][0].shape}")

            print("padding {} volumes...".format(subset))
            self.volumes = [pad_sample(v) for v in self.volumes]
            print(f"pad: {self.volumes[0][0].shape}")

            print("resizing {} volumes...".format(subset))
            self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]
            print(f"resize: {self.volumes[0][0].shape}")

            print("normalizing {} volumes...".format(subset))
            self.volumes = [(normalize_volume(v), normalize_mask(m)) for v, m in self.volumes] 
            self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]
            print(f"normalize: {self.volumes[0][0].shape}")
            exit()

            print("done creating {} dataset".format(subset))
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            vol = 0
            for i in range(len(self.volumes)):
                vol = self.volumes[i]
                patient = self.patients[i]
                path_name = f"{cache_dir}/{patient}"
                np.save(path_name, vol[0])
                np.save(f"{path_name}_mask", vol[1])
            print("done caching {} dataset".format(subset))

        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.transform = transform

    def __len__(self):
        return len(self.patient_slice_index)

    # Unsure if CB will like these PyTorch tensors.
    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]


        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        return image_tensor, mask_tensor