import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
import cv2


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample
    
class CAMUS_dataset(Dataset):
    def __init__(self, base_dir, split, start=0, end=-1, skip=1, transform=None, mode='train'):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.start = start
        self.end = end
        self.skip = skip
        self.data_dir = base_dir          
        self.mode = mode 
        if self.mode == 'val':
            self.val_list = np.array(sorted(os.listdir(self.data_dir)))
        else:
            self.img_list = np.array(sorted(os.listdir(os.path.join(base_dir,split,'images'))))
            self.mask_list = np.array(sorted(os.listdir(os.path.join(base_dir,split,'masks'))))
            if end == -1:
                self.img_list = self.img_list[start::skip]
                self.mask_list = self.mask_list[start::skip]
            else:
                self.img_list = self.img_list[start:end:skip]
                self.mask_list = self.mask_list[start::skip]

        

    def __len__(self):
        if self.mode == 'val':
            return len(self.val_list)
        else:
            return len(self.img_list)

    def __getitem__(self, idx):
        if self.split == "train":
            img_name = self.img_list[idx].strip('\n')
            image_path = os.path.join(self.data_dir, self.split, 'images', img_name)
            mask_path = os.path.join(self.data_dir, self.split, 'masks', img_name)
            image = cv2.imread(image_path,-1).astype('float32')
            cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
            label = cv2.imread(mask_path,-1)
        else:
            if self.mode == 'val':
                img_name = self.val_list[idx].strip('\n')
                image_path = os.path.join(self.data_dir, img_name)
                image = cv2.imread(image_path,-1).astype('float32')
                image = np.expand_dims(image,axis=0)
                cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)

                sample = {'image': image}
                if self.transform:
                    sample = self.transform(sample)
                sample['case_name'] = self.val_list[idx].strip('\n').split('.')[0]
                return sample

            else:
                img_name = self.img_list[idx].strip('\n')
                image_path = os.path.join(self.data_dir, self.split, 'images', img_name)
                mask_path = os.path.join(self.data_dir, self.split, 'masks', img_name)
                image = cv2.imread(image_path,-1).astype('float32')
                image = np.expand_dims(image,axis=0)
                cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
                label = cv2.imread(mask_path,-1)
                label = np.expand_dims(label,axis=0)

        # Input dim should be consistent
        # Since the channel dimension of nature image is 3, that of medical image should also be 3

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.img_list[idx].strip('\n').split('.')[0]
        return sample
    
if __name__=='__main__':
    db_train = CAMUS_dataset(base_dir='/data/yisi/mywork/SAMed/CAMUS', 
                             split="train")
    print('a')