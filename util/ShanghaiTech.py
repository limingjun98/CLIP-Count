# -*- coding : utf-8 -*-
# @FileName  : Shanghai.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Apr 30, 2023
# @Github    : https://github.com/songrise
# @Description: shanghai tech dataset
import numpy as np
from torchvision import transforms
import scipy.ndimage as ndimage
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import scipy as sp
import einops
from scipy import io
import glob as gb
import cv2
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


class ShanghaiTech(Dataset):
    def __init__(self, data_dir:str, split:str, part:str, resize_val:bool=True,
                 preserve_the_original_image: bool = False):
        """
        Parameters
        ----------
        data_dir : str, path to the data directory
        split : str, 'train', 'val' or 'test'
        subset_scale : float, scale of the subset of the dataset to use
        resize_val : bool, whether to random crop validation images to 384x384
        anno_file : str, FSC-133 or FSC-147 
        """
        assert split in ['train', 'test']
        assert part in ['A', 'B']

        if not data_dir:
            data_dir = "data/ShanghaiTech/part_{}/{}_data"

        #!HARDCODED Dec 25: 
        self.data_dir = data_dir.format(part, split)



        self.resize_val = resize_val
        self.im_dir = os.path.join(self.data_dir,'images')
        self.anno_path = os.path.join(self.data_dir , "ground-truth")
        # self.data_split_path = os.path.join(self.data_dir,'ImageSets')
        self.split = split
        # self.split_file = os.path.join(self.data_split_path, split + '.txt')

        # with open(self.split_file,"r") as s:
        #     img_names = s.readlines()
        self.img_paths = gb.glob(os.path.join(self.im_dir, "*.jpg"))
        self.img_names = [p.replace('\\', '/').split("/")[-1].split(".")[0] for p in self.img_paths]
        self.gt_cnt = {}
        for im_name in self.img_names:

            assert os.path.exists(os.path.join(self.im_dir, f"{im_name}.jpg"))
            assert os.path.exists(os.path.join(self.anno_path, f"GT_{im_name}.mat"))
            # the sub package of scipy has been changed, the above code needs to be commented
            # with open(os.path.join(self.anno_path, f"GT_{im_name}.mat"), "rb") as f:
            #     mat = sp.io.loadmat(f)
            #     # the number of count is length of the points
            #     self.gt_cnt[im_name] = len(mat["image_info"][0][0][0][0][0])

            mat = io.loadmat(os.path.join(self.anno_path, f"GT_{im_name}.mat"))
            # the number of count is length of the points
            points_ndarray = mat["image_info"][0][0][0][0][0]
            self.gt_cnt[im_name] = len(points_ndarray)
        # resize the image height to 384, keep the aspect ratio
        self.preprocess = transforms.Compose([
            transforms.Resize(384), 
            transforms.ToTensor(),
        ])
        self.preserve_the_original_image = preserve_the_original_image


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        im_name = self.img_names[idx]
        im_path = os.path.join(self.im_dir, f"{im_name}.jpg")
        img = Image.open(im_path)
        # if the image height larger than width, rotate it
        img_width, img_height = img.size[0], img.size[1]
        whether_rotate = False
        if img_width < img_height:
            img = img.rotate(90, expand=True)
            img_width, img_height = img_height, img_width
            whether_rotate = True
        # if the image is grayscale, convert it to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")


        # create an image conversion operation -lmj
        to_tensor = transforms.ToTensor()
        origin_img_tensor = to_tensor(img)

        img = self.preprocess(img)
        gt_cnt = self.gt_cnt[im_name]

        if self.preserve_the_original_image:
            # create groundTruth
            mat = io.loadmat(os.path.join(self.anno_path, f"GT_{im_name}.mat"))
            points_ndarray = mat["image_info"][0][0][0][0][0]
            # gt_density = torch.zeros(img_height, img_width)
            gt_density = np.zeros([img_height, img_width], dtype='float32')
            for keypoint in points_ndarray:
                if whether_rotate:
                    gt_density[round(keypoint[0]), round(keypoint[1])] = 1.
                else:
                    gt_density[round(keypoint[1]), round(keypoint[0])] = 1.
            gt_density = ndimage.gaussian_filter(gt_density, sigma=(1, 1), order=0)
            gt_density = torch.from_numpy(gt_density)
            gt_density = gt_density * 60
            return img, gt_cnt, origin_img_tensor, gt_density
        else:
            return img, gt_cnt
    

#test
if __name__ == "__main__":
    dataset = ShanghaiTech("E:/experiment/CLIP-Count/data/ShanghaiTech/part_{}/{}_data", split="train", part="A",
                           preserve_the_original_image=True)
    # dataset = ShanghaiTech(None, split="train", part="A")
    # sample one image
    img, cnt, origin_img_tensor, gt_density = dataset[0]

    gt_density = gt_density.detach().cpu().numpy()
    gt_density = einops.repeat(gt_density, 'h w -> c h w', c=3)
    gt_density = gt_density / gt_density.max()  # normalize
    gt_density_write = 1. - gt_density[0]
    gt_density_write = cv2.applyColorMap(np.uint8(255 * gt_density_write), cv2.COLORMAP_JET)
    gt_density_img = Image.fromarray(np.uint8(gt_density_write))
    gt_density_img.save("test_gt_density.png")

    gt_density_write = gt_density_write / 255.
    overlay_write = 0.33 * np.transpose(origin_img_tensor, (1, 2, 0)) + 0.67 * gt_density_write
    overlay_img = Image.fromarray(np.uint8(255 * overlay_write))
    overlay_img.save("overlay_gt_density.png")

    #save image
    img = img.permute(1,2,0).numpy()*255
    print(img.shape)
    print(cnt)
    Image.fromarray(img.astype(np.uint8)).save("test.png")
