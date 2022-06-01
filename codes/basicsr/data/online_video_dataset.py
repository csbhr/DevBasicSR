import os
import random
import numpy as np
import cv2
import torch
from torch.utils import data as data
import torch.nn.functional as F

from basicsr.data.transforms import augment, random_crop
from basicsr.data.data_util import sequence_paths_from_video_folder
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.ResizeRight import resize_right, interp_methods
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class OnlinedVideoBlurDownNoiseCompressDataset(data.Dataset):
    """Online video dataset with degradation:
        Blur + Downsampling + Noise + Compression

    Read GT videos and generate LQ videos online.
    Optional: return degradation configures and let the degradation process onto GPUs.

    Only support: opt['io_backend']['type'] == disk
    Only support: opt['phase'] == 'train'

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            blur_type: Blur kernel type. Optional: 'gaussian'.
                If not given, do not operate Blur process.
            kernel_width (int): kernel size.
            is_aniso (bool): If True, using anisotropic gaussian kernel when blur_type='gaussian'.
            sigma_min (float): Gaussian kernel sigma lower limit.
            sigma_max (float): Gaussian kernel sigma higher limit.

            down_type: Downsampling type. Optional: 'bicubic' | 'bilinear' | 'direct'.
                If not given, do not operate Downsampling process.
            scale (bool): Scale, which will be added automatically.

            noise_type: Noise type. Optional: 'gaussian'.
                If not given, do not operate Noise process.
            noise_level_max (float): Gaussian noise level higher limit [0, noise_level_max].

            compress_type: Compression type. Optional: 'jpeg'.
                If not given, do not operate Compression process.
            jepg_quality (int): JPEG quality.

            phase (str): 'train' or 'val'.
            return_configures_for_gpu_degradation (bool): If True, return degradation
                configures(kernel for blurring), and let the degradation process on GPUs.
    """

    def __init__(self, opt):
        super(OnlinedVideoBlurDownNoiseCompressDataset, self).__init__()

        assert opt['io_backend']['type'] == 'disk', "Only support io_backend's type is disk!"
        assert opt['phase'] == 'train', "Only support phase is train!"

        self.opt = opt
        self.gt_root = opt['dataroot_gt']
        self.num_frame = opt['num_frame']

        self.gt_sequence_paths = sequence_paths_from_video_folder(self.gt_root, self.num_frame)

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        self.random_reverse = opt.get('random_reverse', False)

        self.blur_type = opt['blur_type'] if 'blur_type' in opt else None
        if self.blur_type is not None:
            if self.blur_type == 'gaussian':
                assert 'kernel_width' in opt and 'is_aniso' in opt and 'sigma_min' in opt and 'sigma_max' in opt
            else:
                raise NotImplementedError

        self.down_type = opt['down_type'] if 'down_type' in opt else None
        if self.down_type is not None:
            if self.down_type in ['bicubic', 'bilinear', 'direct']:
                assert 'scale' in opt
            else:
                raise NotImplementedError

        self.noise_type = opt['noise_type'] if 'noise_type' in opt else None
        if self.noise_type is not None:
            if self.noise_type == 'gaussian':
                assert 'noise_level_max' in opt
            else:
                raise NotImplementedError

        self.compress_type = opt['compress_type'] if 'compress_type' in opt else None
        if self.compress_type is not None:
            if self.compress_type == 'jpeg':
                assert 'jepg_quality' in opt
            else:
                raise NotImplementedError

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        seq_gt_paths = self.gt_sequence_paths[index]

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            seq_gt_paths.reverse()

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        img_gts = []
        for img_gt_path in seq_gt_paths:
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=False)
            img_gts.append(img_gt)

        # randomly crop
        img_gt_path = seq_gt_paths[self.num_frame // 2]
        img_gts = random_crop(img_gts, self.opt['gt_size'], img_gt_path)

        # augmentation - flip, rotate
        img_gts = augment(img_gts, self.opt['use_flip'], self.opt['use_rot'])

        if 'return_configures_for_gpu_degradation' in self.opt and self.opt['return_configures_for_gpu_degradation']:
            # operate Blur process
            if self.blur_type is not None and self.blur_type == 'gaussian':
                blur_kernel = self.random_anisotropic_gaussian_kernel(
                    width=self.opt['kernel_width'], is_aniso=self.opt['is_aniso'],
                    sig_min=self.opt['sigma_min'], sig_max=self.opt['sigma_max'])
            else:
                blur_kernel = None
            # BGR to RGB, HWC to CHW, numpy to tensor
            # image range: [0, 1], float32.
            img_gts = [img.astype(np.float32) / 255. for img in img_gts]
            img_gts = img2tensor(img_gts)
            img_gts = torch.stack(img_gts, dim=0)
            # img_lqs: (t, c, h, w)
            # img_gts: (t, c, h, w)
            # key: str
            key = os.path.basename(img_gt_path).split('.')[0]
            return {'gt': img_gts, 'key': key, 'degradation_configures': self.opt, 'blur_kernel': blur_kernel}

        # initialize for degradation process
        img_lqs = img_gts.copy()

        # operate Blur process
        if self.blur_type is not None:
            if self.blur_type == 'gaussian':
                img_lqs = self.operate_gaussian_blur_process(
                    img_lqs, kernel_width=self.opt['kernel_width'], is_aniso=self.opt['is_aniso'],
                    sigma_min=self.opt['sigma_min'], sigma_max=self.opt['sigma_max'])

        # operate Downsampling process
        if self.down_type is not None:
            h, w, c = img_gts[0].shape
            img_gts = [img[:h - h % self.opt['scale'], :w - w % self.opt['scale'], :] for img in img_gts]
            img_lqs = [img[:h - h % self.opt['scale'], :w - w % self.opt['scale'], :] for img in img_lqs]
            if self.down_type == 'bicubic':
                img_lqs = [resize_right.resize(
                    img, scale_factors=1.0 / self.opt['scale'], interp_method=interp_methods.cubic) for img in img_gts]
            elif self.down_type == 'bilinear':
                img_lqs = [resize_right.resize(
                    img, scale_factors=1.0 / self.opt['scale'], interp_method=interp_methods.linear) for img in img_gts]
            elif self.down_type == 'direct':
                img_lqs = [img[::self.opt['scale'], ::self.opt['scale'], :] for img in img_gts]

        # operate Noise process
        if self.noise_type is not None:
            if self.noise_type == 'gaussian':
                img_lqs = self.operate_gaussian_noise_process(
                    img_lqs, noise_level_max=self.opt['noise_level_max'], data_range=255.)

        # operate Compression process
        if self.compress_type is not None:
            if self.compress_type == 'jpeg':
                img_lqs = [cv2.imdecode(cv2.imencode(
                    '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), self.opt['jepg_quality']])[1], 1) for img in img_gts]

        # BGR to RGB, HWC to CHW, numpy to tensor
        # image range: [0, 1], float32.
        img_results = [*img_lqs, *img_gts]
        img_results = [img.astype(np.float32) / 255. for img in img_results]
        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        key = os.path.basename(img_gt_path).split('.')[0]
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.gt_sequence_paths)

    def inv_covariance_matrix(self, sig_x, sig_y, theta):
        # sig_x : x-direction standard deviation
        # sig_x : y-direction standard deviation
        # theta : rotation angle
        D_inv = np.array([[1 / (sig_x ** 2), 0.], [0., 1 / (sig_y ** 2)]])  # inverse of diagonal matrix D
        U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # eigenvector matrix
        inv_cov = np.dot(U, np.dot(D_inv, U.T))  # inverse of covariance matrix
        return inv_cov

    def anisotropic_gaussian_kernel(self, width, inv_cov):
        # width : kernel size of anisotropic gaussian filter
        ax = np.arange(-width // 2 + 1., width // 2 + 1.)
        # avoid shift
        if width % 2 == 0:
            ax = ax - 0.5
        xx, yy = np.meshgrid(ax, ax)
        xy = np.stack([xx, yy], axis=2)
        # pdf of bivariate gaussian distribution with the covariance matrix
        kernel = np.exp(-0.5 * np.sum(np.dot(xy, inv_cov) * xy, 2))
        kernel = kernel / np.sum(kernel)
        return kernel

    def random_anisotropic_gaussian_kernel(self, width=15, is_aniso=True, sig_min=0.2, sig_max=4.0):
        if is_aniso:
            sig_x = np.random.random() * (sig_max - sig_min) + sig_min
            sig_y = np.random.random() * (sig_max - sig_min) + sig_min
            theta = np.random.random() * 3.141 / 2.
        else:
            sig = np.random.random() * (sig_max - sig_min) + sig_min
            sig_x = sig
            sig_y = sig
            theta = 0.
        inv_cov = self.inv_covariance_matrix(sig_x, sig_y, theta)
        kernel = self.anisotropic_gaussian_kernel(width, inv_cov)
        kernel = kernel.astype(np.float32)
        return kernel

    def operate_gaussian_blur_process(self, hr_list, kernel_width=15, is_aniso=True, sigma_min=0.2, sigma_max=4.0):
        gaussian_kernel = self.random_anisotropic_gaussian_kernel(
            width=kernel_width, is_aniso=is_aniso, sig_min=sigma_min, sig_max=sigma_max
        )
        hr_list = [
            torch.from_numpy(np.array(hr).astype('float32')).permute(2, 0, 1).unsqueeze(0).float() for hr in hr_list
        ]
        kernel = torch.from_numpy(gaussian_kernel).unsqueeze(0).unsqueeze(0).float()

        _, c, h, w = hr_list[0].size()
        ks = kernel_width
        ps = ks // 2
        hr_list = [F.pad(hr, pad=[ps, ps, ps, ps], mode='replicate') for hr in hr_list]
        blur_list = [
            F.conv2d(hr.view(c, 1, h + 2 * ps, w + 2 * ps), kernel, bias=None, stride=1, padding=0).view(1, c, h, w)
            for hr in hr_list
        ]
        blur_list = [
            blur.clamp(0, 255).round()[0].detach().numpy().transpose(1, 2, 0).astype('uint8') for blur in blur_list
        ]
        return blur_list

    def operate_gaussian_noise_process(self, clean_list, noise_level_max=10, data_range=255.):
        noise_level = np.random.random() * noise_level_max
        noise = np.random.randn(*clean_list[0].shape) * data_range * 0.01 * noise_level
        clean_list = [clean.astype('float32') for clean in clean_list]
        out_list = [clean + noise for clean in clean_list]
        out_list = [np.round(np.clip(out, 0, data_range)).astype('uint8') for out in out_list]
        return out_list
