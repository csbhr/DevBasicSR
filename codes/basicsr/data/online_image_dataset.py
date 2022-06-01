from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment, random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.ResizeRight import resize_right, interp_methods
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np
import cv2
import torch
import torch.nn.functional as F


@DATASET_REGISTRY.register()
class OnlineImageBlurDownNoiseCompressDataset(data.Dataset):
    """Online image dataset with degradation:
        Blur + Downsampling + Noise + Compression

    Read GT images and generate LQ images online.
    Optional: return degradation configures and let the degradation process onto GPUs.

    Only support: opt['io_backend']['type'] == disk

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
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
        super(OnlineImageBlurDownNoiseCompressDataset, self).__init__()

        assert opt['io_backend']['type'] == 'disk', "Only support io_backend's type is disk!"

        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        self.paths = paths_from_folder(self.gt_folder)

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

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=False)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt = random_crop(img_gt, gt_size, gt_path)
            # flip, rotation
            img_gt = augment(img_gt, self.opt['use_flip'], self.opt['use_rot'])

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
            img_gt = img_gt.astype(np.float32) / 255.
            img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
            # normalize
            if self.mean is not None or self.std is not None:
                normalize(img_gt, self.mean, self.std, inplace=True)
            return {'gt': img_gt, 'gt_path': gt_path, 'degradation_configures': self.opt, 'blur_kernel': blur_kernel}

        # initialize for degradation process
        img_lq = img_gt

        # operate Blur process
        if self.blur_type is not None:
            if self.blur_type == 'gaussian':
                img_lq = self.operate_gaussian_blur_process(
                    img_lq, kernel_width=self.opt['kernel_width'], is_aniso=self.opt['is_aniso'],
                    sigma_min=self.opt['sigma_min'], sigma_max=self.opt['sigma_max'])

        # operate Downsampling process
        if self.down_type is not None:
            h, w, c = img_gt.shape
            img_gt = img_gt[:h - h % self.opt['scale'], :w - w % self.opt['scale'], :]
            img_lq = img_lq[:h - h % self.opt['scale'], :w - w % self.opt['scale'], :]
            if self.down_type == 'bicubic':
                img_lq = resize_right.resize(
                    img_lq, scale_factors=1.0 / self.opt['scale'], interp_method=interp_methods.cubic)
            elif self.down_type == 'bilinear':
                img_lq = resize_right.resize(
                    img_lq, scale_factors=1.0 / self.opt['scale'], interp_method=interp_methods.linear)
            elif self.down_type == 'direct':
                img_lq = img_lq[::self.opt['scale'], ::self.opt['scale'], :]

        # operate Noise process
        if self.noise_type is not None:
            if self.noise_type == 'gaussian':
                img_lq = self.operate_gaussian_noise_process(
                    img_lq, noise_level_max=self.opt['noise_level_max'], data_range=255.)

        # operate Compression process
        if self.compress_type is not None:
            if self.compress_type == 'jpeg':
                img_lq = cv2.imdecode(cv2.imencode(
                    '.jpg', img_lq, [int(cv2.IMWRITE_JPEG_QUALITY), self.opt['jepg_quality']])[1], 1)

        # BGR to RGB, HWC to CHW, numpy to tensor
        # image range: [0, 1], float32.
        img_gt = img_gt.astype(np.float32) / 255.
        img_lq = img_lq.astype(np.float32) / 255.
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

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

    def operate_gaussian_blur_process(self, hr, kernel_width=15, is_aniso=True, sigma_min=0.2, sigma_max=4.0):
        gaussian_kernel = self.random_anisotropic_gaussian_kernel(
            width=kernel_width, is_aniso=is_aniso, sig_min=sigma_min, sig_max=sigma_max
        )
        hr = torch.from_numpy(np.array(hr).astype('float32')).permute(2, 0, 1).unsqueeze(0).float()
        kernel = torch.from_numpy(gaussian_kernel).unsqueeze(0).unsqueeze(0).float()

        _, c, h, w = hr.size()
        ks = kernel_width
        ps = ks // 2
        hr = F.pad(hr, pad=[ps, ps, ps, ps], mode='replicate')
        blur = F.conv2d(hr.view(c, 1, h + 2 * ps, w + 2 * ps), kernel, bias=None, stride=1, padding=0).view(1, c, h, w)
        blur = blur.clamp(0, 255).round()[0].detach().numpy().transpose(1, 2, 0).astype('uint8')
        return blur

    def operate_gaussian_noise_process(self, clean, noise_level_max=10, data_range=255.):
        noise_level = np.random.random() * noise_level_max
        noise = np.random.randn(*clean.shape) * data_range * 0.01 * noise_level
        clean = clean.astype('float32')
        out = clean + noise
        out = np.round(np.clip(out, 0, data_range)).astype('uint8')
        return out
