import os
import random
import torch
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.data.data_util import paired_sequence_paths_from_video_folder
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedVideoDataset(data.Dataset):
    """Paired video dataset loading from disk.

    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Only support: opt['io_backend'] == disk

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).
            gt_mid_frame (bool): If True, only return the gt of median frame.

            scale (int): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(PairedVideoDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.num_frame = opt['num_frame']

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert opt['io_backend']['type'] == 'disk', "Only support io_backend's type is disk!"

        self.sequence_paths = paired_sequence_paths_from_video_folder(
            [self.lq_root, self.gt_root], ['lq', 'gt'], self.num_frame, self.filename_tmpl
        )

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        # temporal augmentation configs
        self.random_reverse = opt.get('random_reverse', False)
        self.gt_mid_frame = opt.get('gt_mid_frame', False)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']

        seq_gt_paths = self.sequence_paths[index]['gt_path']
        seq_lq_paths = self.sequence_paths[index]['lq_path']

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            seq_gt_paths.reverse()
            seq_lq_paths.reverse()

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        img_gts = []
        for img_gt_path in seq_gt_paths:
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)
        img_lqs = []
        for img_lq_path in seq_lq_paths:
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        # randomly crop
        img_gt_path = seq_gt_paths[self.num_frame // 2]
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        if self.gt_mid_frame:
            img_gts = img_gts[self.num_frame // 2]

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        key = os.path.basename(img_gt_path).split('.')[0]
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.sequence_paths)
