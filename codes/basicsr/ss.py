import torch
import numpy as np
from basicsr.data.paired_video_dataset import PairedVideoDataset
from basicsr.data.video_test_dataset import VideoRecurrentSplitClipsTestDataset

opt = {
    'dataroot_gt': '/home/csbhr/Disk-2T/Dataset/Video_SR/REDS/val_REDS4/sharp',
    'dataroot_lq': '/home/csbhr/Disk-2T/Dataset/Video_SR/REDS/val_REDS4/sharp_bicubic_X4',
    'name': 'aaa',
    'padding': 'reflection',
    'num_frame': 15,
    'num_overlap': 5,
    'gt_size': 256,
    'interval_list': [1],
    'random_reverse': False,
    'use_flip': True,
    'use_rot': True,
    'cache_data': True,
    'io_backend': {'type': 'disk'},
}

data = VideoRecurrentSplitClipsTestDataset(opt)
da0 = data[0]
da1 = data[1]
a1 = da0['lq'][-5:, :, :, :]
a2 = da1['lq'][:5, :, :, :]
diff = a1 - a2
dsum = torch.sum(torch.abs(diff))
print()
for da in data:
    print()
