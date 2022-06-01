import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch_util import ResidualBlocksWithInputConv, PixelShufflePack, flow_warp
from .spynet_arch import SpyNet
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class EnhancedBasicVSR(nn.Module):
    """Enhanced BasicVSR network structure.

    Support either x4 upsampling or same size output.

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        extract_blocks (int, optional): The number of residual blocks in feature
            extraction module. Default: 1.
        propagation_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 15.
        recons_blocks (int, optional): The number of residual blocks in reconstruction
            module. Default: 3.
        propagation_branches (list[str], optional): The names of the propagation branches.
            Default: ('backward_1', 'forward_1').
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
        spynet_pretrained (str, optional): Pre-trained model path of SPyNet.
            Default: None.
        cpu_cache_length (int, optional): When the length of sequence is larger
            than this value, the intermediate features are sent to CPU. This
            saves GPU memory, but slows down the inference speed. You can
            increase this number if you have a GPU with large memory.
            Default: 100.
    """

    def __init__(self,
                 mid_channels=64,
                 extract_blocks=1,
                 propagation_blocks=15,
                 recons_blocks=3,
                 is_low_res_input=True,
                 propagation_branches=('backward_1', 'forward_1'),
                 spynet_pretrained=None,
                 cpu_cache_length=100):

        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = SpyNet(load_path=spynet_pretrained)

        # feature extraction module
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, num_blocks=extract_blocks)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, num_blocks=extract_blocks))

        # propagation branches
        self.backbone = nn.ModuleDict()
        self.propagation_branches = propagation_branches
        for i, module in enumerate(self.propagation_branches):
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks=propagation_blocks)

        # reconstruction module
        self.reconstruction = ResidualBlocksWithInputConv(
            (len(self.propagation_branches) + 1) * mid_channels, mid_channels, num_blocks=recons_blocks)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lqs):
        """Forward function for BasicVSR++.
        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        feats = {}

        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                _feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(_feat)
                torch.cuda.empty_cache()
        else:
            _feats = self.feat_extract(lqs.view(-1, c, h, w))
            _feats = _feats.view(n, t, -1, _feats.size(2), _feats.size(3))
            feats['spatial'] = [_feats[:, i, :, :, :] for i in range(0, t)]

        # compute flows
        flows_forward, flows_backward = self.compute_flow(lqs.clone())

        # feature propgation
        for module in self.propagation_branches:
            feats[module] = []

            if 'backward' in module:
                flows = flows_backward
            else:
                flows = flows_forward

            feats = self.propagate(feats, flows, module)
            if self.cpu_cache:
                del flows
                torch.cuda.empty_cache()

        out = self.reconstruct(lqs, feats)

        return out

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.
        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """
        if not self.is_low_res_input:
            n, t, c, h, w = lqs.size()
            lqs = F.interpolate(lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        is_mirror_extended = False
        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                is_mirror_extended = True

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = flows_backward.flip(1)
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()
            torch.cuda.empty_cache()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, branch_name):
        """Propagate the latent features throughout the sequence.
        Args:
            feats (Dict[list[tensor]]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            branch_name (str): The name of the propagation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
        """

        n, _, _, h, w = flows.size()
        t = len(feats['spatial'])

        # In forward: flow_idx=frame_idx-1
        # In backward: flow_idx=frame
        frame_idx = range(0, t)
        flow_idx = range(-1, t - 1)
        if 'backward' in branch_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i in range(t):
            feat_current = feats['spatial'][frame_idx[i]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()

            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()
                feat_prop = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

            # concatenate and residual blocks
            feat_l = [feat_current] + [
                feats[k][frame_idx[i]] for k in feats if k not in ['spatial', branch_name]
            ] + [feat_prop]
            if self.cpu_cache:
                feat_l = [f.cuda() for f in feat_l]

            feat_prop = feat_prop + self.backbone[branch_name](torch.cat(feat_l, dim=1))
            feats[branch_name].append(feat_prop)

            if self.cpu_cache:
                feats[branch_name][-1] = feats[branch_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in branch_name:
            feats[branch_name] = feats[branch_name][::-1]

        return feats

    def reconstruct(self, lqs, feats):
        """Compute the output image given the features.
        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.
        """
        outputs = []
        t = len(feats['spatial'])

        for i in range(t):
            feat_l = [feats[k].pop(0) for k in feats]
            hr = torch.cat(feat_l, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)
