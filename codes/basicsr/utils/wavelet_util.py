import torch


def haar_dwt(x):
    '''Haar Wavelet Transform
        Inputs: Tensor with shape [b, c, h, w]
        Outputs: Tensor with shape [b, 4, c, h/2, w/2]
    '''
    x01 = x[:, :, 0::2, :]
    x02 = x[:, :, 1::2, :]
    x1 = x01[:, :, :, 0::2] / 2
    x2 = x02[:, :, :, 0::2] / 2
    x3 = x01[:, :, :, 1::2] / 2
    x4 = x02[:, :, :, 1::2] / 2

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    h = torch.stack((x_LL, x_HL, x_LH, x_HH), dim=1)

    return h


def harr_iwt(x):
    '''Inverse Haar Wavelet Transform
        Inputs: Tensor with shape [b, 4, c, h/2, w/2]
        Outputs: Tensor with shape [b, c, h, w]
    '''
    b, r, c, h, w = x.size()
    assert r == 4
    x1 = x[:, 0, :, :, :] / 2
    x2 = x[:, 1, :, :, :] / 2
    x3 = x[:, 2, :, :, :] / 2
    x4 = x[:, 3, :, :, :] / 2

    h = torch.zeros([b, c, 2 * h, 2 * w]).type_as(x)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


if __name__ == '__main__':
    import numpy as np
    import cv2
    from basicsr.utils.img_util import img2tensor, tensor2img, imwrite


    def reconstruct(x):
        x_l_top = x[:, 0, :, :, :]
        x_r_top = x[:, 1, :, :, :]
        x_l_bot = x[:, 2, :, :, :]
        x_r_bot = x[:, 3, :, :, :]
        x_top = torch.cat((x_l_top, x_r_top), dim=3)
        x_bot = torch.cat((x_l_bot, x_r_bot), dim=3)
        x_rec = torch.cat((x_top, x_bot), dim=2)
        return x_rec


    img = cv2.imread(
        '/home/csbhr/Disk-2T/Dataset/Video_SR/REDS/val_REDS4/sharp/011/00000033.png'
    ).astype(np.float32) / 255.
    img_tensor = img2tensor(img).unsqueeze(0).cuda()

    ori_img = tensor2img(img_tensor)
    imwrite(ori_img, '/home/csbhr/Desktop/fdownload/ori_img.png')

    wavelet_tensor = haar_dwt(img_tensor)

    wavelet_tensor_combine = reconstruct(wavelet_tensor)
    wavelet_img = tensor2img(wavelet_tensor_combine)
    imwrite(wavelet_img, '/home/csbhr/Desktop/fdownload/wavelet_img.png')

    i_wavelet_tensor = harr_iwt(wavelet_tensor)

    i_wavelet_img = tensor2img(i_wavelet_tensor)
    imwrite(i_wavelet_img, '/home/csbhr/Desktop/fdownload/i_wavelet_img.png')

    diff = torch.sum(torch.abs(img_tensor - i_wavelet_tensor))
    print(diff)
