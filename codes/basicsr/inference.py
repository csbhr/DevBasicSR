import os
import cv2
import glob
import torch
import logging
import argparse
import time

from basicsr.archs.video_decompress_sr_arch import VideoDecompressSR
from basicsr.utils import get_env_info, get_root_logger, get_time_str, mkdir_and_rename, tensor2img
from basicsr.data.data_util import read_img_seq
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from basicsr.utils.inference_util import inference_flipx8


def inference_pipeline(args):
    #### configurations
    input_data_folder = args.input_data_folder
    gt_data_folder = args.gt_data_folder
    model_state_path = args.model_state_path
    name_flag = args.name_flag
    save_folder = args.save_folder
    N_in = args.N_in
    self_ensemble = args.self_ensemble
    save_imgs = args.save_imgs

    #### mkdir and initialize loggers
    save_folder = os.path.join(save_folder, name_flag)
    mkdir_and_rename(save_folder)
    log_file = os.path.join(save_folder, f"test_{name_flag}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(f'Inference: {name_flag}.')
    logger.info(f'Input data folder: {input_data_folder}.')
    logger.info(f'GT data folder: {gt_data_folder}.')
    logger.info(f'Model state path: {model_state_path}.')
    logger.info(f'Save folder: {save_folder}.')
    logger.info(f'N_in: {N_in}.')
    logger.info(f'self_ensemble: {self_ensemble}.')
    logger.info(f'save_imgs: {save_imgs}.')

    #### set distributed setting
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create model
    model = VideoDecompressSR(
        lrdecom_mid_channels=128,
        lrdecom_num_blocks=25,
        lrdecom_max_residue_magnitude=10,
        lrdecom_is_low_res_input=False,
        vsr_mid_channels=64,
        vsr_num_blocks=7,
        vsr_max_residue_magnitude=10,
        vsr_is_low_res_input=True,
        cpu_cache_length=50
    )
    model.load_state_dict(
        torch.load(model_state_path, map_location=lambda storage, loc: storage)['params'], strict=True)
    model = model.cuda()
    model.eval()
    logger.info('Model VideoDecompressSR is created.')
    logger.info(f'Load pretrain model from {model_state_path}.')

    #### Inference
    logger.info('Starting inference...')
    save_folder = os.path.join(save_folder, 'results')
    os.makedirs(save_folder, exist_ok=True)
    all_video_psnr, all_video_ssim = [], []

    subfolder_l = sorted(os.listdir(input_data_folder))
    for subfolder in subfolder_l:
        os.makedirs(os.path.join(save_folder, subfolder), exist_ok=True)
        input_paths = sorted(glob.glob(os.path.join(input_data_folder, subfolder, '*')))
        gt_paths = sorted(glob.glob(os.path.join(gt_data_folder, subfolder, '*')))

        # split test clips with N_in
        test_clips = []
        for i in range(len(input_paths) // N_in):
            test_clips.append(input_paths[i * N_in:(i + 1) * N_in])
        if len(input_paths) % N_in != 0:
            test_clips.append(input_paths[-N_in:])

        # test each clip
        outputs_clips = []
        outputs_clips_path = []
        for i, clip in enumerate(test_clips):
            lqs = read_img_seq(clip).unsqueeze(0).cuda()
            with torch.no_grad():
                st_time = time.time()
                if self_ensemble:
                    outputs = inference_flipx8(model, lqs)
                else:
                    outputs = model(lqs)[-1]
                ed_time = time.time()
            if i == len(test_clips) - 1 and len(input_paths) % N_in != 0:
                outputs = outputs[0, -(len(input_paths) % N_in):, :, :, :]
                outputs_clips.extend([tensor2img(outputs[j]) for j in range(outputs.size()[0])])
                outputs_clips_path.extend(clip[-(len(input_paths) % N_in):])
            else:
                outputs = outputs[0, :, :, :, :]
                outputs_clips.extend([tensor2img(outputs[j]) for j in range(outputs.size()[0])])
                outputs_clips_path.extend(clip)
            logger.info(f'Inference done: {subfolder} {i + 1}/{len(test_clips)}, '
                        f'Time: {(ed_time - st_time) / N_in}s per frame.')
            torch.cuda.empty_cache()

        # save results
        video_psnr = []
        video_ssim = []
        for img, path, gt_path in zip(outputs_clips, outputs_clips_path, gt_paths):
            gt_img = cv2.imread(gt_path)
            video_psnr.append(calculate_psnr(img, gt_img, crop_border=4))
            video_ssim.append(calculate_ssim(img, gt_img, crop_border=4))
            if save_imgs:
                basename = os.path.basename(path).split('.')[0]
                cv2.imwrite(os.path.join(save_folder, subfolder, f'{basename}.png'), img)
        logger.info(f'Video done: {subfolder}, '
                    f'PSNR={sum(video_psnr) / len(video_psnr)}, SSIM={sum(video_ssim) / len(video_ssim)}.')
        all_video_psnr.extend(video_psnr)
        all_video_ssim.extend(video_ssim)
    logger.info(f'Inference done, Average '
                f'PSNR={sum(all_video_psnr) / len(all_video_psnr)}, SSIM={sum(all_video_ssim) / len(all_video_ssim)}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BasicSR-Inference')
    parser.add_argument('--input_data_folder', type=str,
                        default='/opt/data/private/baihaoran/Dataset/GOPRO/test/blur')
    parser.add_argument('--gt_data_folder', type=str,
                        default='/opt/data/private/baihaoran/Dataset/GOPRO/test/gt')
    parser.add_argument('--model_state_path', type=str,
                        default='../trained_models/net_g_526000.pth')
    parser.add_argument('--save_folder', type=str, default='../results')
    parser.add_argument('--name_flag', type=str, default='inference_gopro_526000')
    parser.add_argument('--N_in', type=int, default=10)
    parser.add_argument('--self_ensemble', type=bool, default=False)
    parser.add_argument('--save_imgs', type=bool, default=True)
    args = parser.parse_args()

    inference_pipeline(args)
