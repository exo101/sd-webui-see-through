import os.path as osp
import argparse
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
from tqdm import tqdm

from common.utils.io_utils import json2dict, dict2json, load_img_depth, save_psd, find_all_imgs
from common.utils import inference_utils
from common.utils.inference_utils import apply_layerdiff, apply_marigold, further_extr
from common.utils.torch_utils import seed_everything

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='workspace/layerdiff_output')
    parser.add_argument('--srcp', type=str, default='assets/test_image.png', help='input image')
    parser.add_argument('--seed', type=int, default=42)
    # 默认使用本地模型路径，用户需要手动下载模型到这个目录
    default_model_dir = os.path.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'models')
    os.makedirs(default_model_dir, exist_ok=True)
    
    # 本地模型路径
    default_layerdiff_model = os.path.join(default_model_dir, 'seethroughv0.0.2_layerdiff3d')
    default_depth_model = os.path.join(default_model_dir, 'seethroughv0.0.1_marigold')
    
    parser.add_argument('--repo_id_layerdiff', default=default_layerdiff_model)
    parser.add_argument('--repo_id_depth', default=default_depth_model)
    parser.add_argument('--vae_ckpt', default=None)
    parser.add_argument('--unet_ckpt', default=None)
    parser.add_argument('--resolution', type=int, default=1280)
    parser.add_argument('--save_to_psd', action='store_true')
    parser.add_argument('--tblr_split', action='store_true', help='try split parts (handwear, eyes, etc) into left-right components')
    args = parser.parse_args()
    srcp = args.srcp

    if osp.isdir(srcp):
        imglist = find_all_imgs(srcp, abs_path=True)
    else:
        imglist = [srcp]

    for srcp in tqdm(imglist):

        seed_everything(args.seed)

        print('running layerdiff...')
        apply_layerdiff(srcp, args.repo_id_layerdiff, save_dir=args.save_dir, seed=args.seed, vae_ckpt=args.vae_ckpt, unet_ckpt=args.unet_ckpt, resolution=args.resolution)
        
        print('running marigold...')
        apply_marigold(srcp, args.repo_id_depth, save_dir=args.save_dir, seed=args.seed, resolution=args.resolution)

        srcname = osp.basename(osp.splitext(srcp)[0])
        saved = osp.join(args.save_dir, srcname)
        further_extr(saved, rotate=False, save_to_psd=args.save_to_psd, tblr_split=args.tblr_split)
