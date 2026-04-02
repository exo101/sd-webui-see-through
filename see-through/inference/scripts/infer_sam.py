import argparse
from omegaconf import OmegaConf

from PIL import Image

import os
import os.path as osp
import numpy as np
from  tqdm import tqdm
from einops import reduce
import click
import cv2
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from utils.io_utils import load_exec_list, find_all_imgs



def sam_parse_body_samples(config):

    from live2d.scrap_model import animal_ear_detected, Drawable, VALID_BODY_PARTS_V2
    from utils.cv import fgbg_hist_matching, quantize_image, random_crop, rle2mask, mask2rle, img_alpha_blending, resize_short_side_to, batch_save_masks, batch_load_masks
    from utils.torch_utils import seed_everything, init_model_from_pretrained
    from utils.visualize import visualize_segs_with_labels
    from modules.semanticsam import SemanticSam, Sam
    import torch


    seed_everything(42)

    config = OmegaConf.load(config)
    
    exec_list = config.exec_list
    ckpt = config.ckpt
    rank_to_worldsize = config.rank_to_worldsize
    save_dir = config.save_dir
    save_to_local = config.get('save_to_local', False)

    if not save_to_local:
        os.makedirs(save_dir, exist_ok=True)

    if osp.isdir(exec_list):
        exec_list = find_all_imgs(exec_list, abs_path=True)

    exec_list = load_exec_list(exec_list, rank_to_worldsize=rank_to_worldsize)

    model: SemanticSam = init_model_from_pretrained(
        pretrained_model_name_or_path=ckpt,
        module_cls=SemanticSam,
        download_from_hf=False,
        model_args=dict(class_num=19)
    ).to(device='cuda')

    model_name = osp.splitext(osp.basename(ckpt))[0]

    for ii, p in enumerate(tqdm(exec_list[0:])):
        try:

            # instance_mask, crop_xyxy, score = load_detected_character(p)
            # if instance_mask is None:
            #     print(f'skip {p}, no character instance detected')
            #     continue
            
            # lmodel = Live2DScrapModel(p, crop_xyxy=crop_xyxy, pad_to_square=False)
            # lmodel.init_drawable_visible_map()
            # final_img = compose_from_drawables(lmodel.drawables)

            img = np.array(Image.open(p).convert('RGB'))
            with torch.inference_mode():
                preds = model.inference(img)[0]
                masks_np = (preds > 0).to(device='cpu', dtype=torch.bool).numpy()

            # save_tmp_img(visualize_segs_with_labels(masks_np, final_img[..., :3], VALID_BODY_PARTS_V1, reference_img=final_img[..., :3]))
            # print(f'save to ' + osp.join(model_dir, f'{model_name}_masks.json'))
            if save_to_local:
                saved = osp.dirname(p)
            else:
                saved = save_dir
            batch_save_masks(masks_np, osp.join(saved, f'{osp.basename(p)}_masks.json'))


        except Exception as e:
            raise
            print(f'Failed to process {p}: {e}')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='local_configs/evalsam_iter1.yaml')

    args = parser.parse_args()

    sam_parse_body_samples(args.config)