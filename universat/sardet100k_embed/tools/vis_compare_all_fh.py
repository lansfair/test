import os
import argparse

import matplotlib.pyplot as plt
import torch
import rasterio
import cv2
import numpy as np
from tqdm import tqdm
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot


filepath = os.path.dirname(__file__)


def draw_sem_seg(image, sem_seg, classes, palette, alpha=0.8):
    num_classes = len(classes)
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)
    colors = [palette[label] for label in labels]
    mask = np.zeros_like(image, dtype=np.uint8)
    for label, color in zip(labels, colors):
        mask[sem_seg[0] == label, :] = color
    return (image*(1-alpha) + mask*alpha).astype(np.uint8)


def visualize_comparison(original_img, mask_img, result_img, save_path, dmeta):
    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    # plot origin image
    axes[0].imshow(original_img)
    axes[0].set_title('original image', fontsize=12)
    axes[0].axis('off')

    # plot mask
    # if mask_img.ndim == 2:
    #     im2 = axes[1].imshow(mask_img, cmap='tab20', vmin=0, vmax=19)
    # else:
    #     im2 = axes[1].imshow(mask_img[0,:,:], cmap='tab20')
    axes[1].imshow(draw_sem_seg(original_img, mask_img, dmeta['classes'], dmeta['palette']))
    print(dmeta['palette'])
    axes[1].set_title('ground truth', fontsize=12)
    axes[1].axis('off')

    # plot result
    axes[2].imshow(result_img)
    axes[2].set_title('model pred', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return plt.close()
    # print(f"对比图保存至:{save_path}")


def compare_single(model, img_filepath, mask_filepath, dmeta):
    result = inference_model(model, img_filepath)

    # print(result.pred_sem_seg.data)

    with rasterio.open(img_filepath) as tif:
        bands = tif.read()
    if bands.ndim == 3:
        bands = np.moveaxis(bands, 0, 2)
    rgb_img = bands[..., [3, 2, 1]]
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    rgb_img *= 255
    rgb_img = np.uint8(rgb_img)

    img_filename = os.path.split(img_filepath)[-1]
    cv2.imwrite(os.path.join(filepath, 'assets', f"{os.path.splitext(img_filename)[0]}.png"), rgb_img)
    
    # generate seg result
    src_filename = f"{os.path.splitext(img_filename)[0]}.png"
    src_filepath = os.path.join(filepath, 'assets', src_filename)
    dst_filepath = os.path.join(filepath, 'results', src_filename)
    show_result_pyplot(model, src_filepath, result, show=False, out_file=dst_filepath, with_labels=False)

    cpr_filepath = os.path.join(filepath, 'compare', src_filename)

    # read result
    result_bgr = cv2.imread(dst_filepath)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # read ground truth
    with rasterio.open(mask_filepath) as mask_src:
        mask_data = mask_src.read()

    return visualize_comparison(rgb_img, mask_data, result_rgb, cpr_filepath, dmeta)


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize model performance')
    parser.add_argument('--conf_path', help='load config file path')
    parser.add_argument('--ckpt_path', help='load checkpoint path')
    parser.add_argument('--img_dir', help='the dir to img')
    parser.add_argument('--ann_dir', help='the dir to annotation')
    args = parser.parse_args()
    return args


# if '__main__' == __name__:
#     config_file = os.path.join(filepath, 'configs/cashew-plant_copernicus-fm-base_upernet_e80.py')
#     checkpoint_file = os.path.join(filepath, 'checkpoints/cashew-plant_copernicus-fm-base_upernet_e80.pth')
#     model = init_model(config_file, checkpoint_file, device='cpu')
#     if not torch.cuda.is_available():
#         model = revert_sync_batchnorm(model)

#     tif_dir = "/mnt/ht2-nas2/EO_test/zyf/data/geobench2tif/m-cashew-plant/img_dir/test"
#     msk_dir = "/mnt/ht2-nas2/EO_test/zyf/data/geobench2tif/m-cashew-plant/ann_dir/test"

#     tif_files = sorted(os.listdir(tif_dir))
#     msk_files = sorted(os.listdir(msk_dir))

#     for tif_name, msk_name in tqdm(zip(tif_files, msk_files)):
#         tif_path = os.path.join(tif_dir, tif_name)
#         msk_path = os.path.join(msk_dir, msk_name)
#         # print(tif_path)
#         # print(msk_path)
#         compare_single(model, tif_path, msk_path, model.dataset_meta)


if '__main__' == __name__:

    args = parse_args()

    config_file = args.conf_path # os.path.join(filepath, 'configs/sa-crop-type_copernicus-fm-base_lp_wft_e50_lr_5e-4.py')
    checkpoint_file = args.ckpt_path   # os.path.join(filepath, 'checkpoints/sa-crop-type_copernicus-fm-base_lp_wft_e50_lr_5e-4.pth')
    model = init_model(config_file, checkpoint_file, device='cpu')
    if not torch.cuda.is_available():
        model = revert_sync_batchnorm(model)
    # model.dataset_meta['task'] 
    # print(model)
    tif_dir = args.img_dir
    msk_dir = args.ann_dir

    tif_files = sorted(os.listdir(tif_dir))
    msk_files = sorted(os.listdir(msk_dir))

    for tif_name, msk_name in tqdm(zip(tif_files, msk_files)):
        tif_path = os.path.join(tif_dir, tif_name)
        msk_path = os.path.join(msk_dir, msk_name)
        # print(tif_path)
        # print(msk_path)
        compare_single(model, tif_path, msk_path, model.dataset_meta)