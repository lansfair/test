import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import rasterio
import cv2
import numpy as np
from tqdm import tqdm
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot, MMSegInferencer


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
    axes[1].imshow(draw_sem_seg(original_img, mask_img, dmeta['classes'], dmeta['palette']))
    axes[1].set_title('ground truth', fontsize=12)
    axes[1].axis('off')

    # plot result
    axes[2].imshow(result_img)
    axes[2].set_title('model pred', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return plt.close()


def compare_single(inferencer, img_filepath, mask_filepath, dmeta, adir='assets', rdir='resutls', cdir='compare'):
    result = inferencer(img_filepath, out_dir=rdir, show=False)

    with rasterio.open(img_filepath) as tif:
        bands = tif.read()
    if bands.ndim == 3:
        bands = np.moveaxis(bands, 0, 2)
    rgb_img = bands[..., [3, 2, 1]]
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    rgb_img *= 255
    rgb_img = np.uint8(rgb_img)

    img_filename = os.path.split(img_filepath)[-1]
    cv2.imwrite(os.path.join(filepath, adir, f"{os.path.splitext(img_filename)[0]}.png"), rgb_img)
    
    # generate seg result
    src_filename = f"{os.path.splitext(img_filename)[0]}.png"
    src_filepath = os.path.join(filepath, adir, src_filename)
    dst_filepath = os.path.join(filepath, rdir, src_filename)
    show_result_pyplot(model, src_filepath, result, show=False, out_file=dst_filepath, with_labels=False)

    cpr_filepath = os.path.join(filepath, cdir, src_filename)

    # read result
    result_bgr = cv2.imread(dst_filepath)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # read ground truth
    with rasterio.open(mask_filepath) as mask_src:
        mask_data = mask_src.read()

    return visualize_comparison(rgb_img, mask_data, result_rgb, cpr_filepath, dmeta)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'dataset',
        type=str
    )
    parser.add_argument(
        'config',
        type=str
    )
    parser.add_argument(
        '-w', '--weight',
        type=str,
        default=''
    )
    parser.add_argument(
        '--work_dir',
        type=str,
        dest='work_dir',
        default='.'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=None
    )
    args = parser.parse_args()

    tif_dir = os.path.join(args.dataset, 'img_dir')
    tif_files = sorted(os.listdir(tif_dir))
    msk_dir = os.path.join(args.dataset, 'ann_dir')
    msk_files = sorted(os.listdir(msk_dir))
    configfile = Path(args.config)
    weightfile = args.weight if args.weight else os.path.join('/mnt/ht2-nas2/EO_test/openmmlab-archive/src/v1/mmseg/projects/olmoearth/m-sa-crop-type/checkpoints', f'{configfile.stem}.pth')
    workdir = os.path.join(args.work_dir, 'plot', configfile.stem)

    origins_dir = os.path.join(workdir, 'origins')
    results_dir = os.path.join(workdir, 'results')
    compare_dir = os.path.join(workdir, 'compare')
    os.makedirs(origins_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    device = 'cpu' if args.device is None else f'cuda:{args.device}'

    # model = init_model(configfile, weightfile, device=device)
    # if not torch.cuda.is_available():
    #     model = revert_sync_batchnorm(model)
    inferencer = MMSegInferencer(model=str(configfile), weights=str(weightfile), device=device)

    prog = tqdm(range(len(tif_files)))
    for tif_name, msk_name in zip(tif_files, msk_files):
        tif_path = os.path.join(tif_dir, tif_name)
        msk_path = os.path.join(msk_dir, msk_name)
        compare_single(inferencer, tif_path, msk_path, model.dataset_meta, origins_dir, results_dir, compare_dir)
        prog.update()


if '__main__' == __name__:
    main()
    # tif_dir = "/mnt/ht2-nas2/EO_test/zyf/data/geobench2tif/m-cashew-plant/img_dir/test"
    # msk_dir = "/mnt/ht2-nas2/EO_test/zyf/data/geobench2tif/m-cashew-plant/ann_dir/test"

    # cfg_name = 'cashew-plant_copernicus-fm-base_lp_wft_lr-5e-4_e50'
    # config_file = os.path.join(filepath, 'configs', f'{cfg_name}.py')
    # checkpoint_file = os.path.join(filepath, 'checkpoints', f'{cfg_name}.pth')

    # origins_dir = f'assets/{cfg_name}'
    # results_dir = f'results/{cfg_name}'
    # compare_dir = f'compare/{cfg_name}'

    # os.makedirs(os.path.join(filepath, origins_dir), exist_ok=True)
    # os.makedirs(os.path.join(filepath, results_dir), exist_ok=True)
    # os.makedirs(os.path.join(filepath, compare_dir), exist_ok=True)

    # model = init_model(config_file, checkpoint_file, device='cpu')
    # if not torch.cuda.is_available():
    #     model = revert_sync_batchnorm(model)

    # tif_files = sorted(os.listdir(tif_dir))
    # msk_files = sorted(os.listdir(msk_dir))

    # prog = tqdm(range(len(tif_files)))
    # for tif_name, msk_name in zip(tif_files, msk_files):
    #     tif_path = os.path.join(tif_dir, tif_name)
    #     msk_path = os.path.join(msk_dir, msk_name)
    #     compare_single(model, tif_path, msk_path, model.dataset_meta, origins_dir, results_dir, compare_dir)
    #     prog.update()