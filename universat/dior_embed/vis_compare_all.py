import os
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import rasterio
import numpy as np


DATASET_METAINFO = {
    'classes': [
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6'
    ],
    'palette': [
        [255, 255, 255], 
        [255, 0, 0], 
        [255, 255, 0], 
        [0, 0, 255],
        [159, 129, 183], 
        [0, 255, 0], 
        [255, 195, 128]
    ]
}


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
    axes[2].imshow(draw_sem_seg(original_img, result_img, dmeta['classes'], dmeta['palette']))
    axes[2].set_title('model pred', fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return plt.close()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'datadir',
        type=str
    )
    parser.add_argument(
        'results',
        type=str
    )
    parser.add_argument(
        'compare',
        type=str
    )
    args = parser.parse_args()
    os.makedirs(args.compare, exist_ok=True)

    test_input_dir = os.path.join(args.datadir, 'img_dir')
    test_label_dir = os.path.join(args.datadir, 'ann_dir')
    test_infer_dir = args.results

    for fn in os.listdir(test_infer_dir):
        ifn = os.path.join(test_input_dir, f"{os.path.splitext(fn)[0]}.tif")
        lfn = os.path.join(test_label_dir, fn)
        rfn = os.path.join(test_infer_dir, fn)
        with rasterio.open(ifn) as tif:
            bands = tif.read()
        if bands.ndim == 3:
            bands = np.moveaxis(bands, 0, 2)
        i_data = bands[..., [3, 2, 1]]
        i_data = (i_data-i_data.min()) / (i_data.max()-i_data.min())
        i_data *= 255
        i_data = np.uint8(i_data)
        with rasterio.open(lfn) as f:
            l_data = f.read()
        with rasterio.open(rfn) as f:
            r_data = f.read()
        r_data = np.repeat(r_data, 3, axis=0)
        fn = os.path.join(args.compare, f"{Path(ifn).stem}.png")
        visualize_comparison(i_data, l_data, r_data, fn, DATASET_METAINFO)


if '__main__' == __name__:
    main()
