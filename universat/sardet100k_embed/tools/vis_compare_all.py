import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from tqdm import tqdm
from mmengine.dataset import Compose
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, show_result_pyplot


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


def compare_single(model, sample_idx, dmeta, pipeline,
                   adir='assets', rdir='resutls', cdir='compare'):
    ds_cfg = model.cfg.test_dataloader.dataset

    data = dict(
        sample_idx=sample_idx,
        task_name=ds_cfg.task_name,
        benchmark_name=ds_cfg.benchmark_name,
        split="test",
        partition_name=ds_cfg.partition_name,
        geobench_format=ds_cfg.geobench_format,
        geobench_root=ds_cfg.geobench_root,
        dataset_name="crop_type",
        olmoearth_modality="sentinel2_l2a",
        olmoearth_num_timesteps=1,
        olmoearth_band_names=list(ds_cfg.band_names),
    )

    data = pipeline(data)

    raw_image = data['olmoearth_raw_img']
    rgb_img = raw_image[..., [3, 2, 1]]
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    rgb_img *= 255
    rgb_img = np.uint8(rgb_img)

    mask_data = data['gt_seg_map']
    if mask_data.ndim == 2:
        mask_data = mask_data[None, ...]

    batch_data = {"inputs": [data["inputs"]], "data_samples": [data["data_samples"]]}
    with torch.no_grad():
        result = model.test_step(batch_data)
    if isinstance(result, (list, tuple)):
        result = result[0]

    img_name = f"sample_{sample_idx:04d}.png"

    cv2.imwrite(os.path.join(filepath, adir, img_name), rgb_img)

    src_filepath = os.path.join(filepath, adir, img_name)
    dst_filepath = os.path.join(filepath, rdir, img_name)
    show_result_pyplot(model, src_filepath, result, show=False, out_file=dst_filepath, with_labels=False)

    cpr_filepath = os.path.join(filepath, cdir, img_name)

    result_bgr = cv2.imread(dst_filepath)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return visualize_comparison(rgb_img, mask_data, result_rgb, cpr_filepath, dmeta)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='(ignored) dataset directory, data is loaded from model config instead'
    )
    parser.add_argument(
        '--config',
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

    model = init_model(configfile, weightfile, device=device)
    if not torch.cuda.is_available():
        model = revert_sync_batchnorm(model)
    if 'task_name' not in model.dataset_meta:
        model.dataset_meta['task_name'] = 'm-SA-crop_type'
        print("yes")

    pipeline_cfg = [dict(t) for t in model.cfg.test_pipeline]
    for t in pipeline_cfg:
        if t.get('type') == 'LoadGeoBenchS2OfficialNorm':
            t['keep_raw_input'] = True
    pipeline = Compose(pipeline_cfg)

    ds_cfg = model.cfg.test_dataloader.dataset
    try:
        import geobench
        task_iterator = geobench.task_iterator
    except Exception:
        from geobench.task import task_iterator

    for _task in task_iterator(benchmark_name=ds_cfg.benchmark_name):
        if _task.dataset_name == "crop_type":
            gb_dataset = _task.get_dataset(split="test", partition_name="default", format="hdf5")
            num_samples = len(gb_dataset)
            break
    else:
        raise RuntimeError("Cannot find crop_type task in benchmark")

    prog = tqdm(range(num_samples))
    for sample_idx in range(num_samples):
        compare_single(model, sample_idx, model.dataset_meta, pipeline,
                       origins_dir, results_dir, compare_dir)
        prog.update()


if '__main__' == __name__:
    main()
