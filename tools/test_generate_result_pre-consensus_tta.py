# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import imp
import os
import warnings
from numbers import Number
from collections import defaultdict
import os.path as osp

import mmcv
import numpy as np
import torch
import torchvision as tv
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import get_root_logger, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('out', help='output result file')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--device', default=None, help='device used for testing. (Deprecated)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    args = parser.parse_args()

    return args


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    **show_kwargs):
    """Test model with local single gpu.

    This method tests model with a single gpu and supports showing results.

    Args:
        model (:obj:`torch.nn.Module`): Model to be tested.
        data_loader (:obj:`torch.utils.data.DataLoader`): Pytorch data loader.
        show (bool): Whether to show the test results. Defaults to False.
        out_dir (str): The output directory of result plots of all samples.
            Defaults to None, which means not to write output files.
        **show_kwargs: Any other keyword arguments for showing results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            imgs = data['img'].cuda()
            result1 = model.module.extract_feat(imgs)[0]
            imgs = torch.flip(imgs, [3])
            result2 = model.module.extract_feat(imgs)[0]
            result = torch.stack([result1, result2], dim=0)
            result = torch.mean(result, dim=0)
            result = (result,)

        filenames = [x['filename'] for x in data['img_metas'].data[0]]
        obs_ids = [osp.basename(x).split('.')[0].split('-')[1] for x in filenames]
        result = list(zip(result[0], obs_ids))

        batch_size = len(result)
        results.extend(result)

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES
    if CLASSES is None:
        CLASSES = dataset.CLASSES

    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        if not model.device_ids:
            assert mmcv.digit_version(mmcv.__version__) >= (1, 4, 4), \
                'To test with CPU, please confirm your mmcv version ' \
                'is not lower than v1.4.4'
    model.CLASSES = CLASSES
    outputs = single_gpu_test(model, data_loader)

    results = defaultdict(list)
    for result, obs_id in outputs:
        results[obs_id].append(result)
    
    dropped = 0
    total = 0
    with open(args.out, 'w') as f, open(args.out + '.scores.csv', 'w') as f2:
        f.write('ObservationId,ClassId\n')
        for obs_id, result in results.items():
            avg_feats = torch.mean(torch.stack(result, dim=0), dim=0, keepdim=True)
            scores = model.module.head.simple_test(avg_feats)
            class_id = np.argmax(scores)
            if np.max(scores) < 0.1:
                class_id = -1
                dropped += 1
            total += 1
            f.write(f'{obs_id},{class_id}\n')
            f2.write(f'{obs_id}')
            for s in scores[0]:
                f2.write(f',{s}')
            f2.write('\n')

    print(f'dropped {dropped} out of {total}')


if __name__ == '__main__':
    main()
