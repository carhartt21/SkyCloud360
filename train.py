"""
Training script for SkyCloudNet360.

Supports training with all four 360° adaptation strategies:
- SkyCloudNet (baseline)
- SkyCloudNet-CM  (standard cubemap)
- SkyCloudNet-ECM (extended cubemap)
- SkyCloudNet-TPP (tangent plane projection)
- SkyCloudNet-EQC (equirectangular convolutions)

Usage:
    # Baseline training
    python3 train.py --cfg config/config.yaml

    # Training with tangent plane projection
    python3 train.py --cfg config/config.yaml MODEL.equirect_mode tpp

    # Training with equirectangular convolutions
    python3 train.py --cfg config/config.yaml MODEL.equirect_mode eqc
"""

import os
import time
import math
import argparse
from packaging import version
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from PIL import Image
from tqdm import tqdm

# Our libs
from config import cfg
from data import TrainDataset, ValDataset
from model import ModelBuilder, MultiLabelModule, UnsupervisedSegmentationModule
from model_360 import build_360_model
from utils import (
    AverageMeter, colorEncode, accuracy, intersectionAndUnion,
    setup_logger, fix_seed_for_reproducability
)
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy


def adjust_learning_rate(optimizers, cur_iter, max_iters, cfg):
    """Polynomial learning rate decay."""
    scale_running_lr = ((1. - float(cur_iter) / max_iters) ** cfg.TRAIN.lr_pow)
    
    lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    lr_decoder_seg = cfg.TRAIN.lr_decoder_seg * scale_running_lr
    lr_decoder_attr = cfg.TRAIN.lr_decoder_attr * scale_running_lr

    for param_group in optimizers[0].param_groups:
        param_group['lr'] = lr_encoder
    for param_group in optimizers[1].param_groups:
        param_group['lr'] = lr_decoder_seg
    for param_group in optimizers[2].param_groups:
        param_group['lr'] = lr_decoder_attr


def checkpoint(nets, epoch, cfg):
    """Save model checkpoints."""
    print('Saving checkpoints...')
    (net_encoder, net_decoder_skyseg, net_decoder_attr, net_decoder_cloudseg) = nets
    
    save_dir = cfg.DIR
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(
        net_encoder.state_dict(),
        os.path.join(save_dir, f'encoder_epoch_{epoch}.pth'))
    torch.save(
        net_decoder_skyseg.state_dict(),
        os.path.join(save_dir, f'decoder_seg_epoch_{epoch}.pth'))
    torch.save(
        net_decoder_attr.state_dict(),
        os.path.join(save_dir, f'decoder_attr_epoch_{epoch}.pth'))
    torch.save(
        net_decoder_cloudseg.state_dict(),
        os.path.join(save_dir, f'decoder_cloudseg_epoch_{epoch}.pth'))


def evaluate_during_training(multilabel_module, cloud_seg_module, loader, cfg, gpu,
                              model_360=None, logger=None):
    """Quick evaluation pass during training."""
    use_360 = model_360 is not None and cfg.MODEL.equirect_mode != 'none'

    multilabel_module.eval()
    cloud_seg_module.eval()
    if use_360:
        model_360.eval()

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data[0]
            sky_seg_label = as_numpy(batch_data['seg_label'][0])
            img_resized_list = batch_data['img_data']

            seg_size = (sky_seg_label.shape[0], sky_seg_label.shape[1])

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                if 'info' in feed_dict:
                    del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                if use_360:
                    pred_seg, _ = model_360(feed_dict, seg_size=seg_size)
                else:
                    encoder_result, [_, pred_seg] = multilabel_module(feed_dict, seg_size=seg_size)

            _, sky_pred = torch.max(pred_seg, dim=1)
            sky_pred = as_numpy(sky_pred.squeeze(0).cpu())
            acc, pix = accuracy(sky_pred, sky_seg_label)
            intersection, union = intersectionAndUnion(
                sky_pred, sky_seg_label, cfg.DATASET.num_seg_class)

            acc_meter.update(acc, pix)
            intersection_meter.update(intersection)
            union_meter.update(union)

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    miou = iou.mean()
    acc_avg = acc_meter.average()

    if logger:
        logger.info(f'  [Val] Accuracy: {acc_avg * 100:.2f}%, mIoU: {miou:.4f}')

    return miou, acc_avg


def train_one_epoch(multilabel_module, cloud_seg_module, loader_train, optimizers,
                    epoch, cfg, gpu, model_360=None, logger=None):
    """Train for one epoch."""
    use_360 = model_360 is not None and cfg.MODEL.equirect_mode != 'none'

    multilabel_module.train()
    cloud_seg_module.train()
    if use_360 and hasattr(model_360, 'train'):
        model_360.train()

    # Freeze attribute estimation head weights (as described in paper)
    for param in multilabel_module.decoder_attributes.parameters():
        param.requires_grad = False

    loss_sky_meter = AverageMeter()
    loss_cloud_meter = AverageMeter()
    loss_attr_meter = AverageMeter()
    acc_meter = AverageMeter()
    time_meter = AverageMeter()

    max_iters = cfg.TRAIN.num_epoch * cfg.TRAIN.epoch_iters

    iterator = iter(loader_train)
    pbar = tqdm(total=cfg.TRAIN.epoch_iters, ascii=True,
                desc=f'Epoch {epoch}/{cfg.TRAIN.num_epoch}')

    for i in range(cfg.TRAIN.epoch_iters):
        cur_iter = epoch * cfg.TRAIN.epoch_iters + i
        adjust_learning_rate(optimizers, cur_iter, max_iters, cfg)

        try:
            batch_data = next(iterator)
        except StopIteration:
            iterator = iter(loader_train)
            batch_data = next(iterator)

        tic = time.perf_counter()

        if torch.cuda.is_available():
            batch_data['img_data'] = batch_data['img_data'].cuda()
            batch_data['seg_label'] = batch_data['seg_label'].cuda()
            batch_data['cloud_label'] = batch_data['cloud_label'].cuda()
            batch_data['attr_label'] = batch_data['attr_label'].cuda()

        # Forward pass through sky segmentation + attribute module
        if use_360 and cfg.MODEL.equirect_mode == 'eqc':
            # For EQC mode, the convolutions are modified in-place,
            # so we use the standard forward pass which now uses equirect convolutions
            encoder_result, loss_sky, loss_attr, acc_sky, pred_seg, pred_attr = \
                multilabel_module(batch_data)
        else:
            encoder_result, loss_sky, loss_attr, acc_sky, pred_seg, pred_attr = \
                multilabel_module(batch_data)

        # Forward pass through cloud segmentation module
        pred_cloud, loss_cloud = cloud_seg_module(
            batch_data, encoder_result, pred_seg)

        # Total loss
        loss = loss_sky + loss_cloud
        if isinstance(loss_attr, torch.Tensor):
            loss = loss + loss_attr * 0.1  # downweight attribute loss

        # Backward pass
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # Update meters
        if isinstance(loss_sky, torch.Tensor):
            loss_sky_meter.update(loss_sky.item())
        if isinstance(loss_cloud, torch.Tensor):
            loss_cloud_meter.update(loss_cloud.item())
        if isinstance(loss_attr, torch.Tensor):
            loss_attr_meter.update(loss_attr.item())
        acc_meter.update(acc_sky.item() if isinstance(acc_sky, torch.Tensor) else acc_sky)
        time_meter.update(time.perf_counter() - tic)

        # Display
        if (i + 1) % cfg.TRAIN.disp_iter == 0 and logger:
            logger.info(
                f'Epoch [{epoch}][{i+1}/{cfg.TRAIN.epoch_iters}] '
                f'Loss_sky: {loss_sky_meter.average():.4f} '
                f'Loss_cloud: {loss_cloud_meter.average():.4f} '
                f'Loss_attr: {loss_attr_meter.average():.4f} '
                f'Acc: {acc_meter.average() * 100:.2f}% '
                f'Time: {time_meter.average():.3f}s'
            )

        pbar.update(1)

    pbar.close()
    return loss_sky_meter.average(), acc_meter.average()


def main(cfg, gpu):
    torch.cuda.set_device(gpu)
    fix_seed_for_reproducability(cfg.TRAIN.seed)

    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder if hasattr(cfg.MODEL, 'weights_encoder') 
                and cfg.MODEL.weights_encoder else '')
    net_decoder_attr = builder.build_decoder(
        arch='attribute_head',
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_attr_class,
        weights=cfg.MODEL.weights_decoder_attr)
    net_decoder_skyseg = builder.build_decoder(
        arch='sky_seg_head',
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_seg_class,
        weights=cfg.MODEL.weights_decoder_skyseg)
    net_decoder_cloudseg = builder.build_decoder(
        arch='cloud_seg_head',
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_seg_class + cfg.DATASET.num_cloud_class,
        weights=cfg.MODEL.weights_decoder_cloudseg)

    crit_sky = nn.CrossEntropyLoss(ignore_index=-1)
    crit_attr = nn.CrossEntropyLoss(ignore_index=-1)

    multilabel_module = MultiLabelModule(
        net_encoder, net_decoder_skyseg, net_decoder_attr, crit_sky, crit_attr)

    cloud_seg_module = UnsupervisedSegmentationModule(net_decoder_cloudseg)

    # Build 360° adaptation model if specified
    equirect_mode = cfg.MODEL.equirect_mode
    model_360 = None
    if equirect_mode != 'none':
        logger.info(f'Building 360° adaptation model: SkyCloudNet-{equirect_mode.upper()}')
        kwargs_360 = {}
        if equirect_mode in ('cm', 'ecm'):
            kwargs_360['face_size'] = cfg.MODEL.cubemap_face_size
        if equirect_mode == 'ecm':
            kwargs_360['overlap'] = cfg.MODEL.ecm_overlap
        if equirect_mode == 'tpp':
            kwargs_360['patch_size'] = cfg.MODEL.tpp_patch_size
            kwargs_360['fov'] = math.radians(cfg.MODEL.tpp_fov_deg)
            kwargs_360['num_lat'] = cfg.MODEL.tpp_num_lat
            kwargs_360['num_lon'] = cfg.MODEL.tpp_num_lon
        if equirect_mode == 'eqc':
            kwargs_360['replace_encoder'] = cfg.MODEL.eqc_replace_encoder
            kwargs_360['replace_decoder'] = cfg.MODEL.eqc_replace_decoder

        model_360 = build_360_model(multilabel_module, cloud_seg_module,
                                     mode=equirect_mode, **kwargs_360)

    # Move to GPU
    multilabel_module.cuda()
    cloud_seg_module.cuda()
    if model_360 is not None:
        model_360.cuda()

    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,  # batch handled internally by TrainDataset
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)

    # Validation loader (optional)
    loader_val = None
    if cfg.TRAIN.eval and os.path.exists(cfg.DATASET.list_val):
        dataset_val = ValDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_val,
            cfg.DATASET)
        loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=cfg.VAL.batch_size,
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=5,
            drop_last=True)

    # Optimizers
    # Encoder uses a lower learning rate when fine-tuning with pretrained weights
    optimizer_encoder = SGD(
        net_encoder.parameters(),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder_seg = SGD(
        list(net_decoder_skyseg.parameters()) + list(net_decoder_cloudseg.parameters()),
        lr=cfg.TRAIN.lr_decoder_seg,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder_attr = SGD(
        net_decoder_attr.parameters(),
        lr=cfg.TRAIN.lr_decoder_attr,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)

    optimizers = [optimizer_encoder, optimizer_decoder_seg, optimizer_decoder_attr]

    # Load optimizer state if resuming
    if cfg.TRAIN.optim_data and os.path.exists(cfg.TRAIN.optim_data):
        logger.info(f'Loading optimizer state from {cfg.TRAIN.optim_data}')
        optim_state = torch.load(cfg.TRAIN.optim_data)
        for i, opt in enumerate(optimizers):
            if f'optimizer_{i}' in optim_state:
                opt.load_state_dict(optim_state[f'optimizer_{i}'])

    # Fix batch norm if specified
    if cfg.TRAIN.fix_bn:
        for module in multilabel_module.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()

    best_score = cfg.TRAIN.best_score

    # Training loop
    logger.info('='*60)
    logger.info(f'Starting training: {cfg.TRAIN.num_epoch} epochs, '
                f'{cfg.TRAIN.epoch_iters} iters/epoch')
    logger.info(f'360° mode: {equirect_mode}')
    logger.info('='*60)

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        # Train
        loss, acc = train_one_epoch(
            multilabel_module, cloud_seg_module, loader_train,
            optimizers, epoch, cfg, gpu, model_360=model_360, logger=logger)

        logger.info(f'Epoch {epoch} completed - Loss: {loss:.4f}, Acc: {acc*100:.2f}%')

        # Evaluate
        if cfg.TRAIN.eval and loader_val is not None and \
                (epoch + 1) % cfg.TRAIN.eval_step == 0:
            miou, val_acc = evaluate_during_training(
                multilabel_module, cloud_seg_module, loader_val, cfg, gpu,
                model_360=model_360, logger=logger)

            if miou > best_score:
                best_score = miou
                logger.info(f'  New best mIoU: {best_score:.4f} - saving checkpoint')
                checkpoint(
                    (net_encoder, net_decoder_skyseg, net_decoder_attr, net_decoder_cloudseg),
                    f'best', cfg)

        # Save checkpoint every epoch
        checkpoint(
            (net_encoder, net_decoder_skyseg, net_decoder_attr, net_decoder_cloudseg),
            epoch + 1, cfg)

        # Save optimizer state
        optim_state = {f'optimizer_{i}': opt.state_dict()
                      for i, opt in enumerate(optimizers)}
        torch.save(optim_state, os.path.join(cfg.DIR, 'optimizer_state.pth'))

    logger.info(f'Training complete! Best mIoU: {best_score:.4f}')


if __name__ == '__main__':
    assert version.Version(torch.__version__) >= version.Version('1.4.0'), \
        'PyTorch>=1.4.0 is required'

    parser = argparse.ArgumentParser(
        description='SkyCloudNet360 Training'
    )
    parser.add_argument(
        '--cfg',
        default='config/config.yaml',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='gpu to use'
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)

    logger = setup_logger(distributed_rank=0)
    logger.info('Loaded configuration file {}'.format(args.cfg))
    logger.info(f'360° adaptation mode: {cfg.MODEL.equirect_mode}')

    # Set up weight paths for loading pretrained weights (if available)
    cfg.defrost()
    if not hasattr(cfg.MODEL, 'weights_encoder') or not cfg.MODEL.weights_encoder:
        encoder_path = os.path.join(cfg.DIR, f'encoder_{cfg.VAL.checkpoint}')
        if os.path.exists(encoder_path):
            cfg.MODEL.weights_encoder = encoder_path
        else:
            cfg.MODEL.weights_encoder = ''

    if not cfg.MODEL.weights_decoder_attr:
        attr_path = os.path.join(cfg.DIR, f'decoder_attr_{cfg.VAL.checkpoint}')
        if os.path.exists(attr_path):
            cfg.MODEL.weights_decoder_attr = attr_path

    if not cfg.MODEL.weights_decoder_skyseg:
        seg_path = os.path.join(cfg.DIR, f'decoder_seg_{cfg.VAL.checkpoint}')
        if os.path.exists(seg_path):
            cfg.MODEL.weights_decoder_skyseg = seg_path

    if not cfg.MODEL.weights_decoder_cloudseg:
        cloud_path = os.path.join(cfg.DIR, f'decoder_cloudseg_{cfg.VAL.checkpoint}')
        if os.path.exists(cloud_path):
            cfg.MODEL.weights_decoder_cloudseg = cloud_path

    os.makedirs(cfg.DIR, exist_ok=True)
    cfg.freeze()

    main(cfg, args.gpu)
