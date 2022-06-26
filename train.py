import os
import time
import argparse
from datetime import datetime

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np

import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

from datasets.coco import COCO, COCO_eval
from datasets.pascal import PascalVOC, PascalVOC_eval

# from nets.resdcn import get_pose_net
from nets.hourglass import get_hourglass

from utils.losses import _neg_loss, _ae_loss, _reg_loss, Loss
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.keypoint import _decode, _rescale_dets, _tranpose_and_gather_feature

from lib.nms.nms import soft_nms, soft_nms_merge



# Training settings
parser = argparse.ArgumentParser(description='cornernet')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='/mnt/e/data/image_detection')
parser.add_argument('--log_name', type=str, default='test')

parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'pascal'])
parser.add_argument('--arch', type=str, default='tiny_hourglass')

parser.add_argument('--img_size', type=int, default=511)
parser.add_argument('--split_ratio', type=float, default=1.0)

parser.add_argument('--lr', type=float, default=2.5e-4)
parser.add_argument('--lr_step', type=str, default='45,60')

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=70)

parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=1)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]


def main():
  saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
  logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
  summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
  print = logger.info
  print(cfg)

  torch.manual_seed(100)
  torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training

  num_gpus = torch.cuda.device_count()
  if cfg.dist:
    cfg.device = torch.device('cuda:%d' % cfg.local_rank)
    torch.cuda.set_device(cfg.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=num_gpus, rank=cfg.local_rank)
  else:
    cfg.device = torch.device('cuda:0')

  print('Setting up data...')
  Dataset = COCO if cfg.dataset == 'coco' else PascalVOC
  train_dataset = Dataset(cfg.data_dir, 'train', split_ratio=cfg.split_ratio, img_size=cfg.img_size)
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                  num_replicas=num_gpus,
                                                                  rank=cfg.local_rank)
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.batch_size // num_gpus
                                             if cfg.dist else cfg.batch_size,
                                             shuffle=not cfg.dist,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True,
                                             drop_last=True,
                                             sampler=train_sampler if cfg.dist else None)

  Dataset_eval = COCO_eval if cfg.dataset == 'coco' else PascalVOC_eval
  val_dataset = Dataset_eval(cfg.data_dir, 'val', test_scales=[1.], test_flip=False)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                           shuffle=False, num_workers=1, pin_memory=True,
                                           collate_fn=val_dataset.collate_fn)

  print('Creating model...')
  if 'hourglass' in cfg.arch:
    model = get_hourglass[cfg.arch]
  elif 'resdcn' in cfg.arch:
    model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]), num_classes=train_dataset.num_classes)
  else:
    raise NotImplementedError

  if cfg.dist:
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(cfg.device)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[cfg.local_rank, ],
                                                output_device=cfg.local_rank)
  else:
    # todo don't use this, or wrapped it with utils.losses.Loss() !
    model = model.to(cfg.device)

  optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
  lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.1)

  def train(epoch):
    print('\n%s Epoch: %d' % (datetime.now(), epoch))
    model.train()
    tic = time.perf_counter()
    for batch_idx, batch in enumerate(train_loader):
      for k in batch:
        batch[k] = batch[k].to(device=cfg.device, non_blocking=True)

      # batch: {'image': image,
      #  'hmap_tl': hmap_tl, 'hmap_br': hmap_br,
      #  'regs_tl': regs_tl, 'regs_br': regs_br,
      #  'inds_tl': inds_tl, 'inds_br': inds_br,
      #  'ind_masks': ind_masks}
      outputs = model(batch['image'])
      # 如果batch=2的情况下, 由于是zip(*outputs), 所以输出结果是tuple
      # hmap_tl = tuple(torch.Size([2, 80, 128, 128])), tuple里面只有一个tensor
      # hmap_br = tuple(torch.Size([2, 80, 128, 128])), tuple里面只有一个tensor
      # embd_tl = tuple(torch.Size([2, 1, 128, 128])), tuple里面只有一个tensor
      # embd_br = tuple(torch.Size([2, 1, 128, 128])), tuple里面只有一个tensor
      # regs_tl = tuple(torch.Size([2, 2, 128, 128])), tuple里面只有一个tensor
      # regs_br = tuple(torch.Size([2, 2, 128, 128])), tuple里面只有一个tensor
      hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br = zip(*outputs)
      # 注意batch['inds_tl']是多个batch, 假设batch=2, 那么batch['inds_tl'] = (2,128)
      # (2, 128, 1), 也就是embedding是一个128维度的向量, 一张图像的所有的

      embd_tl = [_tranpose_and_gather_feature(e, batch['inds_tl']) for e in embd_tl] # batch['inds_tl']=(2,128)
      embd_br = [_tranpose_and_gather_feature(e, batch['inds_br']) for e in embd_br]

      # regs_tl = tuple(torch.Size([2, 2, 128, 128]))
      # regs_br = tuple(torch.Size([2, 2, 128, 128])), tuple里面只有一个tensor
      regs_tl = [_tranpose_and_gather_feature(r, batch['inds_tl']) for r in regs_tl]
      regs_br = [_tranpose_and_gather_feature(r, batch['inds_br']) for r in regs_br]


      # loss 相关
      # batch['hmap_tl'] = (80, 128, 128)
      focal_loss = _neg_loss(hmap_tl, batch['hmap_tl']) + \
                   _neg_loss(hmap_br, batch['hmap_br'])

      # regs_tl
      reg_loss = _reg_loss(regs_tl, batch['regs_tl'], batch['ind_masks']) + \
                 _reg_loss(regs_br, batch['regs_br'], batch['ind_masks'])  # 这一项是offset loss
      pull_loss, push_loss = _ae_loss(embd_tl, embd_br, batch['ind_masks']) # 计算embedding的

      loss = focal_loss + 0.1 * pull_loss + 0.1 * push_loss + reg_loss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        duration = time.perf_counter() - tic
        tic = time.perf_counter()
        print('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
              ' focal_loss= %.5f pull_loss= %.5f push_loss= %.5f reg_loss= %.5f' %
              (focal_loss.item(), pull_loss.item(), push_loss.item(), reg_loss.item()) +
              ' (%d samples/sec)' % (cfg.batch_size * cfg.log_interval / duration))

        step = len(train_loader) * epoch + batch_idx
        summary_writer.add_scalar('focal_loss', focal_loss.item(), step)
        summary_writer.add_scalar('pull_loss', pull_loss.item(), step)
        summary_writer.add_scalar('push_loss', push_loss.item(), step)
        summary_writer.add_scalar('reg_loss', reg_loss.item(), step)
    return

  def val_map(epoch):
    print('\n%s Val@Epoch: %d' % (datetime.now(), epoch))
    model.eval()
    # torch.cuda.empty_cache()

    results = {}
    with torch.no_grad():
      for inputs in val_loader:
        img_id, inputs = inputs[0]

        detections = []
        for scale in inputs:
          inputs[scale]['image'] = inputs[scale]['image'].to(cfg.device)
          output = model(inputs[scale]['image'])[-1]
          det = _decode(*output, ae_threshold=0.5, K=100, kernel=3)
          det = det.reshape(det.shape[0], -1, 8).detach().cpu().numpy()
          if det.shape[0] == 2:
            det[1, :, [0, 2]] = inputs[scale]['fmap_size'][0, 1] - det[1, :, [2, 0]]
          det = det.reshape(1, -1, 8)

          _rescale_dets(det, inputs[scale]['ratio'], inputs[scale]['border'], inputs[scale]['size'])
          det[:, :, 0:4] /= scale
          detections.append(det)

        detections = np.concatenate(detections, axis=1)[0]
        # reject detections with negative scores
        detections = detections[detections[:, 4] > -1]
        classes = detections[..., -1]

        results[img_id] = {}
        for j in range(val_dataset.num_classes):
          keep_inds = (classes == j)
          results[img_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
          soft_nms_merge(results[img_id][j + 1], Nt=0.5, method=2, weight_exp=10)
          # soft_nms(results[img_id][j + 1], Nt=0.5, method=2)
          results[img_id][j + 1] = results[img_id][j + 1][:, 0:5]

        scores = np.hstack([results[img_id][j][:, -1] for j in range(1, val_dataset.num_classes + 1)])
        if len(scores) > val_dataset.max_objs:
          kth = len(scores) - val_dataset.max_objs
          thresh = np.partition(scores, kth)[kth]
          for j in range(1, val_dataset.num_classes + 1):
            keep_inds = (results[img_id][j][:, -1] >= thresh)
            results[img_id][j] = results[img_id][j][keep_inds]

    eval_results = val_dataset.run_eval(results, save_dir=cfg.ckpt_dir)
    print(eval_results)
    summary_writer.add_scalar('val_mAP/mAP', eval_results[0], epoch)

  print('Starting training...')
  for epoch in range(1, cfg.num_epochs + 1):
    # train_sampler.set_epoch(epoch)
    train(epoch)
    if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
      val_map(epoch)
    print(saver.save(model.module.state_dict(), 'checkpoint'))
    lr_scheduler.step(epoch)  # move to here after pytorch1.1.0

  summary_writer.close()


if __name__ == '__main__':
  # with DisablePrint(local_rank=cfg.local_rank):
  main()
