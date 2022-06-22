import os
import cv2
import json
import math
import numpy as np

import torch
from torch.utils.data import Dataset

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

from utils.image import random_crop, crop_image
from utils.image import color_jittering_, lighting_
from utils.image import draw_gaussian, gaussian_radius

COCO_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
              'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
              'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
              'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
              'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]

COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]


class COCO(Dataset):
  def __init__(self, data_dir, split, split_ratio=1.0, gaussian=True, img_size=511):
    super(COCO, self).__init__()
    self.split = split
    self.gaussian = gaussian

    self.down_ratio = 4
    self.img_size = {'h': img_size, 'w': img_size}  # {'h': 511, 'w': 511}

    # {'h': 128, 'w': 128} 这里是进行了4倍的下采样
    self.fmap_size = {'h': (img_size + 1) // self.down_ratio, 'w': (img_size + 1) // self.down_ratio}
    self.padding = 128

    self.data_rng = np.random.RandomState(123)
    self.rand_scales = np.arange(0.6, 1.4, 0.1)
    self.gaussian_iou = 0.3

    self.data_dir = os.path.join(data_dir, 'coco')
    self.img_dir = os.path.join(self.data_dir, 'images/%s2017' % split)
    if split == 'test':
      self.annot_path = os.path.join(self.data_dir, 'annotations', 'image_info_test-dev2017.json')
    else:
      self.annot_path = os.path.join(self.data_dir, 'annotations', 'instances_%s2017.json' % split)

    self.num_classes = 80
    self.class_name = COCO_NAMES
    self.valid_ids = COCO_IDS
    self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

    self.max_objs = 128
    self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
    self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
    self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
    self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]

    print('==> initializing coco 2017 %s data.' % split)
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()

    if 0 < split_ratio < 1:
      split_size = int(np.clip(split_ratio * len(self.images), 1, len(self.images)))
      self.images = self.images[:split_size]

    self.num_samples = len(self.images)

    print('Loaded %d %s samples' % (self.num_samples, split))

  def __getitem__(self, index):
    img_id = self.images[index]
    image = cv2.imread(os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name']))
    print(os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name']))
    annotations = self.coco.loadAnns(ids=self.coco.getAnnIds(imgIds=[img_id]))

    labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations])
    bboxes = np.array([anno['bbox'] for anno in annotations])
    if len(bboxes) == 0:
      bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
      labels = np.array([0])
    bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy

    sorted_inds = np.argsort(labels, axis=0)
    bboxes = bboxes[sorted_inds]
    labels = labels[sorted_inds]

    # random crop (for training) or center crop (for validation)
    if self.split == 'train':
      image, bboxes = random_crop(image,
                                  bboxes,
                                  random_scales=self.rand_scales,
                                  new_size=self.img_size,
                                  padding=self.padding)
    else:
      image, border, offset = crop_image(image,
                                         center=[image.shape[0] // 2, image.shape[1] // 2],
                                         new_size=[max(image.shape[0:2]), max(image.shape[0:2])])
      bboxes[:, 0::2] += border[2]
      bboxes[:, 1::2] += border[0]

    # resize image and bbox
    height, width = image.shape[:2]
    image = cv2.resize(image, (self.img_size['w'], self.img_size['h']))
    bboxes[:, 0::2] *= self.img_size['w'] / width
    bboxes[:, 1::2] *= self.img_size['h'] / height

    # discard non-valid bboxes
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, self.img_size['w'] - 1)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, self.img_size['h'] - 1)
    keep_inds = np.logical_and((bboxes[:, 2] - bboxes[:, 0]) > 0,
                               (bboxes[:, 3] - bboxes[:, 1]) > 0)
    bboxes = bboxes[keep_inds]
    labels = labels[keep_inds]

    # randomly flip image and bboxes
    if self.split == 'train' and np.random.uniform() > 0.5:
      image[:] = image[:, ::-1, :]
      bboxes[:, [0, 2]] = image.shape[1] - bboxes[:, [2, 0]] - 1

    # # ----------------------------- debug -----------------------------------------
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # for lab, bbox in zip(labels, bboxes):
    #   plt.gca().add_patch(Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1],
    #                                 linewidth=1, edgecolor='r', facecolor='none'))
    #   plt.text(bbox[0], bbox[1], self.class_name[lab + 1],
    #            bbox=dict(facecolor='b', alpha=0.5), fontsize=7, color='w')
    # plt.show()
    # # -----------------------------------------------------------------------------

    image = image.astype(np.float32) / 255.

    # randomly change color and lighting
    if self.split == 'train':
      color_jittering_(self.data_rng, image)
      lighting_(self.data_rng, image, 0.1, self.eig_val, self.eig_vec)

    image -= self.mean
    image /= self.std
    image = image.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]

    # 这里就是构造GT的heatmap, fmap_size = {'h': 128, 'w': 128}, num_classes = 80,
    hmap_tl = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # (80, 128, 128), 针对每一个类别C, top-left heatmap是对关键点top-left的描述, 范围是0~1的
    hmap_br = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # (80, 128, 128), 针对每一个类别C, bottom-right heatmap 关键点的描述

    # 这里是构造GT的offset, 到最后用于offset监督
    # self.max_objs = 128,
    regs_tl = np.zeros((self.max_objs, 2), dtype=np.float32)  # (128, 2). 是记录每一个关键点的top-left的坐标
    regs_br = np.zeros((self.max_objs, 2), dtype=np.float32)  # (128, 2). 是记录每一个关键点的bottom-right的坐标

    # question: 这里是构造GT的embedding相关, 但是这里为什么使用self.max_obj来进行构造呢
    # 我们用self.max_obj = 128的一维向量是不是说明我们假设Heatmap上的全部都是obj, 然后我们的向量也是整个都是obj
    inds_tl = np.zeros((self.max_objs,), dtype=np.int64)  # array(128)记录每一个top-left关键点的ind
    inds_br = np.zeros((self.max_objs,), dtype=np.int64)  # 记录每一个bottom-right关键点的ind

    num_objs = np.array(min(bboxes.shape[0], self.max_objs))  # (22) 这个是第一个annotation里面的box的数量, 有22个, 说明有22个待检测的物体
    ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)  # (128) 把所有的物体先初步设置为0
    ind_masks[:num_objs] = 1  # (128), 把那些有物体的设置成1. 也就是向量128, value=0的前22个物体设置成1

    for i, ((xtl, ytl, xbr, ybr), label) in enumerate(zip(bboxes, labels)):
      fxtl = (xtl * self.fmap_size['w'] / self.img_size['w'])  # xtl(top-left的x坐标) = 298 * 128 / 511, fxtl = 27.85, 这个让让top-left的x坐标映射到w=128的heatmap尺寸下
      fytl = (ytl * self.fmap_size['h'] / self.img_size['h'])  # top-left的y坐标也是映射到h=128的维度下, fytl=22.97
      fxbr = (xbr * self.fmap_size['w'] / self.img_size['w'])  # bottom-right的y坐标也是如此, fxbr = 74.29
      fybr = (ybr * self.fmap_size['h'] / self.img_size['h'])  # fybr = 70.1

      # 前面有一个i(比如ixtl中的i), 应该是对 top-left GT做的向下取整, 我们使用int(num),就是对数值进行向下取整.
      ixtl = int(fxtl)  # top left x = 27,
      iytl = int(fytl)  # top right y = 22
      ixbr = int(fxbr)  # bottom right x = 74
      iybr = int(fybr)  # bottom right y = 70

      if self.gaussian:
        width = xbr - xtl  # 绝对坐标 bottom-right的x坐标和绝对坐标的 top-left的x坐标的差值. width = 11
        height = ybr - ytl

        width = math.ceil(width * self.fmap_size['w'] / self.img_size['w'])  # 绝对坐标w映射到heatmap的坐标下, width = 22
        height = math.ceil(height * self.fmap_size['h'] / self.img_size['h'])  # 绝对坐标的height映射到heatmap, height = 27
        # 对于当前的heatmap上的obj_box, 使用长和宽计算高斯的半径
        radius = max(0, int(gaussian_radius((height, width), self.gaussian_iou)))  # 求出高斯散射核的半径, gaussian_iou = 0.3
        # hmap_tl:(80, 128, 128), 并且值都是0, 经过draw gassian之后,hmap_tl或者hmap_br有变化 , label = 41(当前的类别), hmap_tl[label]也就是当前的第41个类别的top-left的heatmap, 是128,128
        draw_gaussian(hmap_tl[label], [ixtl, iytl], radius)
        draw_gaussian(hmap_br[label], [ixbr, iybr], radius)
      else:
        hmap_tl[label, iytl, ixtl] = 1
        hmap_br[label, iybr, ixbr] = 1
      # 这个是计算下采样时候, 损失的精度.
      regs_tl[i, :] = [fxtl - ixtl, fytl - iytl]  # 这里的i是每一个物体的index, [27.85 - 27, 22.97 - 22]
      regs_br[i, :] = [fxbr - ixbr, fybr - iybr]  # [74.29 - 74, 70.1 - 70]

      # question: 22 * 128 + 27, array(128), 那么这表示每一个物体的top-left值是有一定的数学关系, 但是不知道目前这个数学公式有什么意义
      # 注意这里并不是把top-left或者bottom-right的点给映射回去, 而是用一种数学关系来描述
      inds_tl[i] = iytl * self.fmap_size['w'] + ixtl
      inds_br[i] = iybr * self.fmap_size['w'] + ixbr  # 70 * 128 + 74, array(128)

    return {'image': image,
            'hmap_tl': hmap_tl, 'hmap_br': hmap_br,
            'regs_tl': regs_tl, 'regs_br': regs_br,
            'inds_tl': inds_tl, 'inds_br': inds_br,
            'ind_masks': ind_masks}

  def __len__(self):
    return self.num_samples


class COCO_eval(COCO):
  def __init__(self, data_dir, split, test_scales=(1,), test_flip=False):
    super(COCO_eval, self).__init__(data_dir, split, gaussian=False)
    self.test_scales = test_scales
    self.test_flip = test_flip

  def __getitem__(self, index):
    img_id = self.images[index]
    image = cv2.imread(os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name']))
    height, width = image.shape[0:2]

    out = {}
    for scale in self.test_scales:
      new_height = int(height * scale)
      new_width = int(width * scale)

      in_height = new_height | 127
      in_width = new_width | 127

      fmap_height, fmap_width = (in_height + 1) // self.down_ratio, (in_width + 1) // self.down_ratio
      height_ratio = fmap_height / in_height
      width_ratio = fmap_width / in_width

      resized_image = cv2.resize(image, (new_width, new_height))
      resized_image, border, offset = crop_image(image=resized_image,
                                                 center=[new_height // 2, new_width // 2],
                                                 new_size=[in_height, in_width])

      resized_image = resized_image / 255.
      resized_image -= self.mean
      resized_image /= self.std
      resized_image = resized_image.transpose((2, 0, 1))[None, :, :, :]  # [H, W, C] to [C, H, W]

      if self.test_flip:
        resized_image = np.concatenate((resized_image, resized_image[..., ::-1].copy()), axis=0)

      out[scale] = {'image': resized_image,
                    'border': border,
                    'size': [new_height, new_width],
                    'fmap_size': [fmap_height, fmap_width],
                    'ratio': [height_ratio, width_ratio]}

    return img_id, out

  def convert_eval_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self.valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out = list(map(lambda x: float("{:.2f}".format(x)), bbox[0:4]))

          detection = {"image_id": int(image_id),
                       "category_id": int(category_id),
                       "bbox": bbox_out,
                       "score": float("{:.2f}".format(score))}

          detections.append(detection)
    return detections

  def run_eval(self, results, save_dir):
    detections = self.convert_eval_format(results)

    if save_dir is not None:
      result_json = os.path.join(save_dir, "results.json")
      json.dump(detections, open(result_json, "w"))

    coco_dets = self.coco.loadRes(detections)
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

  @staticmethod
  def collate_fn(batch):
    out = []
    for img_id, sample in batch:
      out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
      if k == 'image' else np.array(sample[s][k])[None, ...] for k in sample[s]} for s in sample}))
    return out


if __name__ == '__main__':
  # import pickle
  from tqdm import tqdm

  dataset = COCO('E:\coco_debug', 'train')
  # loader = torch.utils.data.DataLoader(dataset, batch_size=2,
  #                                            shuffle=False, num_workers=0,
  #                                            pin_memory=True, drop_last=True)
  for d in tqdm(dataset):
    pass
  # for d in tqdm(loader):
  #   pass

  dataset = COCO_eval('../data', 'val', test_flip=True, test_scales=[0.5, 0.75, 1, 1.25, 1.5])
  # for d in tqdm(dataset):
  #   pass
  loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True, drop_last=True,
                                             collate_fn=dataset.collate_fn)

  for b in tqdm(loader):
    torch.save(b, '../_debug/imgs2.t7')
    break
    pass

if __name__ == 'x__main__':
  # import pickle
  from tqdm import tqdm

  dataset = COCO('E:\coco_debug', 'train')
  data = dataset[0]
  torch.save(data, '../_debug/db.t7')

  # for d in tqdm(dataset):
  #   pass
