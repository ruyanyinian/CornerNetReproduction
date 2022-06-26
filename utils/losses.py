import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.keypoint import _tranpose_and_gather_feature


def _neg_loss(preds, targets):
  # todo targets > 1-epsilon ?  
  # target = (2, 80, 128, 128), preds=tuple((2, 80, 128, 128)), 其中preds是tuple=1的tensor,
  pos_inds = targets == 1  #等于1的那个真实值设置为true
  # todo targets < 1-epsilon ?
  # targets < 1也就是不是GT的点,这些点包括高斯函数得到的平滑的label值,这些值是大于0小于1的还有本身是0的点
  neg_inds = targets < 1  

  neg_weights = torch.pow(1 - targets[neg_inds], 4) # 不是1的点会很

  loss = 0
  # 因为preds是tuple=1的tensor, 所以这里只是变成了
  for pred in preds: # preds是 tuple(tensor), 其中tuple里面只含有一个tensor
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1-1e-4) # torch.clamp是将所以的tensor的点先知道min和max区间, (2, 80, 128, 128)
    pos_pred = pred[pos_inds] # pos_pred是torch.size(11), 也就是说pos_inds是(2, 80, 128, 128), pred也是(2, 80, 128, 128), 为什么得到的是1个11呢? 也就是说我们两张图(batchsize=2)的情况下一共有11个GT框, 我们从预测的heatmap,根据GT ind取值
    neg_pred = pred[neg_inds] # torch.Size([2621429]),

    # 我们的目的是让pos_loss足够的小, 然后让neg_loss足够的大
    # 这个对应Ldet中的第一个公式, 注意torch log是以2为底的. 所以torch.log(pos_pred)是负数, 而torch.pow(1 - pos_pred, 2)又很小, 所以整体的pos_loss是趋近于0的
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)

    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights  # 这个对应Ldet中的第二个公式

    num_pos = pos_inds.float().sum()  # 这个值就是11
    pos_loss = pos_loss.sum()  # pos的loss是11个值的相加, 在此时的场景下我们的pos_loss是-20,7981
    neg_loss = neg_loss.sum()  # 2621429个值相加, 此时我们的neg_loss是-3151

    if pos_pred.nelement() == 0: # pos_pred.nelement()求出pos_pred这个tensor有多少个值, 一共有11个值
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos  #  (0 - ((-20.79) + (-3151))) / 11= 288.38
  return loss / len(preds)


def _ae_loss(embd0s, embd1s, mask):
  num = mask.sum(dim=1, keepdim=True).float()  # [B, 1]

  pull, push = 0, 0
  for embd0, embd1 in zip(embd0s, embd1s):
    embd0 = embd0.squeeze()  # [B, num_obj]
    embd1 = embd1.squeeze()  # [B, num_obj]

    embd_mean = (embd0 + embd1) / 2

    embd0 = torch.pow(embd0 - embd_mean, 2) / (num + 1e-4)
    embd0 = embd0[mask].sum()
    embd1 = torch.pow(embd1 - embd_mean, 2) / (num + 1e-4)
    embd1 = embd1[mask].sum()
    pull += embd0 + embd1

    push_mask = (mask[:, None, :] + mask[:, :, None]) == 2  # [B, num_obj, num_obj]
    dist = F.relu(1 - (embd_mean[:, None, :] - embd_mean[:, :, None]).abs(), inplace=True)
    dist = dist - 1 / (num[:, :, None] + 1e-4)  # substract diagonal elements
    dist = dist / ((num - 1) * num + 1e-4)[:, :, None]  # total num element is n*n-n
    push += dist[push_mask].sum()
  return pull / len(embd0s), push / len(embd0s)


def _reg_loss(regs, gt_regs, mask):
  # regs: 预测的regs_tl或者regs_br, 他们的shape都是[2, 2, 128, 128]
  # gt_regs = (128, 2)
  # mask = (128, 2)
  num = mask.float().sum() + 1e-4
  mask = mask[:, :, None].expand_as(gt_regs)  # [B, num_obj, 2], (2, 128, 2)
  loss = sum([F.smooth_l1_loss(r[mask], gt_regs[mask], reduction='sum') / num for r in regs])
  return loss / len(regs)


class Loss(nn.Module):
  def __init__(self, model):
    super(Loss, self).__init__()
    self.model = model

  def forward(self, batch):
    outputs = self.model(batch['image'])
    hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br = zip(*outputs)

    embd_tl = [_tranpose_and_gather_feature(e, batch['inds_tl']) for e in embd_tl]
    embd_br = [_tranpose_and_gather_feature(e, batch['inds_br']) for e in embd_br]
    regs_tl = [_tranpose_and_gather_feature(r, batch['inds_tl']) for r in regs_tl]
    regs_br = [_tranpose_and_gather_feature(r, batch['inds_br']) for r in regs_br]

    focal_loss = _neg_loss(hmap_tl, batch['hmap_tl']) + \
                 _neg_loss(hmap_br, batch['hmap_br'])
    reg_loss = _reg_loss(regs_tl, batch['regs_tl'], batch['ind_masks']) + \
               _reg_loss(regs_br, batch['regs_br'], batch['ind_masks'])
    pull_loss, push_loss = _ae_loss(embd_tl, embd_br, batch['ind_masks'])

    loss = focal_loss + 0.1 * pull_loss + 0.1 * push_loss + reg_loss
    return loss.unsqueeze(0), outputs
