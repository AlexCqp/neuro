import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
def show_bboxes(model, im, bbox_real, draw_only_point=True):
    with torch.no_grad():
        pred = (model(im.cuda().unsqueeze(0))[0] * 224).int().cpu().numpy()
    im = im.cpu().numpy().transpose(1, 2, 0)
    bb_h = bbox_real[3] - bbox_real[1]
    bb_w = bbox_real[2] - bbox_real[0]
    bbox_real[bbox_real < 0] = 0
    bbox_real[bbox_real > 223] = 223
    green_p = np.array([0.0, 1.0, 0.0])
    im[bbox_real[1], bbox_real[0]:bbox_real[2] + 1, :] = green_p
    im[bbox_real[3], bbox_real[0]:bbox_real[2] + 1, :] = green_p
    im[bbox_real[1]:bbox_real[3] + 1, bbox_real[0], :] = green_p
    im[bbox_real[1]:bbox_real[3] + 1, bbox_real[2], :] = green_p
    if draw_only_point:
        pred_bb = np.array([pred[0].item() - 15, pred[1].item() - 15,
        pred[0].item() + 15, pred[1].item() + 15])
        pred_bb[pred_bb < 0] = 0
        pred_bb[pred_bb > 223] = 223
        red_p = np.array([1.0, 0.0, 0.0])
        im[pred_bb[1]:pred_bb[3] + 1, pred_bb[0]:pred_bb[2] + 1,:] = red_p
    else:
        pred_bb = np.array([pred[0].item() - bb_w // 2,
        pred[1].item() - bb_h // 2,
        pred[0].item() + bb_w // 2,
        pred[1].item() + bb_h // 2])
        pred_bb[pred_bb < 0] = 0
        pred_bb[pred_bb > 223] = 223
        red_p = np.array([1.0, 0.0, 0.0])
        im[pred_bb[1], pred_bb[0]:pred_bb[2] + 1, :] = red_p
        im[pred_bb[3], pred_bb[0]:pred_bb[2] + 1, :] = red_p
        im[pred_bb[1]:pred_bb[3] + 1, pred_bb[0], :] = red_p
        im[pred_bb[1]:pred_bb[3] + 1, pred_bb[2], :] = red_p
    return im
def get_bboxes(model, batch, name, path=None):
    im1, im2, _ = batch[1]
    valid_idxes = (batch[-1][0][:, :2] != batch[-1][0][:,2:]).all(1).nonzero().squeeze().tolist()
    n_rows = len(valid_idxes)
    n_cols = 2
    fig = plt.figure(figsize=(n_cols * 10, n_rows * 10))
    for j, i in enumerate(valid_idxes):
        ax = fig.add_subplot(n_rows, n_cols, 2 * j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(show_bboxes(model, im1[i].clone(), batch[-1][0][i]))
        ax = fig.add_subplot(n_rows, n_cols, 2 * j + 2)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(show_bboxes(model, im2[i].clone(), batch[-1][1][i]))
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,hspace=0.1, wspace=0.1)
    if path is not None:
        img_path = os.path.join(path, name)
    else:
        img_path = f'./{name}.png'
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
    # plt.show()
def cosine_scheduler(base_value, final_value, epochs,niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value,base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
    
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()