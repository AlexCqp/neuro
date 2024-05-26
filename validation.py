import torch
import numpy as np
def validate(model,valid_loader,test_valid_loader,test_loader,tb=None,epoch=None):
    model.eval()
    min_dist = []
    val_loc_losses = []
    cen_bias_list = []
    with torch.no_grad():
        for batch in valid_loader:
            im1, im2, _ = batch[1]
            dx, dy = batch[2]
            im1 = im1.cuda()
            im2 = im2.cuda()
            dx = dx.float().cuda() / 224
            dy = dy.float().cuda() / 224
            p1_loc = model(im1)
            p2_loc = model(im2)
            dist = ((p1_loc[:, 0] - p2_loc[:, 0] - dx).pow(2) + (p1_loc[:, 1] - p2_loc[:, 1] - dy).pow(2))
            loc_loss = dist.mean()
            isorted = dist.argsort()
            for i in isorted:
                if (batch[-1][0][i][0] == batch[-1][0][i][2]) and (batch[-1][0][i][1] == batch[-1][0][i][3]):
                    continue
                else:
                    min_dist.append(dist[i].detach().cpu().numpy())
                    min_dist_bb = (batch[-1][0][i], batch[-1][1][i])
                    min_dist_img = (im1[i], im2[i])
                    # Calculation a bias for the current batch
                    pcen = model(torch.stack(min_dist_img))
                    target_cen =torch.tensor([[(min_dist_bb[0][0] + min_dist_bb[0][2]).float() /448,(min_dist_bb[0][1] + min_dist_bb[0][3]).float() / 448],[(min_dist_bb[1][0] + min_dist_bb[1][2]).float() / 448,(min_dist_bb[1][1] + min_dist_bb[1][3]).float() / 448]])
                    cen_bias = (target_cen.cuda() - pcen).detach().mean(0)

    cen_bias_list.append(cen_bias.detach().cpu().numpy())
    val_loc_losses.append(loc_loss.cpu().detach().item())
    cen_bias =torch.from_numpy(np.asarray(cen_bias_list).mean(0)).cuda()
    min_dist = np.min(min_dist)
    del cen_bias_list
    # For average valid center baseline
    total_valid_imgs = 0
    vx = 0
    vy = 0
    with torch.no_grad():
        for batch in test_valid_loader:
            valid_idxes = (batch[-1][:, :2] != batch[-1][:,2:]).all(1).nonzero().squeeze().tolist()
            if isinstance(valid_idxes, int):
                valid_idxes = [valid_idxes]
            for i in valid_idxes:
                bbox_real = batch[-1][i]
                total_valid_imgs += 1
                vx += (bbox_real[2] + bbox_real[0]) // 2
                vy += (bbox_real[3] + bbox_real[1]) // 2
    vx = vx.float()
    vy = vy.float()
    vx /= total_valid_imgs
    vy /= total_valid_imgs
    # Get validation and 2 baseline results
    total_imgs = 0
    matched_imgs = 0
    matched_baseline1 = 0
    matched_baseline2 = 0
    with torch.no_grad():
        for batch in test_loader:
            im1 = batch[1]
            im1 = im1.cuda()
            p1_loc = model(im1)
            valid_idxes = (batch[-1][:, :2] != batch[-1][:,2:]).all(1).nonzero().view(-1).tolist()
        if isinstance(valid_idxes, int):
            valid_idxes = [valid_idxes]
        for i in valid_idxes:
            total_imgs += 1
            bbox_real = batch[-1][i]
            pred = ((p1_loc[i] + cen_bias) *224).int().cpu().numpy()
            bb_h = bbox_real[3] - bbox_real[1]
            bb_w = bbox_real[2] - bbox_real[0]
            pred_bb = np.array([pred[0].item() - bb_w // 2,pred[1].item() - bb_h // 2,pred[0].item() + bb_w // 2,pred[1].item() + bb_h // 2])
            in_bbox = (max(pred_bb[0], bbox_real[0]),max(pred_bb[1], bbox_real[1]),min(pred_bb[2], bbox_real[2]),min(pred_bb[3], bbox_real[3]))
            in_square = max(0, (in_bbox[2] - in_bbox[0] +1)) * max(0, (in_bbox[3] - in_bbox[1] + 1))
            if in_square == 0:
                iou = 0
            else:
            # bugfix of rare attribute error
                if str(type(in_square)) == '<class \'torch.Tensor\'>':
                    in_square = in_square.float()
                else:
                    in_square = in_square.astype(np.float)
                iou = in_square / ((bbox_real[2] - bbox_real[0] + 1) * (bbox_real[3] - bbox_real[1] + 1) + (pred_bb[2] - pred_bb[0] + 1) * (pred_bb[3] - pred_bb[1] + 1) - in_square)
            if iou >= 0.5:
                matched_imgs += 1
            # baseline 1
            baseline1_bb = np.array([112 - bb_w // 2, 112 - bb_h // 2, 112 + bb_w // 2, 112 + bb_h // 2])
            in_bbox1 = (max(baseline1_bb[0], bbox_real[0]),max(baseline1_bb[1], bbox_real[1]),min(baseline1_bb[2], bbox_real[2]),min(baseline1_bb[3], bbox_real[3]))
            in_square1 = max(0, (in_bbox1[2] - in_bbox1[0] + 1)) * max(0, (in_bbox1[3] - in_bbox1[1] + 1))
            if in_square1 == 0:
                iou1 = 0
            else:
            # bugfix of rare attribute error
                if str(type(in_square1)) == '<class \'torch.Tensor\'>':
                    in_square1 = in_square1.float()
                else:
                    in_square1 = in_square1.astype(np.float)
                iou1 = in_square1 / ((bbox_real[2] - bbox_real[0] + 1) * (bbox_real[3] - bbox_real[1] + 1) + (baseline1_bb[2] - baseline1_bb[0] + 1) * (baseline1_bb[3] - baseline1_bb[1] + 1) - in_square1)
            if iou1 >= 0.5:
                matched_baseline1 += 1
            # baseline 2
            baseline2_bb = np.array([vx - bb_w // 2, vy -bb_h // 2,vx + bb_w // 2, vy +bb_h // 2])
            in_bbox2 = (max(baseline2_bb[0], bbox_real[0]),max(baseline2_bb[1], bbox_real[1]),min(baseline2_bb[2], bbox_real[2]),min(baseline2_bb[3], bbox_real[3]))
            in_square2 = max(0, (in_bbox2[2] - in_bbox2[0] + 1)) * max(0, (in_bbox2[3] - in_bbox2[1] + 1))
            if in_square2 == 0:
                iou2 = 0
            else:
            # bugfix of rare attribute error
                if str(type(in_square2)) == '<class \'torch.Tensor\'>':
                    in_square2 = in_square2.float()
                else:
                    in_square2 = in_square2.astype(np.float)
                iou2 = in_square2 / ((bbox_real[2] -bbox_real[0] + 1) * (bbox_real[3] - bbox_real[1] + 1)+(baseline2_bb[2] - baseline2_bb[0] + 1)*(baseline2_bb[3] - baseline2_bb[1] + 1) - in_square2)
            if iou2 >= 0.5:
                matched_baseline2 += 1
    print('\tValidation loc loss: ' + str(np.mean(val_loc_losses)))
    print('\tmin dist: ' + str(min_dist))
    # print('\tpcen: ' + str(pcen.detach().cpu().numpy()))
    print('\tcen_bias: ' + str(cen_bias.detach().cpu().numpy()) + '\n')
    print(f'\tResult: {matched_imgs / total_imgs}')
    print(f'\tCenter baseline: {matched_baseline1 / total_imgs}')
    print(f'\tAvg valid center baseline: {matched_baseline2 / total_imgs}')
    if tb is not None:
        tb.add_scalar(f'Validation/Validation_loc_loss',np.mean(val_loc_losses), epoch)
        tb.add_scalar(f'Validation/min_dist', min_dist, epoch)
        tb.add_scalar(f'Validation/Result', matched_imgs / total_imgs, epoch)
        tb.add_histogram('Validation/pcen', pcen, epoch,bins="auto")
        tb.add_histogram('Validation/cen_bias', cen_bias, epoch, bins="auto")
    return matched_imgs / total_imgs
