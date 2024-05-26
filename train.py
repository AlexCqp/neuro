import os
import sys
import utils
import torch
from barbar import Bar
from torch.utils.tensorboard import SummaryWriter
from validation import validate
from utils import get_bboxes
def save_checkpoint(model, model_opt, save_weights_path,experiment_name, epoch, save_type):
    checkpoint = {
    'model_dict': model.state_dict(),
    'optimizer_dict': model_opt.state_dict(),
    }
    if save_type == 'best':
        torch.save(checkpoint, os.path.join(save_weights_path,f'model_best_{experiment_name}.pth'))
    elif save_type == 'last':
        torch.save(checkpoint, os.path.join(save_weights_path,f'model_last_{experiment_name}.pth'))
    elif save_type == "best_acc":
        torch.save(checkpoint, os.path.join(save_weights_path,f'model_best_acc_{experiment_name}.pth'))
    else:
        torch.save(checkpoint, os.path.join(save_weights_path,f'model_{experiment_name}_{epoch}.pth'))
    del checkpoint
def validation_run(model, valid_loader, test_valid_loader,test_loader,tb, epoch, experiment_name,save_result_path):
    print(utils.BColors.FAIL + '\tValidation...' +utils.BColors.ENDC)
    accuracy = validate(model=model,valid_loader=valid_loader,test_valid_loader=test_valid_loader,test_loader=test_loader,tb=tb,epoch=epoch)
    print(utils.BColors.FAIL + '\tSaving image...' +utils.BColors.ENDC)
    name = f'{experiment_name}_{epoch}.png'
    get_bboxes(model=model, batch=next(iter(valid_loader)),name=name, path=save_result_path)
    print(utils.BColors.FAIL + '\tComplete' +utils.BColors.ENDC)
    return accuracy
def train(model,model_opt,scaler,train_loader,start,epochs,enable_tb,experiment_name,weights_dir,results_dir,tb_dir,save_every,valid_loader,test_valid_loader,test_loader,lr_schedule):
    if enable_tb:
        tb = SummaryWriter(os.path.join(tb_dir,experiment_name))
    else:
        tb = None
    save_weights_path = os.path.join(weights_dir,experiment_name)
    if not os.path.exists(save_weights_path):
        os.mkdir(save_weights_path)
    save_results_path = os.path.join(results_dir,experiment_name)
    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)
    model.train()
    total_losses_min = sys.maxsize
    global_accuracy = 0.
    for epoch in range(start, epochs):
        loc_loss_total = 0
        bias_loss_total = 0
        total_loss_total = 0
        current_accuracy = 0.
        print(utils.BColors.OKBLUE + '\nEpoch: [{}/{}]'.format(epoch + 1, epochs) + utils.BColors.ENDC)
        print(f'Current the best accuracy: {global_accuracy}\n')
        for i, batch in enumerate(Bar(train_loader)):
            model_opt.zero_grad()
            it = len(train_loader) * epoch + i
            for j, param_group in enumerate(model_opt.param_groups):
                param_group["lr"] = lr_schedule[it]
            with torch.cuda.amp.autocast():
                im1, im2, fim = batch[1]
                dx, dy = batch[2]
                # not used here... for now.. maybe
                # cls_targets = batch[3].cuda()
                im1 = im1.cuda()
                im2 = im2.cuda()
                fim = fim.cuda()
                dx = dx.float().cuda() / 224
                dy = dy.float().cuda() / 224
                p1_loc = model(im1)
                p2_loc = model(im2)
                loc_loss = ((p1_loc[:, 0] - p2_loc[:, 0] - dx).pow(2).mean() +(p1_loc[:, 1] - p2_loc[:, 1] -dy).pow(2).mean())
                pf_loc = model(fim)
                bias_loss = ((p1_loc[:, 0] + pf_loc[:, 1] - 1).pow(2).mean() + (p1_loc[:, 1] + pf_loc[:, 0] -1).pow(2).mean())
                total_loss = loc_loss + bias_loss
            scaler.scale(total_loss).backward()
            scaler.step(model_opt)
            scaler.update()
            loc_loss_total += loc_loss
            bias_loss_total += bias_loss
            total_loss_total += total_loss
        loc_loss_total /= len(train_loader)
        bias_loss_total /= len(train_loader)
        total_loss_total /= len(train_loader)
        if enable_tb:
            tb.add_scalar(f'Loss/Train_loc_loss',loc_loss_total, epoch)
            tb.add_scalar(f'Loss/Train_bias_loss',bias_loss_total, epoch)
            tb.add_scalar(f'Loss/Train_total_loss',total_loss_total, epoch)
        print(utils.BColors.OKBLUE +'\tLoc loss: {:.6f}'.format(loc_loss_total) +utils.BColors.ENDC)
        print(utils.BColors.OKBLUE +'\tBias loss: {:.6f}'.format(bias_loss_total) +utils.BColors.ENDC)
        print(utils.BColors.OKBLUE +'\tTotal loss: {:.6f}'.format(total_loss_total) +utils.BColors.ENDC)
        current_accuracy = validation_run(model, valid_loader,test_valid_loader, test_loader,tb, epoch,experiment_name, save_results_path)
        if current_accuracy > global_accuracy:
            global_accuracy = current_accuracy
            save_checkpoint(model, model_opt, save_weights_path,experiment_name, epoch, 'best_acc')
