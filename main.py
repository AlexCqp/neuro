from svertok import ResNet18
from train import train
from validation import validate
from utils import get_bboxes, cosine_scheduler, get_world_size
from config import parse_opts
import dataset
import torch
import coord
from torchvision import transforms
from torch.utils.data import dataloader
from models import VisionTransformer
import random
import numpy as np
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_epochs * niter_per_ep)
    iters = np.arange(epochs * niter_per_ep - len(warmup_schedule))
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
if __name__ == '__main__':
    opt = parse_opts()
    if opt.dataset == 'cats':
        train_dataset = dataset.CatsDataset(opt.dataset_path,dataset_type='train')
        train_loader = dataloader.DataLoader(train_dataset,batch_size=opt.batch_size, shuffle=True,collate_fn=lambda p: coord.custom_collate_fn(p, color_transforms,p_flip=0.5, drop_channels=opt.drop_channels, grid=grid))
        valid_dataset = dataset.CatsDataset(opt.dataset_path,dataset_type='valid')
        valid_loader = dataloader.DataLoader(valid_dataset,batch_size=opt.batch_size, shuffle=True,collate_fn=lambda p: coord.custom_collate_fn(p,color_transforms, 0.0))
        test_valid_loader = dataloader.DataLoader(valid_dataset,batch_size=opt.batch_size, shuffle=False, collate_fn=coord.test_custom_collate_fn)
        test_dataset = dataset.CatsDataset(opt.dataset_path,dataset_type='test')
        test_loader = dataloader.DataLoader(test_dataset,batch_size=8, shuffle=True,collate_fn=coord.test_custom_collate_fn)
    elif opt.dataset == 'cub200':
        train_dataset = dataset.CUB200(opt.dataset_path,dataset_type='train')
        train_loader = dataloader.DataLoader(train_dataset,batch_size=opt.batch_size, shuffle=True,collate_fn=lambda p: coord.custom_collate_fn(p, color_transforms, p_flip=0.5,drop_channels=opt.drop_channels,grid=grid))
        valid_dataset = dataset.CUB200(opt.dataset_path,dataset_type='valid')
        valid_loader = dataloader.DataLoader(valid_dataset,batch_size=opt.batch_size, shuffle=True,collate_fn=lambda p: coord.custom_collate_fn(p,color_transforms, 0.0, crops_dict=None))
        test_valid_loader = dataloader.DataLoader(valid_dataset,batch_size=opt.batch_size, shuffle=False,collate_fn=coord.test_custom_collate_fn)
        test_dataset = dataset.CUB200(opt.dataset_path,dataset_type='test')
        test_loader = dataloader.DataLoader(test_dataset,batch_size=8, shuffle=True,collate_fn=coord.test_custom_collate_fn)
    elif opt.dataset == 'rectangles':
        train_dataset = dataset.BlackRectangles(opt.dataset_path, dataset_type='train')
        train_loader = dataloader.DataLoader(train_dataset,batch_size=opt.batch_size, shuffle=True,collate_fn=lambda p: coord.custom_collate_fn(p, color_transforms,p_flip=0.5,drop_channels=opt.drop_channels,grid=grid))
        valid_dataset =dataset.BlackRectangles(opt.dataset_path, dataset_type='valid')
        valid_loader = dataloader.DataLoader(valid_dataset,batch_size=opt.batch_size, shuffle=True,collate_fn=lambda p: coord.custom_collate_fn(p, color_transforms,0.0))
        test_valid_loader = dataloader.DataLoader(valid_dataset,batch_size=opt.batch_size, shuffle=False,collate_fn=coord.test_custom_collate_fn)
        test_dataset = dataset.BlackRectangles(opt.dataset_path,dataset_type='test')
        test_loader = dataloader.DataLoader(test_dataset,batch_size=8, shuffle=True,collate_fn=coord.test_custom_collate_fn)
    # GridMask
    if opt.use_grid:
        grid = dataset.GridMask(prob=1)
    else:
        grid = None
    color_transforms = transforms.ColorJitter(0.3, 0.15, 0.1)
    # color_transforms = transforms.ColorJitter(0, 0, 0)
    # Model and optimizer definition
    if opt.arch == 'vit':
        model = VisionTransformer(img_size=224,patch_size=16,n_classes=1000,embed_dim=768,depth=12,n_heads=12,mlp_ratio=4.,qkv_bias=True,p=0.1,attn_p=0.1)
        vit_pretrained = torch.load('weights/vit_base_p16_224.pth')
        model.load_state_dict(vit_pretrained)
        model.head = torch.nn.Sequential(torch.nn.Linear(768,512),torch.nn.ReLU(),torch.nn.Linear(512,2))
    else:
        model = ResNet18(2)
    model.cuda()
    print(model)
    if opt.arch == 'vit':
        model_opt = torch.optim.Adam(model.parameters())
    else:
        model_opt = torch.optim.AdamW(model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    # ============ init scheduler ============
    lr_schedule = cosine_scheduler(
    opt.lr * (opt.batch_size * get_world_size()) / 16, # linear scaling rule
    1e-6,
    opt.epochs, len(train_loader),
    warmup_epochs=10,
    )
    if opt.model_path != '':
        checkpoint = torch.load(opt.model_path)
        model.load_state_dict(checkpoint['model_dict']) #model_dict/state_dict
        model_opt.load_state_dict(checkpoint['optimizer_dict'])
        del checkpoint
    if opt.mode == 'train':
        max_lr = 0.01  # Максимальная скорость обучения
        min_lr = 0.0001  # Минимальная скорость обучения
        epochs = 100  # Количество эпох обучения
        niter_per_ep = len(train_loader)  # Количество итераций на эпоху
        warmup_epochs = 10  # Количество эпох для разогрева

        lr_schedule = cosine_scheduler(
            base_value=max_lr,
            final_value=min_lr,
            epochs=epochs,
            niter_per_ep=niter_per_ep,
            warmup_epochs=warmup_epochs
        )
        train(model=model,model_opt=model_opt,scaler=scaler,train_loader=train_loader,start=opt.continue_training,epochs=opt.epochs,enable_tb=opt.enable_tb,experiment_name=opt.experiment_name,results_dir=opt.results_dir,weights_dir=opt.weights_dir,tb_dir=opt.tb_dir,save_every=opt.save_every,valid_loader=valid_loader,test_valid_loader=test_valid_loader,test_loader=test_loader, lr_schedule=lr_schedule)
    if opt.mode == 'validation':
        accuracy = validate(model=model,valid_loader=valid_loader,test_valid_loader=test_valid_loader,test_loader=test_loader)
    # get bbox images
    get_bboxes(model=model, batch=next(iter(valid_loader)),name=opt.experiment_name)
