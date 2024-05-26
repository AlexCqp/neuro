import torch
import numpy as np
import random
from torchvision import transforms
import GridMask as gm

def oracle(w, h, patch_size, ref_bb, valid_range=0.8):
    # Примерная реализация функции oracle
    # Подразумевает, что координаты кропа выбираются в пределах допустимого диапазона valid_range

    max_offset_w = int((w - patch_size) * valid_range)
    max_offset_h = int((h - patch_size) * valid_range)

    x00 = random.randint(0, max_offset_w)
    y00 = random.randint(0, max_offset_h)

    x01 = random.randint(0, max_offset_w)
    y01 = random.randint(0, max_offset_h)

    return x00, x01, y00, y01
def custom_collate_fn(samples, color_transforms, p_flip=0.5, drop_channels=False, grid=None, use_oracle=False):
    bs = len(samples)
    patch_size = 224	# input image size 224x224

    classes = torch.tensor([s[2] for s in samples], dtype=torch.long) - 1

    names = [s[0] for s in samples]
    coordinates1 = torch.zeros(bs, 4, dtype=torch.long)
    coordinates2 = torch.zeros(bs, 4, dtype=torch.long)

    dx = torch.zeros(bs, dtype=torch.float)
    dy = torch.zeros(bs, dtype=torch.float)

    tim1 = []
    tim2 = []
    fim = []

    for i in range(bs):
        ref_img = samples[i][1]
        ref_bb = samples[i][3]
        w, h = ref_img.size

        ref_img = color_transforms(ref_img)

        # in case of using
        if torch.rand(1) < p_flip:
            ref_img = transforms.functional.hflip(ref_img)
        if not ref_bb[0] is None and not ref_bb[2] is None:
            ref_bb = (w - ref_bb[2], ref_bb[1], w - ref_bb[0], ref_bb[3])
        if torch.rand(1) < p_flip:

            ref_img = transforms.functional.vflip(ref_img)
        if not ref_bb[1] is None and not ref_bb[3] is None:
            ref_bb = (ref_bb[0], h - ref_bb[3], ref_bb[2], h - ref_bb[1])

        if use_oracle:
            x00, x01, y00, y01 = oracle(w, h, patch_size, ref_bb, valid_range=0.8)
        else:
            x00 = random.randint(0, w - patch_size)
            x01 = random.randint(0, w - patch_size)

            y00 = random.randint(0, h - patch_size)
            y01 = random.randint(0, h - patch_size)

        img0 = ref_img.crop((x00, y00, x00 + patch_size, y00 + patch_size))
        img1 = ref_img.crop((x01, y01, x01 + patch_size, y01 + patch_size))
        img0 = np.asarray(img0, dtype=np.float32)
        img1 = np.asarray(img1, dtype=np.float32)
        if img0.shape[2] == 4:
            img0 = img0[:, :, :3]
            img1 = img1[:, :, :3]

        coordinates1[i][0] = (ref_bb[0] - x00) if not ref_bb[0] is None else -1
        coordinates1[i][1] = (ref_bb[1] - y00) if not ref_bb[1] is None else -1
        coordinates1[i][2] = (ref_bb[2] - x00) if not ref_bb[2] is None else -1
        coordinates1[i][3] = (ref_bb[3] - y00) if not ref_bb[3] is None else -1
        coordinates2[i][0] = (ref_bb[0] - x01) if not ref_bb[0] is None else -1
        coordinates2[i][1] = (ref_bb[1] - y01) if not ref_bb[1] is None else -1
        coordinates2[i][2] = (ref_bb[2] - x01) if not ref_bb[2] is None else -1
        coordinates2[i][3] = (ref_bb[3] - y01) if not ref_bb[3] is None else -1

        dx[i] = x01 - x00
        dy[i] = y01 - y00

        if drop_channels:

            # Call chromatic aberration removal (random channels drop)

            img0, img1 = gm.GridMask.removal_chromatic(img0), gm.GridMask.removal_chromatic(img1)

        if grid is not None:
            img0, img1 = grid(img0, img1)

        # new
        tim1.append(torch.from_numpy(img0.transpose(2, 0, 1) /

        255))
        tim2.append(torch.from_numpy(img1.transpose(2, 0, 1) /
        255))


    fim.append(torch.from_numpy(np.fliplr(np.flipud(img0)).copy().transpose(2, 0, 1) / 255))
    return names, (torch.stack(tim1), torch.stack(tim2), torch.stack(fim)), (dx, dy), classes, (coordinates1, coordinates2)


    #	using center crop instead of random crop 
def test_custom_collate_fn(samples):

    bs = len(samples) 
    patch_size = 224
    cat_classes = torch.tensor([s[2] for s in samples], dtype=torch.long) - 1
    names = [s[0] for s in samples]
    coordinates = torch.zeros(bs, 4, dtype=torch.long)
    tim = []
    for i in range(bs):
        ref_img = samples[i][1]
        ref_bb = samples[i][3]

        w, h = ref_img.size

        x_start = int(w / 2 - patch_size / 2)

        y_start = int(h / 2 - patch_size / 2)

        img = ref_img.crop((x_start, y_start, x_start + patch_size, y_start + patch_size))

        coordinates[i][0] = (ref_bb[0] - x_start) if not ref_bb[0] is None else -1

        coordinates[i][1] = (ref_bb[1] - y_start) if not ref_bb[1] is None else -1

        coordinates[i][2] = (ref_bb[2] - x_start) if not ref_bb[2] is None else -1

        coordinates[i][3] = (ref_bb[3] - y_start) if not ref_bb[3] is None else -1

        tim.append(torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255)[:3, :, :])

    return names, torch.stack(tim), cat_classes, coordinates
