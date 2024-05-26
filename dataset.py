import os
import matplotlib.pyplot as plt
import xmltodict
import pickle
import re
import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2

class CatsDataset(Dataset):
    def __init__(self, dataset_path, dataset_type='train',
        min_dim=300):
        super().__init__()
        img_path = os.path.join(dataset_path, 'images')
        annotations_path = os.path.join(dataset_path,'annotations') \
        # xmls_path = os.path.join(annotations_path, 'xmls')
        self.imgs = []
        self.names = []
        self.classes = []
        self.coordinates = []
        annotation_pattern = re.compile(r'([A-z0-9]+_\d+) (\d+)')
        with open(os.path.join(annotations_path, 'list.txt'),'r') as list_file:
            lines = list_file.readlines()[6:]
        if dataset_type not in ('train', 'valid', 'test'):
            raise Exception(f'Unknown dataset type:"{dataset_type}"')
        else:
            with open(os.path.join(dataset_path,f'{dataset_type}_idx.pickle'), 'rb') as idx_file:
                idxes = pickle.load(idx_file)
        filted_lines = [lines[i] for i in idxes]
        for line in filted_lines:
            name, class_id = annotation_pattern.match(line.strip("\n")).groups()
            self.names.append(name)
            img = Image.open(os.path.join(img_path,f'{name}.jpg'))
            w, h = img.size
            if w < h:
                scale_ratio = min_dim / w
                img = img.resize((min_dim, int(h *scale_ratio)))
            else:
                scale_ratio = min_dim / h
                img = img.resize((int(w * scale_ratio),min_dim))
                self.imgs.append(img)
                self.classes.append(int(class_id))
            try:
                with open(os.path.join(annotations_path, 'xmls',f'{name}.xml'), 'r') as file:
                    raw_xml = file.readlines()
                xml_dict = xmltodict.parse(raw_xml[0])
                if type(xml_dict['annotation']['object']) is list:
                    object_dict = xml_dict['annotation']['object'][0]
                else:
                    object_dict = xml_dict['annotation']['object']
                self.coordinates.append((int(int(object_dict['bndbox']['xmin']) * scale_ratio),
                int(int(object_dict['bndbox']['ymin']) * scale_ratio),int(int(object_dict['bndbox']['xmax']) * scale_ratio),

                int(int(object_dict['bndbox']['ymax']) * scale_ratio)))
            except FileNotFoundError as err:
                self.coordinates.append((None, None, None,None))
            except Exception as ex:
                print(xml_dict['annotation']['object'])
                print(type(xml_dict['annotation']['object']) is list)
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        name = self.names[idx]
        ref_img = self.imgs[idx]
        cat_cls = self.classes[idx]
        coord = self.coordinates[idx]
        return name, ref_img, cat_cls, coord
class CUB200(Dataset):
    def __init__(self, dataset_path, dataset_type='train',min_dim=300):
        super().__init__()
        with open(os.path.join(dataset_path,'cub200_dataset.pickle'), 'rb') as f:
            data = pickle.load(f)
        images_dir = os.path.join(dataset_path, 'images')
        if dataset_type == 'train':
            data = data["train"]
        elif dataset_type == 'valid':
            data = data["valid"]
        else:
            data = data["test"]
        self.imgs = []
        self.names = []
        self.classes = []
        self.coordinates = []
        for data_object in data:
            img_path = data_object[0]
            img_class = data_object[1]
            xywh_bbox = data_object[2]
            img = Image.open(os.path.join(images_dir, img_path))
            w, h = img.size
            if w < h:
                scale_ratio = min_dim / w
                img = img.resize((min_dim, int(h * scale_ratio)))
            else:
                scale_ratio = min_dim / h
                img = img.resize((int(w * scale_ratio), min_dim))
                # to [xmin, ymin, xmax, ymax]
                img_bbox = [xywh_bbox[0], xywh_bbox[1], xywh_bbox[0] + xywh_bbox[2], xywh_bbox[1] + xywh_bbox[3]]
            if len(np.asarray(img).shape) < 3:
                continue
            else:
                self.imgs.append(img)
                self.names.append(img_path.split('/')[-1])
                self.classes.append(img_class)
                self.coordinates.append((int(img_bbox[0] *scale_ratio),
                int(img_bbox[1] *scale_ratio),
                int(img_bbox[2] *scale_ratio),
                int(img_bbox[3] *scale_ratio)))
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        name = self.names[idx]
        ref_img = self.imgs[idx]
        cls = self.classes[idx]
        coord = self.coordinates[idx]
        return name, ref_img, cls, coord