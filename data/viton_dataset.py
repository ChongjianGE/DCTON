import os
import random
import json
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import ipdb


class VITONDataset(BaseDataset):
    def __init__(self, opt):
        super(VITONDataset, self).__init__(opt)
        self.dataroot = opt.dataroot
        self.fine_height = opt.load_height
        self.fine_width = opt.load_width
        self.radius = 5
        self.params = {'flip': 0}
        self.opt = opt
        random.seed(2048)

        if opt.isTrain:
            suffix_img = '_img'
            self.dir_img = os.path.join(self.dataroot, opt.phase + suffix_img)
            self.img_paths = sorted(make_dataset(self.dir_img, opt.max_dataset_size))

            suffix_label = '_label'
            self.dir_label = os.path.join(self.dataroot, opt.phase + suffix_label)
            self.label_paths = sorted(make_dataset(self.dir_label, opt.max_dataset_size))

            suffix_edge = '_edge'
            self.dir_edge = os.path.join(self.dataroot, opt.phase + suffix_edge)
            self.edge_paths = sorted(make_dataset(self.dir_edge, opt.max_dataset_size))

            suffix_clothes = '_color'
            self.dir_clothes = os.path.join(self.dataroot, opt.phase + suffix_clothes)
            self.clothes_paths = sorted(make_dataset(self.dir_clothes, opt.max_dataset_size))

            suffix_densepose = '_densepose'
            self.dir_densepose = os.path.join(self.dataroot, opt.phase + suffix_densepose)
            self.densepose_paths = sorted(make_dataset(self.dir_densepose, opt.max_dataset_size))

            self.img_size = len(self.img_paths)
            self.transform_rgb = get_transform(self.opt, self.params, grayscale=False)
            self.transform_l = get_transform(self.opt, self.params, grayscale=True, convert=False)
        else:
            suffix_img = '_img'
            self.dir_img = os.path.join(self.dataroot, opt.phase + suffix_img)
            self.img_paths = sorted(make_dataset(self.dir_img, opt.max_dataset_size))

            suffix_label = '_label'
            self.dir_label = os.path.join(self.dataroot, opt.phase + suffix_label)
            self.label_paths = sorted(make_dataset(self.dir_label, opt.max_dataset_size))

            suffix_edge = '_edge'
            self.dir_edge = os.path.join(self.dataroot, opt.phase + suffix_edge)
            self.edge_paths = sorted(make_dataset(self.dir_edge, opt.max_dataset_size))

            suffix_clothes = '_color'
            self.dir_clothes = os.path.join(self.dataroot, opt.phase + suffix_clothes)
            self.clothes_paths = sorted(make_dataset(self.dir_clothes, opt.max_dataset_size))

            suffix_densepose = '_densepose'
            self.dir_densepose = os.path.join(self.dataroot, opt.phase + suffix_densepose)
            self.densepose_paths = sorted(make_dataset(self.dir_densepose, opt.max_dataset_size))

            self.img_size = len(self.img_paths)
            self.transform_rgb = get_transform(self.opt, self.params, grayscale=False)
            self.transform_l = get_transform(self.opt, self.params, grayscale=True, convert=False)

        if opt.isTrain:
            self.select = False
        else:
            self.select = True

        if self.select:
            img_paths = []
            label_paths = []
            edge_paths = []
            clothes_paths = []
            densepose_paths = []
            names = []
            with open('./demo_data/test.txt', 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    # c_name = im_name.split('_')[0]+'_1.jpg'
                    img_paths.append(os.path.join('./demo_data/test_img/', im_name))
                    label_paths.append(os.path.join('./demo_data/test_label/', im_name.replace('.jpg', '.png')))
                    edge_paths.append(os.path.join('./demo_data/test_edge/', c_name))
                    clothes_paths.append(os.path.join('./demo_data/test_color/', c_name))
                    densepose_paths.append(os.path.join('./demo_data/test_densepose/', im_name.replace('.jpg', '.npy')))
                    names.append(im_name.replace('.jpg', '_') + c_name)

            self.img_paths = img_paths
            self.label_paths = label_paths
            self.edge_paths = edge_paths
            self.clothes_paths = clothes_paths
            self.densepose_paths = densepose_paths
            self.names = names

    def __getitem__(self, index):
        # A_data
        # index_a = 126
        index_a = index % self.img_size
        # index_a = index % len(self.img_paths)
        # img_a_path = self.img_paths[index % len(self.img_paths)]
        # label_a_path = self.label_paths[index % len(self.img_paths)]
        # edge_a_path = self.edge_paths[index % len(self.edge_paths)]
        # cloth_a_path = self.clothes_paths[index % len(self.clothes_paths)]
        # dense_a_path = self.densepose_paths[index % len(self.densepose_paths)]
        img_a_path = self.img_paths[index_a]
        label_a_path = self.label_paths[index_a]
        edge_a_path = self.edge_paths[index_a]
        cloth_a_path = self.clothes_paths[index_a]
        dense_a_path = self.densepose_paths[index_a]
        pose_a_path = img_a_path.replace('.png', '_keypoints.json').replace('.jpg', '_keypoints.json').replace(
            self.opt.phase + '_img',
            self.opt.phase + '_pose')

        img_a = Image.open(img_a_path).convert('RGB')
        label_a = Image.open(label_a_path).convert('L')
        edge_a = Image.open(edge_a_path).convert('L')
        cloth_a = Image.open(cloth_a_path).convert('RGB')

        img_a = self.transform_rgb(img_a)
        cloth_a = self.transform_rgb(cloth_a)
        label_a = self.transform_l(label_a) * 255
        edge_a = self.transform_l(edge_a)
        pose_a = self.get_pose_tensor(pose_a_path)
        dense_a = np.load(dense_a_path)
        dense_a = torch.from_numpy(dense_a)[0].unsqueeze(0)

        # B_data
        index_b = random.randint(0, self.img_size - 1)
        if self.opt.isTrain == False:
            index_b = len(self.img_paths) - index - 1
        if self.select == True:
            index_b = index

        # index_b = len(self.img_paths) - index -1
        img_b_path = self.img_paths[index_b]
        label_b_path = self.label_paths[index_b]
        edge_b_path = self.edge_paths[index_b]
        cloth_b_path = self.clothes_paths[index_b]
        dense_b_path = self.densepose_paths[index_b]
        pose_b_path = img_b_path.replace('.png', '_keypoints.json').replace('.jpg', '_keypoints.json').replace(
            self.opt.phase + '_img',
            self.opt.phase + '_pose')

        img_b = Image.open(img_b_path).convert('RGB')
        label_b = Image.open(label_b_path).convert('L')
        edge_b = Image.open(edge_b_path).convert('L')
        cloth_b = Image.open(cloth_b_path).convert('RGB')

        img_b = self.transform_rgb(img_b)
        cloth_b = self.transform_rgb(cloth_b)
        label_b = self.transform_l(label_b) * 255
        edge_b = self.transform_l(edge_b)
        pose_b = self.get_pose_tensor(pose_b_path)
        dense_b = np.load(dense_b_path)
        dense_b = torch.from_numpy(dense_b)[0].unsqueeze(0)

        if self.select:
            input_dict = {'img_a': img_a, 'cloth_a': cloth_a, 'label_a': label_a, 'edge_a': edge_a, 'pose_a': pose_a,
                          'img_b': img_b, 'cloth_b': cloth_b, 'label_b': label_b, 'edge_b': edge_b, 'pose_b': pose_b,
                          'dense_a': dense_a, 'dense_b': dense_b, 'name': self.names[index]}
        else:
            input_dict = {'img_a': img_a, 'cloth_a': cloth_a, 'label_a': label_a, 'edge_a': edge_a, 'pose_a': pose_a,
                          'img_b': img_b, 'cloth_b': cloth_b, 'label_b': label_b, 'edge_b': edge_b, 'pose_b': pose_b,
                          'dense_a': dense_a, 'dense_b': dense_b}

        return input_dict

    def get_pose_tensor(self, pose_path):
        with open(os.path.join(pose_path), 'r') as f:
            pose_label = json.load(f)
            try:
                pose_data = pose_label['people'][0]['pose_keypoints']
            except IndexError:
                pose_data = [0 for i in range(54)]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))
        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            one_map = self.transform_rgb(one_map.convert('RGB'))  # TODO check the transform function
            pose_map[i] = one_map[0]
        p_tensor = pose_map
        return p_tensor

    def __len__(self):
        return self.img_size
