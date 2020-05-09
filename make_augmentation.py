# -*- coding: utf-8 -*-

CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep']

import os
import shutil
import cv2
import numpy as np
import random
import torch

DATASET_PATH = 'VOCdevkit/VOC2007/'

class DataAugmentation:
    def __init__(self):
        self.img_path = DATASET_PATH + "JPEGImages/"  # 原始图像所在的路径
        self.label_path = DATASET_PATH + "labels/"  # 图像对应的label文件(.txt文件)的路径
        self.save_img_path = DATASET_PATH + "JPEGImages_augmentation/"  # 增广后的数据存储路径
        self.train_path = DATASET_PATH + "ImageSets/Main/train.txt"  # 训练集图像名称文件
        self.val_path = DATASET_PATH + "ImageSets/Main/val.txt"  # 测试集图像名称文件
        # 存储路径
        self.save_img_path = DATASET_PATH + "JPEGImages_augmentation/"   # 增广后的数据存储路径
        self.save_label_path = DATASET_PATH + "labels_augmentation/"  # 增广后的标签(.txt文件)存储路径
        self.save_train_path = DATASET_PATH + "ImageSets/Main/train_augmentation.txt" # 增广后的训练集图像名称的txt文件
        self.save_val_path = DATASET_PATH + "ImageSets/Main/val_augmentation.txt"  # 增广后的测试集图像名称的txt文件

        # 为存储增广后的图片和label创建文件夹
        if os.path.exists(self.save_img_path):
            shutil.rmtree(self.save_img_path)
        os.mkdir(self.save_img_path)
        if os.path.exists(self.save_label_path):
            shutil.rmtree(self.save_label_path)
        os.mkdir(self.save_label_path)

        # 数据增广
        self.augmentation(is_train=True)    # train set
        self.augmentation(is_train=False)   # val set

    # 读取图像对应的bbox信息，按1维的方式储存，每5个元素表示一个bbox的(cls,xc,yc,w,h)
    # xc,yc,w, 是归一化的结果, xc,w是横向的占比，yc,h是纵向占比
    #   -------------------
    #   |          |      |
    #   |         yc      |
    #   |         |       |
    #   | -- xc --*       |
    #   |                 |
    #   -------------------
    def readLabels(self, path, filename):
        with open(path + filename + ".txt") as f:
            bbox = f.read().split('\n')
            bbox = [x.split() for x in bbox]
            bbox = [float(x) for y in bbox for x in y]
            if len(bbox) % 5 != 0:
                raise ValueError("File:" + path + filename + ".txt" + "——bbox Extraction Error!")
        return bbox


    def augmentation(self, is_train=True):
        file_list = []
        if is_train:
            with open(self.train_path, 'r') as f:
                filenames = [x.strip() for x in f]
            if os.path.exists(self.save_train_path):
                os.remove(self.save_train_path)
        else:
            with open(self.val_path, 'r') as f:
                filenames = [x.strip() for x in f]
            if os.path.exists(self.save_val_path):
                os.remove(self.save_val_path)

        for item in range(len(filenames)):
            img = cv2.imread(self.img_path + filenames[item] + ".jpg")  # 读取原始图像
            # 读取图像对应的bbox信息，按1维的方式储存
            bbox = self.readLabels(self.label_path, filenames[item])

            # 数据增广
            self.save_img(img, bbox, filenames[item])
            file_list.append(filenames[item])
            # 颠倒
            bbox_temp = bbox.copy()
            img_temp = img.copy()
            img_temp, bbox_temp = self.rand_flip(img_temp, bbox_temp)
            save_name = filenames[item] + "_flip"
            self.save_img(img_temp, bbox_temp, save_name)
            file_list.append(save_name)

            # 放缩
            bbox_temp = bbox.copy()
            img_temp = img.copy()
            img_temp = self.rand_scale(img_temp)
            save_name = filenames[item] + "_scale"
            self.save_img(img_temp, bbox_temp, save_name)
            file_list.append(save_name)

            # 滤波
            bbox_temp = bbox.copy()
            img_temp = img.copy()
            img_temp = self.rand_blur(img_temp)
            save_name = filenames[item] + "_blur"
            self.save_img(img_temp, bbox_temp, save_name)
            file_list.append(save_name)

            # 亮度
            bbox_temp = bbox.copy()
            img_temp = img.copy()
            img_temp = self.rand_bright(img_temp)
            save_name = filenames[item] + "_bright"
            self.save_img(img_temp, bbox_temp, save_name)
            file_list.append(save_name)

            # 色调
            bbox_temp = bbox.copy()
            img_temp = img.copy()
            img_temp = self.rand_hue(img_temp)
            save_name = filenames[item] + "_hue"
            self.save_img(img_temp, bbox_temp, save_name)
            file_list.append(save_name)

            # 饱和度
            bbox_temp = bbox.copy()
            img_temp = img.copy()
            img_temp = self.rand_saturation(img_temp)
            save_name = filenames[item] + "_sat"
            self.save_img(img_temp, bbox_temp, save_name)
            file_list.append(save_name)

            # 随机剪切
            bbox_temp = bbox.copy()
            img_temp = img.copy()
            img_temp, bbox_temp = self.rand_crop(img_temp, bbox_temp)
            save_name = filenames[item] + "_crop"
            self.save_img(img_temp, bbox_temp, save_name)
            file_list.append(save_name)

        if is_train:
            f = open(self.save_train_path, 'w')
        else:
            f = open(self.save_val_path, 'w')
        f.write('\n'.join(file_list))
        f.close()


    # save image
    def save_img(self, im, bbox, save_name):
        cv2.imwrite(self.save_img_path + save_name + '.jpg', im)
        f = open(self.save_label_path + save_name + '.txt', 'w')
        for i in range(len(bbox)):
            f.write(str(bbox[i]))
            if i % 5 == 4:
                f.write('\n')
            else:
                f.write(' ')
        f.close()

    # 翻转
    def rand_flip(self, im, boxes):
        # 做上下和左右翻转
        im[:, :, 0] = np.flip(im[:, :, 0],1).copy()
        im[:, :, 1] = np.flip(im[:, :, 1],1).copy()
        im[:, :, 2] = np.flip(im[:, :, 2],1).copy()
        h, w, _ = im.shape
        # boxes按1维的方式储存，每5个元素表示一个bbox的(cls,xc,yc,w,h)
        for i in range(len(boxes) // 5):
            boxes[i * 5 + 1] = 1.0 - boxes[i * 5 + 1]
            # boxes[i * 5 + 2] = 1.0 - boxes[i * 5 + 2]
        return im, boxes

        # 缩放，横纵向缩放尺度在0.8-1.2间随机选择

    # 放缩
    def rand_scale(self, im):
        scalew = random.uniform(0.8, 1.2)
        scaleh = random.uniform(0.8, 1.2)
        h, w, _ = im.shape
        im = cv2.resize(im, (int(w * scalew), int(h * scaleh)))
        return im

        # 滤波

    # 滤波
    def rand_blur(self, im):
        im = cv2.blur(im, (5, 5))
        return im

        # hsv通道转换

    # 亮度
    def rand_bright(self, im):
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(im_hsv)
        adjust = random.choice([0.5, 1.5])
        v = v * adjust
        v = np.clip(v, 0, 255).astype(im_hsv.dtype)
        im_hsv = cv2.merge((h, s, v))
        im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return im

        # 色调

    # 色调
    def rand_hue(self, im):
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(im_hsv)
        adjust = random.choice([0.5, 1.5])
        h = h * adjust
        h = np.clip(h, 0, 255).astype(im_hsv.dtype)
        im_hsv = cv2.merge((h, s, v))
        im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return im

        # 饱和度

    # 饱和度
    def rand_saturation(self, im):
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(im_hsv)
        adjust = random.choice([0.5, 1.5])
        s = s * adjust
        s = np.clip(s, 0, 255).astype(im_hsv.dtype)
        im_hsv = cv2.merge((h, s, v))
        im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return im

        # 裁剪

    # 裁剪
    def rand_crop(self, im, bbox):
        temp_bbox = bbox.copy()
        temp_bbox = torch.Tensor(temp_bbox)
        temp_bbox = temp_bbox.reshape(-1, 5)
        boxes = temp_bbox[:, 1:]
        labeles = temp_bbox[:, 0]

        h, w, c = im.shape
        rand_h = random.uniform(0.6, 1)
        rand_w = random.uniform(0.6, 1)
        rand_x = random.uniform(0, 1 - rand_w)
        rand_y = random.uniform(0, 1 - rand_h)

        cen = boxes[:, 0:2].clone()
        cen = cen - torch.Tensor([[rand_x, rand_y]]).expand_as(cen)
        mask1 = (cen[:, 0] > 0) & (cen[:, 0] < rand_w)
        mask2 = (cen[:, 1] > 0) & (cen[:, 0] < rand_h)
        mask = (mask1 & mask2).view(-1, 1)
        # 筛选出中心点仍留在裁剪区域内的box
        boxes_crop = boxes[mask.expand_as(boxes)].view(-1, 4)
        labeles_crop = labeles[mask.view(-1)]
        if len(boxes_crop) == 0:  # 裁剪区域内没有box，故不做裁剪
            return im, bbox

        boxes_crop = boxes_crop - torch.Tensor([[rand_x, rand_y, 0, 0]]).expand_as(boxes_crop)
        lr_crop = boxes_crop[:, 0:2] - boxes_crop[:, 2:] / 2.0  # 左上角坐标
        rd_crop = boxes_crop[:, 0:2] + boxes_crop[:, 2:] / 2.0  # 右下角

        # 判断box边框是否在裁剪区域内，更新box位置
        for i, value in enumerate(lr_crop):
            lr_crop[i, 0] = torch.clamp(value[0], 0, max=rand_w) / rand_w
            lr_crop[i, 1] = torch.clamp(value[1], 0, max=rand_h) / rand_h
            rd_crop[i, 0] = torch.clamp(rd_crop[i, 0], 0, max=rand_w) / rand_w
            rd_crop[i, 1] = torch.clamp(rd_crop[i, 1], 0, max=rand_h) / rand_h
            boxes_crop[i, 2] = rd_crop[i, 0] - lr_crop[i, 0]
            boxes_crop[i, 3] = rd_crop[i, 1] - lr_crop[i, 1]
            boxes_crop[i, 0] = (rd_crop[i, 0] + lr_crop[i, 0]) / 2.0
            boxes_crop[i, 1] = (rd_crop[i, 1] + lr_crop[i, 1]) / 2.0
        # import pdb;pdb.
        img_crop = im[int(rand_y * h):int((rand_y + rand_h) * h), int(rand_x * w):int((rand_x + rand_w) * w)]
        labeles_crop = labeles_crop.reshape(-1, 1)
        temp = torch.cat((labeles_crop, boxes_crop), 1).tolist()
        temp = [float(x) for y in temp for x in y]
        return img_crop, temp


if __name__ == '__main__':
    arg = DataAugmentation()
    arg.augmentation(True)