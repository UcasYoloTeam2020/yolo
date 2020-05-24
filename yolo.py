# -*- coding: utf-8 -*-

import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as tvmodel
from Resnet_dilat import ResNet_dilat

CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
NUM_BBOX = 2

def calculate_iou(bbox1,bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的交集
    if bbox1[2]<bbox2[0] or bbox1[0]>bbox2[2] or bbox1[3]<bbox2[1] or bbox1[1]>bbox2[3]:
        pass
    else:
        intersect_bbox[0] = max(bbox1[0],bbox2[0])
        intersect_bbox[1] = max(bbox1[1],bbox2[1])
        intersect_bbox[2] = min(bbox1[2],bbox2[2])
        intersect_bbox[3] = min(bbox1[3],bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积

    if area_intersect>0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0



class Loss_yolov1(nn.Module):
    def __init__(self,use_focal=False):
        super(Loss_yolov1,self).__init__()
        self.use_focal=use_focal
    def focus_loss_calculate(self, pred, labels,batch_num,gridx_num,gridy_num,m_power=2,weight_factor=0.25):
        focus_loss = 0.0
        for class_num in range(np.size(CLASSES)):
            focus_loss +=  \
                -weight_factor \
                * torch.pow((1- pred[batch_num,10+class_num,gridx_num,gridy_num]),m_power)\
                * torch.log(pred[batch_num,10+class_num,gridx_num,gridy_num]) \
                * labels[batch_num,10+class_num,gridx_num,gridy_num] 
        return focus_loss  
        # return focus_loss/batch_num  

    def forward(self, pred, labels):
        """
        :param pred: (batchsize,17,7,7)的网络输出数据
        :param labels: (batchsize,17,7,7)的样本标签数据
        :return: 当前批次样本的平均损失
        """
        ALPHA = 0.25  #focus loss weight factor for imbalance of PN sample;
        GAMMA = 2 

        num_gridx, num_gridy = labels.size()[-2:]  # 划分网格数量
        num_b = 2              # 每个网格的bbox数量
        num_cls = np.size(CLASSES)  # 类别数量
        noobj_confi_loss = 0.  # 不含目标的网格损失(只有置信度损失)
        coor_loss = 0.         # 含有目标的bbox的坐标损失
        obj_confi_loss = 0.  # 含有目标的bbox的置信度损失
        class_loss = 0.  # 含有目标的网格的类别损失
        n_batch = labels.size()[0]  # batchsize的大小

        for i in range(n_batch):  
            for m in range(7):  
                for n in range(7):  
                    # # y方向网格循环
                    if labels[i,4,m,n]==1:# 如果包含物体
                        # 将数据(px,py,w,h)转换为(x1,y1,x2,y2)
                        # 先将px,py转换为cx,cy，即相对网格的位置转换为标准化后实际的bbox中心位置cx,xy
                        # 然后再利用(cx-w/2,cy-h/2,cx+w/2,cy+h/2)转换为xyxy形式，用于计算iou
                        bbox1_pred_xyxy = ((pred[i,0,m,n]+n)/num_gridx - pred[i,2,m,n]/2,(pred[i,1,m,n]+m)/num_gridy - pred[i,3,m,n]/2,
                                           (pred[i,0,m,n]+n)/num_gridx + pred[i,2,m,n]/2,(pred[i,1,m,n]+m)/num_gridy + pred[i,3,m,n]/2)
                        bbox2_pred_xyxy = ((pred[i,5,m,n]+n)/num_gridx - pred[i,7,m,n]/2,(pred[i,6,m,n]+m)/num_gridy - pred[i,8,m,n]/2,
                                           (pred[i,5,m,n]+n)/num_gridx + pred[i,7,m,n]/2,(pred[i,6,m,n]+m)/num_gridy + pred[i,8,m,n]/2)
                        bbox_gt_xyxy = ((labels[i,0,m,n]+n)/num_gridx - labels[i,2,m,n]/2,(labels[i,1,m,n]+m)/num_gridy - labels[i,3,m,n]/2,
                                        (labels[i,0,m,n]+n)/num_gridx + labels[i,2,m,n]/2,(labels[i,1,m,n]+m)/num_gridy + labels[i,3,m,n]/2)
                        
                        iou1 = calculate_iou(bbox1_pred_xyxy,bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy,bbox_gt_xyxy)
                        # 选择iou大的bbox作为负责物体
                        if iou1 >= iou2:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i,0:2,m,n] - labels[i,0:2,m,n])**2) \
                                        + torch.sum((pred[i,2:4,m,n].sqrt()-labels[i,2:4,m,n].sqrt())**2))
                            obj_confi_loss = obj_confi_loss + (pred[i,4,m,n] - iou1)**2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i,9,m,n]-iou2)**2)
                        else:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i,5:7,m,n] - labels[i,5:7,m,n])**2) \
                                        + torch.sum((pred[i,7:9,m,n].sqrt()-labels[i,7:9,m,n].sqrt())**2))
                            obj_confi_loss = obj_confi_loss + (pred[i,9,m,n] - iou2)**2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中,注意，对于标签的置信度应该是iou1
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i, 4, m, n]-iou1) ** 2)
                        if self.use_focal:
                            class_loss = class_loss + self.focus_loss_calculate(pred,labels,i,m,n,GAMMA,ALPHA)
                        else:
                            class_loss = class_loss + torch.sum((pred[i,10:,m,n] - labels[i,10:,m,n])**2)
                    else:  # 如果不包含物体
                        noobj_confi_loss = noobj_confi_loss + 0.5 * torch.sum(pred[i,[4,9],m,n]**2)

        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss

        return loss/n_batch


class YOLOv1_resnet(nn.Module):
    def __init__(self):#backbone="resnet34"):
        super(YOLOv1_resnet,self).__init__()
        
        resnet = tvmodel.resnet34(pretrained=True)
        # resnet.load_state_dict(torch.load('./models_pkl/resnet34-333f7ec4.pth'),False)

        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中

        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(512,1024,3,padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(),
            nn.Conv2d(1024,1024,3,stride=2,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.LeakyReLU(),
            nn.Linear(4096,7*7*(5*NUM_BBOX+len(CLASSES))),
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )

    def forward(self, input):
        self.padding_mode='none'
        input = self.resnet(input)
        input = self.Conv_layers(input)
        input = input.view(input.size()[0],-1)
        input = self.Conn_layers(input)
        return input.reshape(-1, (5*NUM_BBOX+len(CLASSES)), 7, 7)  # 记住最后要reshape一下输出数据
    def backbone_name():
        return 'resnet34'

class YOLOv1_vgg(nn.Module):
    def __init__(self):
        super(YOLOv1_vgg,self).__init__()
        
        # 调用torchvision里的vgg16预训练模型
        
        #预先下载
        vgg = tvmodel.vgg16(pretrained=True)
        # vgg.load_state_dict(torch.load('./models_pkl/vgg16-00b39a1b.pth'),False)

        self.vgg = vgg.features
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(512,1024,3,padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(),
            nn.Conv2d(1024,1024,3,stride=2,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.LeakyReLU(),
            nn.Linear(4096,7*7*(5*NUM_BBOX+len(CLASSES))),
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )

    def forward(self, input):
        input = self.vgg(input)
        input = self.Conv_layers(input)
        input = input.view(input.size()[0],-1)
        input = self.Conn_layers(input)
        return input.reshape(-1, (5*NUM_BBOX+len(CLASSES)), 7, 7)  # 记住最后要reshape一下输出数据
    def backbone_name(self):
        return 'vgg16'


class YOLOv1_dilat(nn.Module):
    def __init__(self):#backbone="resnet34"):
        super(YOLOv1_resnet,self).__init__()
        

        self.resnet = Resnet_dilat()  # 去除resnet的最后两
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(512,1024,3,padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(),
            nn.Conv2d(1024,1024,3,stride=2,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.LeakyReLU(),
            nn.Linear(4096,7*7*(5*NUM_BBOX+len(CLASSES))),
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )

    def forward(self, input):
        input = self.resnet(input)
        input = self.Conv_layers(input)
        input = input.view(input.size()[0],-1)
        input = self.Conn_layers(input)
        return input.reshape(-1, (5*NUM_BBOX+len(CLASSES)), 7, 7)  # 记住最后要reshape一下输出数据
 