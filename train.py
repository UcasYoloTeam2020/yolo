# -*- coding: utf-8 -*-


#为了加快计算，只用了7类，
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
# CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
#           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
#           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

import os
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as tvmodel
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate
import time
import shutil
from yolo import Loss_yolov1,YOLOv1_resnet,YOLOv1_vgg,YOLOv1_dilat

DATASET_PATH = '../YOLO/VOCdevkit/VOC2007/'
NUM_BBOX = 2
labels_for_eval_Path=DATASET_PATH + "labels_for_eval/"

def clr_eval_label():    #清除用于评估map的labels文件夹标签
  if os.path.exists(labels_for_eval_Path):
     shutil.rmtree(labels_for_eval_Path)
  os.mkdir(labels_for_eval_Path)

def convert_bbox2labels(bbox):
    """将bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
    注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式"""
    gridsize = 1.0/7
    labels = np.zeros((7,7,5*NUM_BBOX+len(CLASSES)))  # 注意，此处需要根据不同数据集的类别个数进行修改
    for i in range(len(bbox)//5):
        gridx = int(bbox[i*5+1] // gridsize)  # 当前bbox中心落在第gridx个网格,列
        gridy = int(bbox[i*5+2] // gridsize)  # 当前bbox中心落在第gridy个网格,行
        if gridx>=7 or gridy>=7:        
            return None          #随机裁剪造成的 bbox中心在图像边缘，为无效数据
        # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
        gridpx = bbox[i * 5 + 1] / gridsize - gridx
        gridpy = bbox[i * 5 + 2] / gridsize - gridy
        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 10+int(bbox[i*5])] = 1
    # import pdb;pdb.set_trace()
    return labels
def my_collate(batch):
    #剔除cinvert_bbox2中 label置为None的数据
    "Puts each data field into a tensor with outer dimension batch size"

    batch = [(img,labels,filename) for (img,labels,filename) in batch if labels is not None]
    if batch==[]:
      return (None,None,None)
    return default_collate(batch)
class VOC2007(Dataset):
    def __init__(self,is_train=True,is_aug=True,mk_eval_label=False):
        """
        :param is_train: 调用的是训练集(True)，还是验证集(False)
        :param is_aug:  是否进行数据增广
        """
        self.filenames = []  # 储存数据集的文件名称
        if is_train:
            with open(DATASET_PATH + "ImageSets/Main/train.txt", 'r') as f: # 调用包含训练集图像名称的txt文件
                self.filenames = [x.strip() for x in f]
        else:
            with open(DATASET_PATH + "ImageSets/Main/val.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]
        if mk_eval_label:
            print("重新生成eval_lables标签")
            clr_eval_label()
        self.imgpath = DATASET_PATH + "JPEGImages/"  # 原始图像所在的路径
        self.labelpath = DATASET_PATH + "labels/"  # 图像对应的label文件(.txt文件)的路径
        self.is_aug = is_aug
        self.mk_eval_label=mk_eval_label

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        
        img = cv2.imread(self.imgpath+self.filenames[item]+".jpg")  # 读取原始图像
        h,w = img.shape[0:2]
        input_size = 448  # 输入YOLOv1网络的图像尺寸为448x448
        # 因为数据集内原始图像的尺寸是不定的，所以需要进行适当的padding，将原始图像padding成宽高一致的正方形
        # 然后再将Padding后的正方形图像缩放成448x448
        padw, padh = 0, 0  # 要记录宽高方向的padding具体数值，因为padding之后需要调整bbox的位置信息
        if h>w:
            padw = (h - w) // 2
            img = np.pad(img,((0,0),(padw,padw),(0,0)),'constant',constant_values=0)
        elif w>h:
            padh = (w - h) // 2
            img = np.pad(img,((padh,padh),(0,0),(0,0)), 'constant', constant_values=0)
        img = cv2.resize(img,(input_size,input_size))
        # 图像增广部分，这里不做过多处理，因为改变bbox信息还蛮麻烦的
        if self.is_aug:
            aug = transforms.Compose([transforms.ToTensor()])
            img = aug(img)
        # 读取图像对应的bbox信息，按1维的方式储存，每5个元素表示一个bbox的(cls,xc,yc,w,h)
        with open(self.labelpath+self.filenames[item]+".txt") as f:
            bbox = f.read().split('\n')
        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        if len(bbox)%5!=0:
            raise ValueError("File:"+self.labelpath+self.filenames[item]+".txt"+"——bbox Extraction Error!")
        if self.mk_eval_label:
         ff=open(labels_for_eval_Path+self.filenames[item]+".txt",'w')#保存[cls,xc,yc,w,h格式的标签]，方便评估map
        # 根据padding、图像增广等操作，将原始的bbox数据转换为修改后图像的bbox数据
        for i in range(len(bbox)//5):
            if padw != 0:
                bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w + padw) / h
                bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h
            elif padh != 0:
                bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h + padh) / w
                bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w
            if self.mk_eval_label:
                ff.write(' '.join([str(x) for x in bbox[i*5:i*5+5]])+'\n')#保存标签，方便评估模型map
            # 此处可以写代码验证一下，查看padding后修改的bbox数值是否正确，在原图中画出bbox检验
        labels = convert_bbox2labels(bbox)  # 将所有bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
        # 此处可以写代码验证一下，经过convert_bbox2labels函数后得到的labels变量中储存的数据是否正确
        if labels is None:
           return (None,None,None)       
        labels = transforms.ToTensor()(labels)
        filename=self.filenames[item]
        return img,labels,filename


def train_info2log(trainlog,num_batch,batchsize,lr):
    #注释函数

    print(time.strftime('%Y-%m-%d %H:%m',time.localtime(time.time())),file=trainlog)
    print('数据集：VOC2007_7类',file=trainlog)
    print("---训练集数量 %d"%(num_batch*batchsize),file=trainlog)
    print("---BatchSize:  %d  , lr : %.8f "%(batchsize,lr),file=trainlog)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'   
    trainlog = open('./train.log', mode = 'a',encoding='utf-8')
    epoch = 100
    batchsize = 8
    lr = 0.00001
    train_data = VOC2007()
    train_dataloader = DataLoader(VOC2007(is_train=True),batch_size=batchsize,collate_fn=my_collate,shuffle=True)
    train_info2log(trainlog,len(train_dataloader),batchsize,lr)
    model = YOLOv1_vgg().cuda()
    model.load_state_dict(torch.load("../YOLOv1_vgg16_epoch50.pth"))
    # torch.save(model.state_dict(),"../YOLOv1_vgg16_epoch50.pth")
    # 冻结resnet34特征提取层，特征提取层不参与参数更新
    for layer in model.children():
        layer.requires_grad = False
        break
    criterion = Loss_yolov1(use_focal=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    for e in range(epoch):
        model.train()    
        for i,(inputs,labels,filename) in enumerate(train_dataloader):
            inputs = inputs.cuda()
            labels = labels.float().cuda()
            pred = model(inputs)

            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch %d/%d| Step %d/%d| Loss: %.2f"%(e,epoch,i,len(train_data)//batchsize,loss))
            assert not torch.isnan(loss).any()
            
        print("epoch %d : loss %.4f"%(e,loss),file=trainlog)    
        if (e+1)%10==0:
                import pdb;pdb.set_trace()
                torch.save(model.state_dict(),"./models_pkl/YOLOv1_"+model.backbone_name()+'_epoch'+str(e+1)+".pth")
                print("save model param :./models_pkl/YOLOv1_"+model.backbone_name()+'_epoch'+str(e+1)+".pth",file=trainlog)
                print(time.strftime('%Y-%m-%d %H:%m',time.localtime(time.time())),file=trainlog)



