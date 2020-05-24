##  概述
- 国科大19深度学习大作业。 
  本次作业，我们拟实现目标检测经典算法YOLOV1的网络框架，并评估其性能。为此，我们参考了部分网络上YOLO V1相关的代码制作了本次作业的代码，实现了基于Resnet、VGG的YOLOV1框架，并在YOLOV1_resnet的基础上做了数据增广、focal loss和空洞卷积的尝试
##  运行环境
- Pytorch : 1.0.0
- Python : 3.6
- Opencv : 3.4.3.18
- Torchvision: 0.2.1
##  数据集初始化说明
1. 模型训练及测试的数据集为VOC2007，数据集下载可运行download_voc07.sh  
 `chmod u+x download_voc07.sh`  
 `./download_voc07.sh`   
 `tar -xf VOCtrainval_06-Nov-2007.tar`  
2.  由于计算资源有限，我们仅使用了VOC2007中的7类目标，若要调整目标数量，需在train.py中修改CLASSES中的类别
3. 提取VOC2007选择的类别标签信息，转成YOLOv1的标签，并保存训练集和验证集的数据列表：  
    `python xmlToLabels.py`
4. 如需要在预训练模型上的训练集和验证集上训练和测试,执行以下命令：   
  `cp pretrain_datalist/*.txt VOCdevkit/VOC2007/ImageSets/Main/`

##  训练
1. 模型默认使用无增广数据集训练，如需数据增广，请先运行：  
`python make_augmentation.py`  
并更改VOC2007中训练数据列表及图片、标签位置分别为：  
`ImageSets\Main\train_augmentation.txt`   
`JPEGImages_augmentation`   
`labels_augmentation`  
2. 训练: `python train.py`  

## 测试
1.  测试预训练模型，需要从百度网盘下载，下载链接为：  
链接：https://pan.baidu.com/s/1HdVBVkSBXjYLimKnSKRXDQ   
提取码：w8n0   
2. 选择测试的模型，并更改模型的的载入路径  
- 测试YOLOv1_resnet：  
   `model = YOLOv1_resnet().cuda()`   
   `model.load_state_dict(torch.load("models_pth/YOLOv1_resnet_epoch50.pth"))`   
- 测试YOLOv1_vgg:   
   `model = YOLOv1_vgg().cuda()`   
   `model.load_state_dict(torch.load("../YOLOv1_vgg16_epoch50.pth"))`   
- 测试YOLOv1_resnet_DataAug:    
`model = YOLOv1_resnet().cuda()`   
`model.load_state_dict(torch.load("models_pth/YOLOv1_resnet_DataAug_epoch40.pth"))`   
3. 测试：
` python test.py`   


