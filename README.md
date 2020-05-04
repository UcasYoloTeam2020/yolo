# yolo
深度学习大作业
## 代码说明 from 徐珊波
1. 训练集和测试集都来自PASCAL VOC2007
2. 为了加快训练速度，现在的训练和测试都用的是动物这7类的图片。
3. train应该可以直接训练
4. test还有一点问题。我电脑的显卡没办法跑，不知道在gpu上会不会报错。
5. 如果之后要加数据，只要改数据文件，并把 .py 里的CLASSES改为对应的类别就可以。并且要用xmlToLabels.py把xml的标注文件变为VOCdevkit\VOC2007\labels 里的对应txt文件。同时，重新对数据进行划分，放在VOCdevkit\VOC2007\ImageSets\Main里的train.txt和val.txt。
## 代码测试说明 from 姚东盼
### 环境：
-  python3.x
-  torch  ： 0.4.1
-  opencv :  3.4.3.18
-  torchvision: 0.2.1

## 数据增广说明  
### 作用  
生成增广后的图片数据及其标签文件  
### 运行方式  
`当前路径下运行make_augmentation.py`  
- 增广后的图片存储在VOCdevkit\VOC2007\JPEGImages_augmentation  
- 增广后的标签存储在VOCdevkit\VOC2007\labels_augmentation  
- 增广后的训练集、测试集图片名称文件存储在VOCdevkit\VOC2007\ImageSets\Main\train_augmentation.txt、VOCdevkit\VOC2007\ImageSets\Main\val_augmentation.txt  
### 对应的修改  
`若使用增广数据，需将train.py及test.py中的图片路径、标签路径、图片名称路径替换为增广后的路径`

## Train.py 更新说明：
1.增广后个别数据，可能由于随机裁剪原因，部分数据标签的bbox中心在img边缘，宽度为0。程序报错。
  添加my_collate函数，剔除无效数据。
2.添加log输出。
