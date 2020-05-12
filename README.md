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
### test代码说明
` 代码可直接运行
1. 代码输出测试结果并保存bbox信息
2. 测试结果保存在\Pred_img\中
3. bbox结果以txt格式保存在VOCdevkit\VOC2007\bboxs_preds\中
4. 可通过标志位Train设置测试验证集或者训练集
### eval_voc代码说明
1. 运行eval_voc代码前需要运行更新的test文件，生成预测的bbox信息
2. 生成预测的bbox信息后，可直接运行eval_voc
3. 一些默认参数说明：
- 默认的labels查询目录为增广数据目录\VOCdevkit\VOC2007\labels_augmentation\
- eval_voc中CLASS、DATASET_PATH和预测bbox目录默认于test文件同步，若不需要同步时可手动设置

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
1. 增广后个别数据，可能由于随机裁剪原因，部分数据标签的bbox中心在img边缘，宽度为0。程序报错。
  添加my_collate函数，剔除无效数据。
2. 添加log输出。

## Resnet_dilat文件说明from赵千帆
 可以直接调用文件中的resnet_dilat函数代替model.resnet32赋值给resnet。

## map计算更新说明：
1. 由于训练集训练的时候，对照片做了padding，然后做了归一化。为了避免在eval时，重新计算padding的参数。因此在数据集里面，添加了一个mk_eval_label标志位，用于确定是否保存padding后格式的标签。test测试时，可以将改标志位置True。保存标签。 
2. 保存标签的位置在'VOCdevkit/VOC2007/labels_for_eval/‘ 
3. test运行时，需要将mk_eval_labels置为True  
4. eval_voc运行时，需要将labels_path 替换为 VOCdevkit/VOC2007/labels_for_eval/ 再运行 
` 代码详细修改说明(更新的代码） 
- train数据集修改,分别在L26-32 L72-75 L79 L112 L122
- test dataloader改为：val_dataloader = DataLoader(VOC2007(is_train=Train,mk_eval_label=True), batch_size=1, shuffle=False)
- eval_voc 修改：labels_Path修改为VOCdevkit/VOC2007/labels_for_eval/

