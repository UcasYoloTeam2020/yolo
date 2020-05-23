# 新的xmltolabels,以便于数据增广
# CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
          'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
          'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
# print(len(CLASSES))
import xml.etree.ElementTree as ET
import os
import cv2
import random
import shutil

# 为了方便数据增广，这部分函数其实没用
def convert(size, box):
    """将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
    并进行归一化
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return (x, y, w, h)
    
    


def convert_annotation(DATASET_PATH,image_id):
    """把图像image_id的xml文件转换为目标检测的label文件(txt)
    其中包含物体的类别，bbox的左上角点坐标以及bbox的宽、高
    并将四个物理量归一化"""
    in_file = open(DATASET_PATH + 'Annotations/%s' % (image_id))
    image_id = image_id.split('.')[0]
    no_cls = True
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASSES:
            continue
        no_cls = False
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), points)
        with open(DATASET_PATH+'labels_20/%s.txt' % (image_id), 'a+') as out_file:
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    
    if no_cls == False:
        if random.random()<0.7:
            with open(DATASET_PATH+"ImageSets/Main/train.txt",'a+') as f:
                f.write(image_id.split('.')[0]+'\n')
        else:
            with open(DATASET_PATH+"ImageSets/Main/val.txt",'a+') as f:
                f.write(image_id.split('.')[0]+'\n')
    in_file.close()
    
    

def make_label_txt(DATASET_PATH):
    """在labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息"""    
    if os.path.exists(DATASET_PATH+'ImageSets/Main') :
        shutil.rmtree(DATASET_PATH+'ImageSets/Main')
    os.makedirs(DATASET_PATH+'ImageSets/Main')
    if os.path.exists(DATASET_PATH+'labels') :
        shutil.rmtree(DATASET_PATH+'labels')
    os.makedirs(DATASET_PATH+'labels')
    
    filenames = os.listdir(DATASET_PATH + 'Annotations')
    for file in filenames:
        convert_annotation(DATASET_PATH,file)
        
    
    



def show_labels_img(dataset_path,imgname):
    """imgname是输入图像的名称，无下标"""
    img = cv2.imread(dataset_path + 'JPEGImages/' + imgname + '.jpg')
    h, w = img.shape[:2]
    print(w,h)
    label = []
    with open(DATASET_PATH+"labels/"+imgname+".txt",'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]
            print(CLASSES[int(label[0])])
            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))
            cv2.putText(img,CLASSES[int(label[0])],pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            cv2.rectangle(img,pt1,pt2,(0,0,255,2))

    cv2.imshow("img",img)
    cv2.waitKey(0)


DATASET_PATH = "VOCdevkit/VOC2007/"
make_label_txt(DATASET_PATH)
# show_labels_img(DATASET_PATH,"000009")
