#encoding:utf-8


import numpy as np
from test import BBOX_Pred_Path,DATASET_PATH,CLASSES,model_name
from train import calculate_iou
from collections import defaultdict
import os

labels_Path=DATASET_PATH + "labels_for_eval/"  #train代码中生成的标签文件，需要与train代码同步
evoc_file='eval_test.log'
def caculate_ap(rec,prec,use_07_metric=False):
    if use_07_metric:
    # 11 point metric
        ap = 0.
        for t in np.arange(0.,1.1,0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec>=t])
            ap = ap + p/11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.],rec,[1.]))
        mpre = np.concatenate(([0.],prec,[0.]))

        for i in range(mpre.size -1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
def get_map(preds,target,VOC_CLASSES=CLASSES,threshold=0.5,use_07_metric=False,):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    aps = []
    for i,class_ in enumerate(VOC_CLASSES):
        pred = preds[class_] #[[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0: #如果这个类别一个都没有检测到的异常情况
            ap = -1
            print('---class {} ap {}---'.format(class_,ap))
            aps += [ap]
            continue
        #print(pred)
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        for (key1,key2) in target:
            if key2 == class_:
                npos += len(target[(key1,key2)]) #统计这个类别的正样本，在这里统计才不会遗漏
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d,image_id in enumerate(image_ids):
            bb = BB[d] #预测框
            if (image_id,class_) in target:
                BBGT = target[(image_id,class_)] #[[],]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    overlaps = calculate_iou(bbgt,bb)
                    # print(overlaps)
                    # if overlaps<0.1:
                    #   import pdb;pdb.set_trace()
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt) #这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id,class_)] #删除没有box的键值
                        break
                fp[d] = 1-tp[d]
                # if fp[d]>0.8:
                #     import pdb;pdb.set_trace()
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/float(npos)
        prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
        #print(rec,prec)
        ap = caculate_ap(rec, prec, use_07_metric)
        print('---class {} ap {}---'.format(class_,ap))
        print('---class {} ap {}---'.format(class_,ap),file=evoc_log)
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)),file=evoc_log)
    print('---map {}---'.format(np.mean(aps)))

def get_preds(filelist):
    preds=defaultdict(list)
    for i,file in enumerate(filelist):
        with open(BBOX_Pred_Path+file,'r') as f:
                bboxs=f.readlines()
                for bbox in bboxs: 
                    bbox=[float(x.strip()) for x in bbox.split()]
                    #按类添加bbox信息[imgid,confi,x1,x2,y1,y2]
                    preds[CLASSES[int(bbox[0])]].append([file[:-4]]+[bbox[5]]+bbox[1:5])#按bbox信息添加imgid
    return preds

def deconvert_bbox(bbox):
    #把’cls,xc,yc,w,h'转为‘cls,x,y,x,y'
    [CLS,xc,yc,w,h]=[x for x in bbox]
    xmin=xc-w/2
    xmax=xc+w/2
    ymin=yc-h/2
    ymax=yc+h/2

    return [CLS,xmin,ymin,xmax,ymax]

def get_targets(filelist):
    targets=defaultdict(list)
    for i,file in enumerate(filelist):
        with open(labels_Path+file,'r') as f:
            bboxs=f.readlines()
            for bbox in bboxs:
                bbox=[float(x.strip()) for x in bbox.split()]
                bbox=deconvert_bbox(bbox)
                imgid=file[:-4]
                targets[imgid,CLASSES[int(bbox[0])]].append(bbox[1:])
    return targets
if __name__ == '__main__':   
    evoc_log=open(evoc_file,'a+')
    # evoc_log.write('*'*15+model_name+'*'*15+'\n')
    evoc_log.write('*'*5+"预测数据为labels，测试test文件和eval_voc函数"+'*'*5+'\n')
    evoc_log.write('*'*15+"测试训练集"+'*'*15+'\n')
    evoc_log.write('*'*5+"更换labels文件夹：由labels_aug-->labels"+'*'*5+'\n')
    filelist=os.listdir(BBOX_Pred_Path)
    # print(len(filelist))
    preds=get_preds(filelist)
    targets=get_targets(filelist)
    get_map(preds,targets)

