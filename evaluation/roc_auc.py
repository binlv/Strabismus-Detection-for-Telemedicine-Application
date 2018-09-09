#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 19:52:08 2018

@author: JinganFeng
"""

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc

result = np.load('./network1_result/rfcn_test.npy')
#result = np.load('./network2_result/rfcn_test.npy')

#label: {unnormal:1, normal:0}

real_results1 = result[0]
dect_results1 = result[1]
probabilitys1 = result[2]
names1 = result[3]

real_results1 = real_results1.astype(np.int)
dect_results1 = dect_results1.astype(np.int)
probabilitys1 = probabilitys1.astype(np.float32)
names1 = names1.astype(np.str)

y=real_results1
scores = probabilitys1
fpr,tpr,t1hresholds = metrics.roc_curve(y,scores)
plt.plot(fpr,tpr,marker='o')
plt.show()
AUC = auc(fpr,tpr)
print("auc:",AUC)

TP=0
FP=0
FN=0
TN=0

for i in range(len(real_results1)):
    if real_results1[i]==1 and dect_results1[i]==1:
        TP += 1
    if real_results1[i]==0 and dect_results1[i]==1:
        FP += 1
    if real_results1[i]==1 and dect_results1[i]==0:
        FN += 1
    if real_results1[i]==0 and dect_results1[i]==0:
        TN += 1
print("TP:",TP)
print("TN:",TN)
print("FP:",FP)
print("FN:",FN)

TNR = TN/(TN+FP)
TPR = TP/(TP+FN)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("specificity:",TNR)
print("sensitive:",TPR)
print("accuracy:",ACC)

################################################################

result = np.load('./detection_result/iou_result.npy')
iou_unnormal = result[0]
iou_unnormal_name = result[1]
iou_normal = result[2]
iou_normal_name = result[3]

sum1=sum(iou_unnormal)
sum2=sum(iou_normal)
len1=len(iou_unnormal)
len2=len(iou_normal)
avr=(sum1+sum2)/(len1+len2)
print("iou_mean:",avr)
