#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:18:52 2018

@author: JinganFeng
"""
import pickle
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def cal_disp(output,proba,groundtruth,title):
    print(title) 
    roc_auc_disp(proba,groundtruth)
    
    TP=0
    FP=0
    FN=0
    TN=0

    for i in range(len(groundtruth)):
        if groundtruth[i]==1 and output[i]==1:
            TP += 1
        if groundtruth[i]==0 and output[i]==1:
            FP += 1
        if groundtruth[i]==1 and output[i]==0:
            FN += 1
        if groundtruth[i]==0 and output[i]==0:
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
    print('\n')

def roc_auc_disp(proba,groundtruth):
    y=groundtruth
    scores = proba
    fpr,tpr,t1hresholds = metrics.roc_curve(y,scores)
    plt.plot(fpr,tpr,marker='o')
    plt.show()
    AUC = auc(fpr,tpr)
    print("auc:",AUC)

############################################################################################################
result = np.load('./network1_result/train.npy')
real_results1 = result[0]
dect_results1 = result[1]
probabilitys1 = result[2]
names1 = result[3]

real_results1 = real_results1.astype(np.int)
dect_results1 = dect_results1.astype(np.int)
probabilitys1 = probabilitys1.astype(np.float32)
names1 = names1.astype(np.str)

result_fc7 = np.load('./network1_result/train_fc7.npy')

model_fc7 = svm.SVC(probability=True)
model_fc7.fit(result_fc7,real_results1)

svm_output_rfcn_test_fc7 = model_fc7.predict(result_fc7)
svm_output_proba = model_fc7.predict_proba(result_fc7)

cal_disp(svm_output_rfcn_test_fc7,svm_output_proba[:,1],real_results1,'### train_set(fc7+SVM) ###')
       

###############################################################

result = np.load('./network1_result/rfcn_test.npy')
real_results1 = result[0]
dect_results1 = result[1]
probabilitys1 = result[2]
names1 = result[3]

real_results1 = real_results1.astype(np.int)
dect_results1 = dect_results1.astype(np.int)
probabilitys1 = probabilitys1.astype(np.float32)
names1 = names1.astype(np.str)

result_fc7 = np.load('./network1_result/rfcn_test_fc7.npy')

svm_output_rfcn_test_fc7 = model_fc7.predict(result_fc7)
svm_output_proba = model_fc7.predict_proba(result_fc7)

cal_disp(svm_output_rfcn_test_fc7,svm_output_proba[:,1],real_results1,'### test_set with rfcn dt box(fc7+SVM) ###')

############################################################################################################         
         
result = np.load('./network1_result/train.npy')
real_results1 = result[0]
dect_results1 = result[1]
probabilitys1 = result[2]
names1 = result[3]

real_results1 = real_results1.astype(np.int)
dect_results1 = dect_results1.astype(np.int)
probabilitys1 = probabilitys1.astype(np.float32)
names1 = names1.astype(np.str)

result_fc7 = np.load('./network1_result/train_fc7.npy')

model_fc7 = RandomForestClassifier(n_estimators=100)
model_fc7.fit(result_fc7,real_results1)

rf_output_rfcn_test_fc7 = model_fc7.predict(result_fc7)
rf_output_proba = model_fc7.predict_proba(result_fc7)
cal_disp(rf_output_rfcn_test_fc7,rf_output_proba[:,1],real_results1,'### train_set(fc7+RF) ###')
        
###############################################################

result = np.load('./network1_result/rfcn_test.npy')
real_results1 = result[0]
dect_results1 = result[1]
probabilitys1 = result[2]
names1 = result[3]

real_results1 = real_results1.astype(np.int)
dect_results1 = dect_results1.astype(np.int)
probabilitys1 = probabilitys1.astype(np.float32)
names1 = names1.astype(np.str)

result_fc7 = np.load('./network1_result/rfcn_test_fc7.npy')

rf_output_rfcn_test_fc7 = model_fc7.predict(result_fc7)
rf_output_proba = model_fc7.predict_proba(result_fc7)
cal_disp(rf_output_rfcn_test_fc7,rf_output_proba[:,1],real_results1,'### test_set with rfcn dt box(fc7+RF) ###')
      
#############################################################################################################

result = np.load('./network1_result/train.npy')
real_results1 = result[0]
dect_results1 = result[1]
probabilitys1 = result[2]
names1 = result[3]

real_results1 = real_results1.astype(np.int)
dect_results1 = dect_results1.astype(np.int)
probabilitys1 = probabilitys1.astype(np.float32)
names1 = names1.astype(np.str)

result_fc7 = np.load('./network1_result/train_fc7.npy')


model_fc7 = AdaBoostClassifier(n_estimators=100)
model_fc7.fit(result_fc7,real_results1)

rf_output_rfcn_test_fc7 = model_fc7.predict(result_fc7)
rf_output_proba = model_fc7.predict_proba(result_fc7)
cal_disp(rf_output_rfcn_test_fc7,rf_output_proba[:,1],real_results1,'### train_set(fc7+AB) ###')

###############################################################

result = np.load('./network1_result/rfcn_test.npy')
real_results1 = result[0]
dect_results1 = result[1]
probabilitys1 = result[2]
names1 = result[3]

real_results1 = real_results1.astype(np.int)
dect_results1 = dect_results1.astype(np.int)
probabilitys1 = probabilitys1.astype(np.float32)
names1 = names1.astype(np.str)

result_fc7 = np.load('./network1_result/rfcn_test_fc7.npy')

rf_output_rfcn_test_fc7 = model_fc7.predict(result_fc7)
rf_output_proba = model_fc7.predict_proba(result_fc7)
cal_disp(rf_output_rfcn_test_fc7,rf_output_proba[:,1],real_results1,'### test_set with rfcn dt box(fc7+AB) ###')

#############################################################################################################

result = np.load('./network1_result/train.npy')
real_results1 = result[0]
dect_results1 = result[1]
probabilitys1 = result[2]
names1 = result[3]

real_results1 = real_results1.astype(np.int)
dect_results1 = dect_results1.astype(np.int)
probabilitys1 = probabilitys1.astype(np.float32)
names1 = names1.astype(np.str)

result_fc7 = np.load('./network1_result/train_fc7.npy')

model_fc7 = KNeighborsClassifier(2)
model_fc7.fit(result_fc7,real_results1)
  
rf_output_rfcn_test_fc7 = model_fc7.predict(result_fc7)
rf_output_proba = model_fc7.predict_proba(result_fc7)
cal_disp(rf_output_rfcn_test_fc7,rf_output_proba[:,1],real_results1,'### train_set(fc7+NN) ###')

###############################################################

result = np.load('./network1_result/rfcn_test.npy')
real_results1 = result[0]
dect_results1 = result[1]
probabilitys1 = result[2]
names1 = result[3]

real_results1 = real_results1.astype(np.int)
dect_results1 = dect_results1.astype(np.int)
probabilitys1 = probabilitys1.astype(np.float32)
names1 = names1.astype(np.str)

result_fc7 = np.load('./network1_result/rfcn_test_fc7.npy')

rf_output_rfcn_test_fc7 = model_fc7.predict(result_fc7)
rf_output_proba = model_fc7.predict_proba(result_fc7)
cal_disp(rf_output_rfcn_test_fc7,rf_output_proba[:,1],real_results1,'### test_set with rfcn dt box(fc7+NN) ###')




