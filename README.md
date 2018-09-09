The code and the algorithm are for non-commercial use only.

Paper : "Automated Strabismus Detection based on Deep neural networks for Telemedicine Applications"

Author: Jiewei Lu, Jingan Feng, Zhun Fan, Longtao Huang, Ce Zheng, Wenji Li

        (12jwlu1@stu.edu.cn, 13jafeng@stu.edu.cn, zfan@stu.edu.cn, 17lthuang@stu.edu.cn, zhengce@hotmail.com, liwj@stu.edu.cn)

Date  : September 9, 2018

Version : 1.0

Copyright (c) 2018, Jiewei Lu, Jingan Feng.

--------------------------------------------------------------

Notes:
  1) Tensorflow, an open source machine learning framework, is required for the implementation.

  2) R-FCN for eye region segmentation, is based on the TensorFlow Object Detection API. Please refer to the source code at https://github.com/tensorflow/models/tree/master/research/object_detection

  3) Deep CNN for eye region classification, is based on the TensorFlow-Slim image classification model library. Please refer to the source code at https://github.com/tensorflow/models/tree/master/research/slim

  4) scikit-learn, a machine learning library, is required for different classifiers (SVM, Random Forests, Nearest Neighbor and Adaboost) and an evaluation metrics (AUC).

  These libraries can be easily set by packet manager on linux systems
  Because there is an agreement with doctors and patients, we are temporarily unable to publish the data.

--------------------------------------------------------------

This folder contains three sub-directories:

  - detection 
     - eye_detection.py       the source code of using R-FCN to segment eye region
     - IMG_FILE               contains some example images for testing R-FCN
     - XML_FILE               contains corresponding bounding box information for each example image
  - classification
     - eye_classification.py  the source code of using deep CNN to classify eye region
     - network1               contains the design file of the 1st network architecture
     - network2               contains the design file of the 2nd network architecture
     - CROP_IMAGE             contains some example images for testing the deep CNNs
  - evaluation
     - roc_auc.py             the source code of calculating and displaying evaluation metrics of our two deep CNNs
     - svm_train.py           the source code of using feature maps to train different classifiers and displaying evaluation metrics
     - network1_result
        - train.npy           the numpy file saving goundtruth label of train set 
        - train_fc7.npy       the numpy file saving feature map output of train set  
        - rfcn_test.npy       the numpy file saving goundtruth label of test set (detected by R-FCN)
        - rfcn_test_fc7.npy   the numpy file saving feature map output of test set (detected by R-FCN)
     - network2_result
     - detection_result       contains the numpy file saving IOU output of R-FCN for each images in test set

