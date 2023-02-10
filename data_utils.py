from typing import Iterable, Union, List
import torch
from torch.utils.data import TensorDataset
import numpy as np
import os
from sklearn import preprocessing
import random
from sklearn.model_selection import KFold
import scipy.io as sio

def set_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_mat(dirname):
    print(dirname)
    feature_arr = np.load(os.path.join(dirname, "feature.npy"))
    # print(feature_arr.shape)
    label_arr = np.load(os.path.join(dirname, "label.npy"), allow_pickle=True)
    # label_arr += 1
    cumulative_arr = np.load(os.path.join(dirname, "cumulative.npy"))
    return feature_arr, label_arr , cumulative_arr

def kfold_train_test_split_SEED_subject1session1(feature_arr, label_arr):
    feature0 = []
    feature1 = []
    feature2 = []
    print('feature_arr',feature_arr.shape)
    label_arr+=1
    for i in range(feature_arr.shape[0]):
        if label_arr[i] == 0:
                feature0.append((feature_arr[i,:,:,:,:]))

        elif label_arr[i] == 1:
                feature1.append((feature_arr[i,:,:,:,:]))

        else:
                feature2.append((feature_arr[i,:,:,:,:]))

    feature0 = np.array(feature0)
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    label0 = np.hstack((label_arr[np.where(label_arr==0)]))
    label1 = np.hstack((label_arr[np.where(label_arr==1)]))
    label2 = np.hstack((label_arr[np.where(label_arr==2)]))

    idx0 = np.random.permutation(feature0.shape[0])
    idx1 = np.random.permutation(feature1.shape[0])
    idx2 = np.random.permutation(feature2.shape[0])

    data = np.vstack((feature0[idx0], feature1[idx1], feature2[idx2]))
    label = np.hstack((label0[idx0], label1[idx1], label2[idx2]))

    return data, label


def kfold_train_test_split_SEEDIV_subject1session1(feature_arr, label_arr):
    feature0 = []
    feature1 = []
    feature2 = []
    feature3 = []
    print('feature_arr',feature_arr.shape)
    print('label_arr',label_arr.shape)
    label_arr=np.squeeze(label_arr)
    print('label_arr',label_arr.shape)
    for i in range(feature_arr.shape[0]):
        if label_arr[i] == 0:
                feature0.append((feature_arr[i,:,:,:,:]))

        elif label_arr[i] == 1:
                feature1.append((feature_arr[i,:,:,:,:]))

        elif label_arr[i] == 2:
                feature2.append((feature_arr[i,:,:,:,:]))

        else:
                feature3.append((feature_arr[i,:,:,:,:]))

    feature0 = np.array(feature0)
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    feature3 = np.array(feature3)
    label0 = np.hstack((label_arr[np.where(label_arr==0)]))
    label1 = np.hstack((label_arr[np.where(label_arr==1)]))
    label2 = np.hstack((label_arr[np.where(label_arr==2)]))
    label3 = np.hstack((label_arr[np.where(label_arr==3)]))

    idx0 = np.random.permutation(feature0.shape[0])
    idx1 = np.random.permutation(feature1.shape[0])
    idx2 = np.random.permutation(feature2.shape[0])
    idx3 = np.random.permutation(feature3.shape[0])

    data = np.vstack((feature0[idx0], feature1[idx1], feature2[idx2], feature3[idx3]))
    label = np.hstack((label0[idx0], label1[idx1], label2[idx2], label3[idx3]))

    return data, label
