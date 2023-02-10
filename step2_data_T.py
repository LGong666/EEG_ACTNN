import numpy as np
import torch
import torch.nn as nn
import math
import scipy.io as sio
import os

def load_mat(dirname):
    feature_arr = np.load(os.path.join(dirname, "feature.npy"))
    label_arr = np.load(os.path.join(dirname, "label.npy"))
    cumulative_arr = np.load(os.path.join(dirname, "cumulative.npy"))
    return feature_arr, label_arr, cumulative_arr

T = 2

for session in range(3):
    for subject in range(15):
        path = '...\\subject' + str(subject + 1) + 'session' + str(session + 1)
        EEGs, labels, cumulative = load_mat(path)
        EEGs = np.squeeze(EEGs)
        print('features=', EEGs.shape)
        print('labels=', labels.shape)
        print('cumulative=', cumulative)
        # cumulative= [0  235  468  674  912 1097 1292 1529 1745 2010 2247 2482 2715 2950 3188 3394]
        Features = []
        Labels = []

        for i in range(cumulative.shape[0]):
            if i != 0:
                print(i)
                count = (cumulative[i] - cumulative[i - 1]) // T
                print('count=', count)
                for tt in range(count):
                    print(cumulative[i - 1] + tt * T + T)
                    Feature = EEGs[cumulative[i - 1] + tt * T: cumulative[i - 1] + tt * T + T, :, :, :]
                    print('Feature.shape', Feature.shape)
                    Features.append(Feature)
                    print(np.array(Features).shape)
                    Labels.append(labels[cumulative[i - 1]])

        Features = np.array(Features)
        print('Features=', Features.shape)
        Labels = np.array(Labels)
        print(Labels.shape)

        save_path = '...\\subject' + str(subject + 1) + 'session' + str(session + 1)  # save_path

        # SEED-IV数据集
        # save_path = '...\\subject' + str(subject+1) + 'session' + str(session+1)  # save_path

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        feature_path = os.path.join(save_path, "feature.npy")
        label_path = os.path.join(save_path, "label.npy")
        cumulative_samples_path = os.path.join(save_path, "cumulative.npy")
        print("saving feature.npy and label.npy to folder {}".format(save_path))

        np.save(feature_path, Features)
        np.save(label_path, Labels)
        np.save(cumulative_samples_path, cumulative)


