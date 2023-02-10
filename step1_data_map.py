import scipy.io as sio
import numpy as np
import os
from sklearn import preprocessing

# SEED数据集
dir_path = "...\SEED\ExtractedFeatures"

def projection_2D(x):
    matrix_3D = np.zeros([5, 9, 9])
    for spectral in range(x.shape[1]):
        stdScale1 = preprocessing.StandardScaler()
        x[:,spectral] = np.squeeze(stdScale1.fit_transform(x[:,spectral].reshape(-1,1)))

        matrix_3D[spectral, 0, 3] = x[0,spectral]
        matrix_3D[spectral, 0, 4] = x[1,spectral]
        matrix_3D[spectral, 0, 5] = x[2,spectral]
        matrix_3D[spectral, 1, 3] = x[3, spectral]
        matrix_3D[spectral, 1, 5] = x[4, spectral]

        for i in range(9):
            matrix_3D[spectral, 2, i] = x[5+i, spectral]
        for i in range(9):
            matrix_3D[spectral, 3, i] = x[5+9+i, spectral]
        for i in range(9):
            matrix_3D[spectral, 4, i] = x[5+9+9+i, spectral]
        for i in range(9):
            matrix_3D[spectral, 5, i] = x[5+9+9+9+i, spectral]
        for i in range(9):
            matrix_3D[spectral, 6, i] = x[5+9+9+9+9+i, spectral]

        matrix_3D[spectral, 7, 1] = x[50, spectral]
        matrix_3D[spectral, 7, 2] = x[51, spectral]
        matrix_3D[spectral, 7, 3] = x[52, spectral]
        matrix_3D[spectral, 7, 4] = x[53, spectral]
        matrix_3D[spectral, 7, 5] = x[54, spectral]
        matrix_3D[spectral, 7, 6] = x[55, spectral]
        matrix_3D[spectral, 7, 7] = x[56, spectral]

        matrix_3D[spectral, 8, 2] = x[57, spectral]
        matrix_3D[spectral, 8, 3] = x[58, spectral]
        matrix_3D[spectral, 8, 4] = x[59, spectral]
        matrix_3D[spectral, 8, 5] = x[60, spectral]
        matrix_3D[spectral, 8, 6] = x[61, spectral]

    return matrix_3D

feature_types = ["de"]
smooth_method_types = ["LDS"]

# SEED get labels:
label_path = os.path.join(dir_path, "label.mat")
labels = sio.loadmat(label_path)["label"][0]

stacked_arr = None
stacked_label = None
cumulative_samples = [0]

for session in range(3):
    for sub in range(15):
        save_path = "...\\subject"+str(sub+1)+"session"+str(session+1)  # save_path

        # SEED数据集
        if session+1 == 1:
            sub_of_session = ['1_20131027.mat', '2_20140404.mat', '3_20140603.mat', '4_20140621.mat', '5_20140411.mat',
                              '6_20130712.mat', '7_20131027.mat',
                              '8_20140511.mat', '9_20140620.mat', '10_20131130.mat', '11_20140618.mat',
                              '12_20131127.mat', '13_20140527.mat', '14_20140601.mat', '15_20130709.mat']
        elif session+1 == 2:
            sub_of_session = ['1_20131030.mat', '2_20140413.mat', '3_20140611.mat', '4_20140702.mat', '5_20140418.mat',
                              '6_20131016.mat', '7_20131030.mat',
                              '8_20140514.mat', '9_20140627.mat', '10_20131204.mat', '11_20140625.mat',
                              '12_20131201.mat', '13_20140603.mat', '14_20140615.mat', '15_20131016.mat']
        else:
            sub_of_session = ['1_20131107.mat', '2_20140419.mat', '3_20140629.mat', '4_20140705.mat', '5_20140506.mat',
                              '6_20131113.mat', '7_20131106.mat',
                              '8_20140521.mat', '9_20140704.mat', '10_20131211.mat', '11_20140630.mat',
                              '12_20131207.mat', '13_20140610.mat', '14_20140627.mat', '15_20131105.mat']

        # SEED-IV数据集
        # if session+1 == 1:
        #     dir_path = "...\\features\\Extractedfeatures\\1"
        #     sub_of_session = ['1_20160518.mat', '2_20150915.mat', '3_20150919.mat', '4_20151111.mat', '5_20160406.mat',
        #                       '6_20150507.mat', '7_20150715.mat',
        #                       '8_20151103.mat', '9_20151028.mat', '10_20151014.mat', '11_20150916.mat',
        #                       '12_20150725.mat', '13_20151115.mat', '14_20151205.mat', '15_20150508.mat']
        #     label_path = os.path.join(dir_path, "label.mat")
        #     labels = sio.loadmat(label_path)["label"][0]
        #
        # elif session+1 == 2:
        #     dir_path = "...\\features\\Extractedfeatures\\2"
        #     sub_of_session = ['1_20161125.mat', '2_20150920.mat', '3_20151018.mat', '4_20151118.mat', '5_20160413.mat',
        #                       '6_20150511.mat', '7_20150717.mat',
        #                       '8_20151110.mat', '9_20151119.mat', '10_20151021.mat', '11_20150921.mat',
        #                       '12_20150804.mat', '13_20151125.mat', '14_20151208.mat', '15_20150514.mat']
        #     label_path = os.path.join(dir_path, "label.mat")
        #     labels = sio.loadmat(label_path)["session2_label"][0]
        #
        # else:
        #     dir_path = "...\\features\\Extractedfeatures\\3"
        #     sub_of_session = ['1_20161126.mat', '2_20151012.mat', '3_20151101.mat', '4_20151123.mat', '5_20160420.mat',
        #                       '6_20150512.mat', '7_20150721.mat',
        #                       '8_20151117.mat', '9_20151209.mat', '10_20151023.mat', '11_20151011.mat',
        #                       '12_20150807.mat', '13_20161130.mat', '14_20151215.mat', '15_20150527.mat']
        #     label_path = os.path.join(dir_path, "label.mat")
        #     labels = sio.loadmat(label_path)["session3_label"][0]

        num_of_experiment = 15

        folder_name = os.path.join(save_path, "de_LDS")  # save_path

        print("folder name: ", folder_name)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        stacked_arr = None
        stacked_label = None
        cumulative_samples = [0]

        for trial_path in os.listdir(dir_path):
            if trial_path == sub_of_session[sub]:  # trial record for the person
                print(sub_of_session[sub])  # sub-1
                feature2dict = sio.loadmat(os.path.join(dir_path, trial_path))
                for experiment_index in range(num_of_experiment):
                    k = 'de_LDS' + str(experiment_index + 1)
                    print('k=,', k)
                    v = feature2dict[k]
                    print('v=', v.shape)  # v1= (62, 235, 5) for example
                    temp_arr = np.swapaxes(v, 0, 1).reshape(v.shape[1], v.shape[0], v.shape[2])  # v1= (235, 62, 5)
                    print('temp= ', temp_arr.shape)

                    for i in range(temp_arr.shape[0]):
                        spatial_spectral_3D = projection_2D(temp_arr[i, :, :])
                        # print('temp_arr[i, :, :]',temp_arr[i, :, :].shape)    # temp_arr[i, :, :] (62, 5)
                        # print('spatial_spectral_3D', spatial_spectral_3D.shape)  # spatial_spectral_3D (5, 9, 9)
                        spatial_spectral_3D = spatial_spectral_3D.reshape((1, spatial_spectral_3D.shape[0],
                                                                           spatial_spectral_3D.shape[1],
                                                                           spatial_spectral_3D.shape[2]))
                        # print('spatial_spectral_3D', spatial_spectral_3D.shape)  # spatial_spectral_3D (1, 5, 9, 9)
                        if stacked_arr is None:
                            stacked_arr = spatial_spectral_3D.copy()
                        else:
                            stacked_arr = np.vstack((stacked_arr, spatial_spectral_3D))  # vertically stack arrays

                    num_of_samples = temp_arr.shape[0]
                    print('num_of_samples',num_of_samples)
                    cumulative_samples.append(cumulative_samples[-1] + num_of_samples)
                    print('cumulative_samples', np.array(cumulative_samples))
                    temp_labels = np.ones((num_of_samples, 1)) * labels[experiment_index]
                    print('temp_labels', temp_labels.shape)  # (235, 1) for example
                    if stacked_label is None:
                        stacked_label = temp_labels.copy()
                    else:
                        stacked_label = np.vstack((stacked_label, temp_labels))

                print("feature shape:", stacked_arr.shape)
                print("label shape: ", stacked_label)
                cumulative_sample_arr = np.array(cumulative_samples)
                print("cumulative sample shape: ", cumulative_sample_arr)
                feature_path = os.path.join(folder_name, "feature.npy")
                label_path = os.path.join(folder_name, "label.npy")
                cumulative_samples_path = os.path.join(folder_name, "cumulative.npy")
                print("saving feature.npy and label.npy to folder {}".format(folder_name))
                np.save(feature_path, stacked_arr)
                np.save(label_path, stacked_label)
                np.save(cumulative_samples_path, cumulative_sample_arr)
