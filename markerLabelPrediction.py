# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2020/1/10

import pandas as pd
import numpy as np
import tensorflow as tf
import os, sys, warnings, time, copy

warnings.filterwarnings(action='ignore')


def ratioCalculation2(df, model):
    '''
    计算二分类模型的亚群比率
    :param df:
    :return:
    '''
    test = df.values
    predictions = model.predict(test)
    pre_labels = [np.argmax(predictions[i]) for i in range(predictions.shape[0])]
    pre_0_length = len([i for i in pre_labels if i == 0])
    pre_1_length = len([i for i in pre_labels if i == 1])
    length_df = df.shape[0]
    df['class'] = pre_labels
    sub_df = df[df['class'] == 0]
    ratio_0 = pre_0_length / length_df * 100
    ratio_1 = pre_1_length / length_df * 100
    ratio_list = [ratio_0, ratio_1]
    return ratio_list, sub_df, pre_labels


# def ratioCalculation3(df, model):
#     '''
#     计算三分类模型的亚群比率
#     :param df:
#     :return:
#     '''
#     test = df.values
#     predictions = model.predict(test)
#     pre_labels = [np.argmax(predictions[i]) for i in range(predictions.shape[0])]
#     pre_0_length = len([i for i in pre_labels if i == 0])
#     pre_1_length = len([i for i in pre_labels if i == 1])
#     pre_2_length = len([i for i in pre_labels if i == 2])
#     length_df = df.shape[0]
#     df['class'] = pre_labels
#     sub_df = df[df['class'] == 0]
#     ratio_0 = pre_0_length / length_df * 100
#     ratio_1 = pre_1_length / length_df * 100
#     ratio_2 = pre_2_length / length_df * 100
#     ratio_list = [ratio_0, ratio_1, ratio_2]
#     return ratio_list, sub_df, pre_labels


def markerRatioCalculation(sample_df):
    '''
    计算特定marker的标签矩阵
    :param df: 单个样本的CSV文件
    :return: 特定marker的标签矩阵
    '''
    #### Load Models
    input_shape = (None, 36)
    model_CD3 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD3_classfy.h5')
    model_CD3.build(input_shape)
    model_CD4 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD4_classfy.h5')
    model_CD4.build(input_shape)
    model_CD57 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD57_classfy.h5')
    model_CD57.build(input_shape)
    model_CD56 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD56_classfy.h5')
    model_CD56.build(input_shape)
    model_gdTCR = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/gdTCR_classfy.h5')
    model_gdTCR.build(input_shape)
    model_CD8 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD8_classfy.h5')
    model_CD8.build(input_shape)
    model_CD14 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD14_classfy.h5')
    model_CD14.build(input_shape)
    model_Igd = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/Igd_classfy.h5')
    model_Igd.build(input_shape)
    model_CD123 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD123_classfy.h5')
    model_CD123.build(input_shape)
    model_CD85j = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD85j_classfy.h5')
    model_CD85j.build(input_shape)
    model_CD19 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD19_classfy.h5')
    model_CD19.build(input_shape)
    model_CD25 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD25_classfy.h5')
    model_CD25.build(input_shape)
    model_CD39 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD39_classfy.h5')
    model_CD39.build(input_shape)
    model_CD27 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD27_classfy.h5')
    model_CD27.build(input_shape)
    model_CD24 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD24_classfy.h5')
    model_CD24.build(input_shape)
    model_CD45RA = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD45RA_classfy.h5')
    model_CD45RA.build(input_shape)
    model_CD86 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD86_classfy.h5')
    model_CD86.build(input_shape)
    model_CD28 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD28_classfy.h5')
    model_CD28.build(input_shape)
    model_CD197 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD197_classfy.h5')
    model_CD197.build(input_shape)
    model_CD11c = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD11c_classfy.h5')
    model_CD11c.build(input_shape)
    model_CD33 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD33_classfy.h5')
    model_CD33.build(input_shape)
    model_CD152 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD152_classfy.h5')
    model_CD152.build(input_shape)
    model_CD161 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD161_classfy.h5')
    model_CD161.build(input_shape)
    model_CXCR5 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CXCR5_classfy.h5')
    model_CXCR5.build(input_shape)
    model_CD183 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD183_classfy.h5')
    model_CD183.build(input_shape)
    model_CD94 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD94_classfy.h5')
    model_CD94.build(input_shape)
    model_CD127 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD127_classfy.h5')
    model_CD127.build(input_shape)
    model_PD1 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/PD1_classfy.h5')
    model_PD1.build(input_shape)
    model_CD20 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD20_classfy.h5')
    model_CD20.build(input_shape)
    model_CD16 = tf.keras.models.load_model('C:/Users/pc/OneDrive/git_repo/Auto_gate_20_marker/Models/CD16_classfy.h5')
    model_CD16.build(input_shape)

    label_df = pd.DataFrame()

    start = time.time()

    # CD3
    new_df = copy.deepcopy(sample_df)
    ratio_CD3_all, CD3_df, CD3_labels = ratioCalculation2(new_df, model_CD3)
    label_df = label_df.append(pd.DataFrame(CD3_labels).T)
    print('Marker %s has finished!' % 'CD3')
    # CD4
    new_df = copy.deepcopy(sample_df)
    ratio_CD4_all, CD4_df, CD4_labels = ratioCalculation2(new_df, model_CD4)
    label_df = label_df.append(pd.DataFrame(CD4_labels).T)
    print('Marker %s has finished!' % 'CD4')
    # CD57
    new_df = copy.deepcopy(sample_df)
    ratio_CD57_all, CD57_df, CD57_labels = ratioCalculation2(new_df, model_CD57)
    label_df = label_df.append(pd.DataFrame(CD57_labels).T)
    print('Marker %s has finished!' % 'CD57')
    # CD56
    new_df = copy.deepcopy(sample_df)
    ratio_CD56_all, CD56_df, CD56_labels = ratioCalculation2(new_df, model_CD56)
    label_df = label_df.append(pd.DataFrame(CD56_labels).T)
    print('Marker %s has finished!' % 'CD56')
    # gdTCR
    new_df = copy.deepcopy(sample_df)
    ratio_gdTCR_all, gdTCR_df, gdTCR_labels = ratioCalculation2(new_df, model_gdTCR)
    label_df = label_df.append(pd.DataFrame(gdTCR_labels).T)
    print('Marker %s has finished!' % 'gdTCR')
    # CD8
    new_df = copy.deepcopy(sample_df)
    ratio_CD8_all, CD8_df, CD8_labels = ratioCalculation2(new_df, model_CD8)
    label_df = label_df.append(pd.DataFrame(CD8_labels).T)
    print('Marker %s has finished!' % 'CD8')
    # CD14
    new_df = copy.deepcopy(sample_df)
    ratio_CD14_all, CD14_df, CD14_labels = ratioCalculation2(new_df, model_CD14)
    label_df = label_df.append(pd.DataFrame(CD14_labels).T)
    print('Marker %s has finished!' % 'CD14')
    # Igd
    new_df = copy.deepcopy(sample_df)
    ratio_Igd_all, Igd_df, Igd_labels = ratioCalculation2(new_df, model_Igd)
    label_df = label_df.append(pd.DataFrame(Igd_labels).T)
    print('Marker %s has finished!' % 'Igd')
    # CD123
    new_df = copy.deepcopy(sample_df)
    ratio_CD123_all, CD123_df, CD123_labels = ratioCalculation2(new_df, model_CD123)
    label_df = label_df.append(pd.DataFrame(CD123_labels).T)
    print('Marker %s has finished!' % 'CD123')
    # CD85j
    new_df = copy.deepcopy(sample_df)
    ratio_CD85j_all, CD85j_df, CD85j_labels = ratioCalculation2(new_df, model_CD85j)
    label_df = label_df.append(pd.DataFrame(CD85j_labels).T)
    print('Marker %s has finished!' % 'CD85j')
    # CD19
    new_df = copy.deepcopy(sample_df)
    ratio_CD19_all, CD19_df, CD19_labels = ratioCalculation2(new_df, model_CD19)
    label_df = label_df.append(pd.DataFrame(CD19_labels).T)
    print('Marker %s has finished!' % 'CD19')
    # CD25
    new_df = copy.deepcopy(sample_df)
    ratio_CD25_all, CD25_df, CD25_labels = ratioCalculation2(new_df, model_CD25)
    label_df = label_df.append(pd.DataFrame(CD25_labels).T)
    print('Marker %s has finished!' % 'CD25')
    # CD39
    new_df = copy.deepcopy(sample_df)
    ratio_CD39_all, CD39_df, CD39_labels = ratioCalculation2(new_df, model_CD39)
    label_df = label_df.append(pd.DataFrame(CD39_labels).T)
    print('Marker %s has finished!' % 'CD39')
    # CD27
    new_df = copy.deepcopy(sample_df)
    ratio_CD27_all, CD27_df, CD27_labels = ratioCalculation2(new_df, model_CD27)
    label_df = label_df.append(pd.DataFrame(CD27_labels).T)
    print('Marker %s has finished!' % 'CD27')
    # CD24
    new_df = copy.deepcopy(sample_df)
    ratio_CD24_all, CD24_df, CD24_labels = ratioCalculation2(new_df, model_CD24)
    label_df = label_df.append(pd.DataFrame(CD24_labels).T)
    print('Marker %s has finished!' % 'CD24')
    # CD45RA
    new_df = copy.deepcopy(sample_df)
    ratio_CD45RA_all, CD45RA_df, CD45RA_labels = ratioCalculation2(new_df, model_CD45RA)
    label_df = label_df.append(pd.DataFrame(CD45RA_labels).T)
    print('Marker %s has finished!' % 'CD45RA')
    # CD86
    new_df = copy.deepcopy(sample_df)
    ratio_CD86_all, CD86_df, CD86_labels = ratioCalculation2(new_df, model_CD86)
    label_df = label_df.append(pd.DataFrame(CD86_labels).T)
    print('Marker %s has finished!' % 'CD86')
    # CD28
    new_df = copy.deepcopy(sample_df)
    ratio_CD28_all, CD28_df, CD28_labels = ratioCalculation2(new_df, model_CD28)
    label_df = label_df.append(pd.DataFrame(CD28_labels).T)
    print('Marker %s has finished!' % 'CD28')
    # CD197
    new_df = copy.deepcopy(sample_df)
    ratio_CD197_all, CD197_df, CD197_labels = ratioCalculation2(new_df, model_CD197)
    label_df = label_df.append(pd.DataFrame(CD197_labels).T)
    print('Marker %s has finished!' % 'CD197')
    # CD11c
    new_df = copy.deepcopy(sample_df)
    ratio_CD11c_all, CD11c_df, CD11c_labels = ratioCalculation2(new_df, model_CD11c)
    label_df = label_df.append(pd.DataFrame(CD11c_labels).T)
    print('Marker %s has finished!' % 'CD11c')
    # CD33
    new_df = copy.deepcopy(sample_df)
    ratio_CD33_all, CD33_df, CD33_labels = ratioCalculation2(new_df, model_CD33)
    label_df = label_df.append(pd.DataFrame(CD33_labels).T)
    print('Marker %s has finished!' % 'CD33')
    # CD152
    new_df = copy.deepcopy(sample_df)
    ratio_CD152_all, CD152_df, CD152_labels = ratioCalculation2(new_df, model_CD152)
    label_df = label_df.append(pd.DataFrame(CD152_labels).T)
    print('Marker %s has finished!' % 'CD152')
    # CD161
    new_df = copy.deepcopy(sample_df)
    ratio_CD161_all, CD161_df, CD161_labels = ratioCalculation2(new_df, model_CD161)
    label_df = label_df.append(pd.DataFrame(CD161_labels).T)
    print('Marker %s has finished!' % 'CD161')
    # CXCR5
    new_df = copy.deepcopy(sample_df)
    ratio_CXCR5_all, CXCR5_df, CXCR5_labels = ratioCalculation2(new_df, model_CXCR5)
    label_df = label_df.append(pd.DataFrame(CXCR5_labels).T)
    print('Marker %s has finished!' % 'CXCR5')
    # CD183
    new_df = copy.deepcopy(sample_df)
    ratio_CD183_all, CD183_df, CD183_labels = ratioCalculation2(new_df, model_CD183)
    label_df = label_df.append(pd.DataFrame(CD183_labels).T)
    print('Marker %s has finished!' % 'CD183')
    # CD94
    new_df = copy.deepcopy(sample_df)
    ratio_CD94_all, CD94_df, CD94_labels = ratioCalculation2(new_df, model_CD94)
    label_df = label_df.append(pd.DataFrame(CD94_labels).T)
    print('Marker %s has finished!' % 'CD94')
    # CD127
    new_df = copy.deepcopy(sample_df)
    ratio_CD127_all, CD127_df, CD127_labels = ratioCalculation2(new_df, model_CD127)
    label_df = label_df.append(pd.DataFrame(CD127_labels).T)
    print('Marker %s has finished!' % 'CD127')
    # PD1
    new_df = copy.deepcopy(sample_df)
    ratio_PD1_all, PD1_df, PD1_labels = ratioCalculation2(new_df, model_PD1)
    label_df = label_df.append(pd.DataFrame(PD1_labels).T)
    print('Marker %s has finished!' % 'PD1')
    # CD20
    new_df = copy.deepcopy(sample_df)
    ratio_CD20_all, CD20_df, CD20_labels = ratioCalculation2(new_df, model_CD20)
    label_df = label_df.append(pd.DataFrame(CD20_labels).T)
    print('Marker %s has finished!' % 'CD20')
    # CD16
    new_df = copy.deepcopy(sample_df)
    ratio_CD16_all, CD16_df, CD16_labels = ratioCalculation2(new_df, model_CD16)
    label_df = label_df.append(pd.DataFrame(CD16_labels).T)
    print('Marker %s has finished!' % 'CD16')

    print('Label prediction has finished!', '\n', '\n')
    # print('Now start to write out the data. This process is time consuming. Please be patient.^_^')

    # 细胞类标
    label_df = label_df.T
    label_df.columns = ['CD3', 'CD4', 'CD57', 'CD56', 'gdTCR', 'CD8',
                        'CD14', 'Igd', 'CD123', 'CD85j', 'CD19', 'CD25',
                        'CD39', 'CD27', 'CD24', 'CD54RA', 'CD86', 'CD28',
                        'CD197', 'CD11c', 'CD33', 'CDC152', 'CD161', 'CXCR5',
                        'CD183', 'CD94', 'CD127', 'PD1', 'CD20', 'CD16']
    print('Time cost is %s.' % (time.time() - start))
    print('-' * 100)
    return label_df
