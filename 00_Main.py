# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2020/1/10


from FCS import *
from tkinter import filedialog
import pandas as pd
import numpy as np
import tensorflow as tf
import os, re, warnings, copy, time
from markerLabelPrediction import markerRatioCalculation
from immuneAgeSubsets import subsetsRatioCalculation_real
from confidenceCalculation import confidence_calculation


warnings.filterwarnings('ignore')


def normalization(df, feature):
    '''
    1、Calculated the mean and s.d. of frequency values between the 10th and 90th percentiles;
    2、Normalize cellular frequencies by subtraction of the mean and division by s.d.
    :param feature:  The feature to normalized.
    :return:
    '''
    f_list = list(df[feature])
    f_list = [i for i in f_list if i != 0]              # NA values are not computed
    quantile_10 = np.quantile(f_list, 0.1)
    quantile_90 = np.quantile(f_list, 0.9)
    nums_to_calcu = [i for i in f_list if i >= quantile_10 and i <= quantile_90]
    f_mean = np.mean(nums_to_calcu)
    f_std = np.std(nums_to_calcu)
    df[feature] = df[feature].apply(lambda x : (x - f_mean) / f_std)


def scaling(df):
    for subset in df.columns[:-4]:
        normalization(df, subset)
        print('Cell subset "%s" has finished.' % subset)


def confidence_adjuest(df):
    '''
    Corrected the calculation formula of some subsets.
    :param df:
    :return:
    '''
    df.iloc[2,1] = df.iloc[2,1] * df.iloc[1,1] / 100
    df.iloc[39,1] = df.iloc[39,1] * df.iloc[1,1] / 100
    df.iloc[66,1] = df.iloc[66,1] * df.iloc[65,1] / 100
    df.iloc[76,1] = df.iloc[76,1] * df.iloc[65,1] / 100
    return df




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
    print(ratio_list)
    return ratio_list, sub_df, pre_labels


if __name__ == '__main__':
    ###################################################
    ####    1.Read FCS file and convert to CSV     ####
    ###################################################
    # Choose File Path
    Fpath = filedialog.askdirectory()

    start = time.time()

    # Read Panel Information
    panel_file = Fpath + "/panel.xlsx"
    panel_tuple = Fcs.export_panel_tuple(panel_file)
    print(panel_tuple)

    # 根据panel表，对每个fcs重命名
    file_tuple = tuple([Fpath + '/' + filename for filename in os.listdir(Fpath)
                        if os.path.splitext(filename)[1] == ".fcs"])
    rename_by_panel_table(panel_tuple, *file_tuple)

    Fpath = Fpath + "/rename_by_panelTable/"
    os.makedirs(Fpath + "/WriteFcs/")
    os.makedirs(Fpath + "/Output/")

    csv_path = Fpath + '/WriteFcs/'
    output_path = Fpath + '/Output/'

    # 删除通道并转换成CSV文件
    for filename in [filename for filename in os.listdir(Fpath) if os.path.splitext(filename)[1] == ".fcs"]:
        file = Fpath + '/' + filename
        fcs = Fcs(file)
        pars = fcs.pars
        save_channel = fcs.stain_channels_index
        save_channel.extend(fcs.preprocess_channels_index)
        # stain_channel_index.extend(add_index)
        pars = [pars[i] for i in range(0, len(pars)) if i + 1 in save_channel]
        # pars = fcs.delete_channel(pars, 89, 169, 173)                  # IMM panel
        pars = fcs.delete_channel(pars, 139, 162, 168, 169, 173)     # old pannel
        # pars = fcs.marker_rename(pars, ("Event_length", "Event_length"))
        # 根据当前的filename去查找新的name
        new_filename = re.sub("-", "", filename)
        new_filename = re.sub("^.+?_", "", new_filename)
        new_filename = re.sub("^.+?_", "", new_filename)          # 浙一数据打开
        new_file = Fpath + "WriteFcs/" + new_filename
        fcs.write_to(new_file, pars, to='csv')

    ##################################################
    ####       2. Predict Marker Label
    ##################################################
    markers = ('CD57', 'CD3', 'CD56', 'gdTCR', 'CCR6', 'CD14 ',
               'IgD', 'CD123(IL-3R)', 'CD85J', 'CD19', 'CD25', 'CD274(PD-L1)',
               'CD278(ICOS)', 'CD39', 'CD27', 'CD24', 'CD45RA', 'CD86', 'CD28',
               'CD197(CCR7)', 'CD11c ', 'CD33', 'CD152(CTLA-4)', 'CD161', 'CXCR5',
               'CD183(CXCR3)', 'CD94', 'CD127(IL-7Ra)', 'CD279(PD-1)', 'CD38',
               'CD20', 'CD16', 'HLA-DR', 'CD4', 'CD8a', 'CD11b')

    new_samples_path = Fpath + 'WriteFcs/'

    label_frequency_all = pd.DataFrame()
    ratio_all = pd.DataFrame()
    real_all = pd.DataFrame()
    real_adjust_all = pd.DataFrame()
    confidence_all = pd.DataFrame()
    immune_age_all = pd.DataFrame()
    ratio34_all = pd.DataFrame()
    impair_all = pd.DataFrame()
    lung_cancer_all = pd.DataFrame()
    liver_cancer_all = pd.DataFrame()
    colorectal_cancer_all = pd.DataFrame()

    for info in os.listdir(new_samples_path):
        sample_id = info[:11]
        sample_path = output_path+sample_id
        os.makedirs(sample_path)

        # 导入单个样本数据
        sample_df = pd.read_csv(new_samples_path+info)
        sample_df = sample_df.loc[:, markers]

        # 1. Calculate the label matrix for each marker
        pre_labels = markerRatioCalculation(sample_df)
        pre_labels.to_csv(sample_path+'/pre_labels.csv', index=False)
        label_frequency = np.sum(pre_labels)/pre_labels.shape[0]
        label_frequency_df = pd.DataFrame(label_frequency).T
        label_frequency_df.index = [sample_id]
        label_frequency_df.to_csv(sample_path+'/label_frequency.csv')
        label_frequency_all = label_frequency_all.append(label_frequency_df)
        print('Marker ratio calculation has finished!', '\n', '-'*100, '\n')

        # 2. Calculate the ratio of a specific subgroup
        real_merge_df = subsetsRatioCalculation_real(pre_labels)
        real_df = real_merge_df.T
        real_df['subset'] = list(real_df.index)
        real_df.columns = ['frequency', 'subset']
        real_df.index = [i for i in range(real_df.shape[0])]
        real_df = real_df[['subset', 'frequency']]
        real_df['frequency'] = real_df['frequency'].apply(lambda x: x*100)
        real_df.to_excel(sample_path+'/real_df.xlsx', index=False)
        real_merge_df.index = [sample_id]
        real_all = real_all.append(real_merge_df)
        print('Subset ratio calculation has finished!', '\n', '-'*100, '\n')

        # 3. Calculate the relative value of a confidence interval
        real_df_copy = copy.deepcopy(real_df)
        real_df_adjust = confidence_adjuest(real_df_copy)
        confidence_df = confidence_calculation(real_df_adjust)
        confidence_df.to_excel(sample_path+'/confidence.xlsx', index=False)
        confidence_merge_df = confidence_df.T
        confidence_merge_df.columns = list(confidence_df['subset'].values)
        confidence_merge_df = confidence_merge_df.iloc[1:, :]
        confidence_merge_df.index = [sample_id]
        confidence_all = confidence_all.append(confidence_merge_df)
        print('Confidence calculation has finished!', '\n', '-'*100, '\n')
        adjust_df = pd.DataFrame(real_df_adjust['frequency'].values).T
        adjust_df.columns = list(real_df_adjust['subset'].values)
        adjust_df.index = [sample_id]
        real_adjust_all = real_adjust_all.append(adjust_df)

    print('Total time is %s.' % (time.time()-start))

    label_frequency_all.to_excel(output_path+'label_frequency_all.xlsx')
    real_all.to_excel(output_path+'real_all.xlsx')
    confidence_all.to_excel(output_path+'confidence_all.xlsx')


    # Extract the confidence interval 66 subgroup ratios
    select_subsets_df = pd.read_excel('C:/Users/pc/OneDrive/PLTTECH/Project/00_immune_age_project/Rawdata/置信区间选择.xlsx')
    select_subsets = list(select_subsets_df['subset'].values)
    confidence_66_ratio = real_adjust_all.loc[:, select_subsets].T
    confidence_66_ratio.to_excel(output_path+'confidence_66_ratio.xlsx')