# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2020/1/6


from FCS import *
from tkinter import filedialog
import pandas as pd
import numpy as np
import tensorflow as tf
import os, re, warnings, copy


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
    sub_df = df[df['class']==0]
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

    # Read Panel Information
    panel_file = Fpath + "/panel.xlsx"
    panel_tuple = Fcs.export_panel_tuple(panel_file)
    print(panel_tuple)
    # 根据panel表，对每个fcs重命名
    file_tuple = tuple([Fpath + '/' + filename for filename in os.listdir(Fpath)
                        if os.path.splitext(filename)[1] == ".fcs"])
    rename_by_panel_table(panel_tuple, *file_tuple)

    Fpath = Fpath+"/rename_by_panelTable/"
    os.makedirs(Fpath+"/WriteFcs/")
    os.makedirs(Fpath+"/Output/")

    csv_path = Fpath + '/WriteFcs/'
    output_path = Fpath + '/Output/'

    for filename in [filename for filename in os.listdir(Fpath) if os.path.splitext(filename)[1] == ".fcs"]:
        file = Fpath + '/' + filename
        fcs = Fcs(file)
        pars = fcs.pars

        save_channel = fcs.stain_channels_index
        save_channel.extend(fcs.preprocess_channels_index)
        # stain_channel_index.extend(add_index)
        pars = [pars[i] for i in range(0, len(pars)) if i + 1 in save_channel]

        pars = fcs.delete_channel(pars, 89, 169, 173)

        # pars = fcs.marker_rename(pars, ("Event_length", "Event_length"))

        # 根据当前的filename去查找新的name
        new_filename = re.sub("-", "", filename)
        # new_filename = re.sub("^.+?_", "gsH_", new_filename)
        new_file = Fpath + "WriteFcs/" + new_filename

        fcs.write_to(new_file, pars, to='csv')


    ##################################################
    ####            Load Models
    markers = ('CD57', 'CD3', 'CD56', 'gdTCR', 'CCR6', 'CD14 ',
               'IgD', 'CD123(IL-3R)', 'CD85J', 'CD19', 'CD25', 'CD274(PD-L1)',
               'CD278(ICOS)', 'CD39', 'CD27', 'CD24', 'CD45RA', 'CD86', 'CD28',
               'CD197(CCR7)', 'CD11c ', 'CD33', 'CD152(CTLA-4)', 'CD161', 'CXCR5',
               'CD183(CXCR3)', 'CD94', 'CD127(IL-7Ra)', 'CD279(PD-1)', 'CD38',
               'CD20', 'CD16', 'HLA-DR', 'CD4', 'CD8a', 'CD11b')



    new_samples_path = Fpath + 'WriteFcs/'

    CD3_Pos_list = list()
    CD4_Pos_list = list()
    CD57_Pos_list = list()
    CD56_Pos_list = list()
    gdTCR_Pos_list = list()
    CD8_Pos_list = list()

    for info in os.listdir(new_samples_path):
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

        sample_df = pd.read_csv(new_samples_path + info)
        sample_df = sample_df.loc[:, markers]
        # CD3
        new_df = copy.deepcopy(sample_df)
        ratio_CD3_all, CD3_df, CD3_labels = ratioCalculation2(new_df, model_CD3)
        CD3_Pos_list.append(ratio_CD3_all[0])
        # CD4
        new_df = copy.deepcopy(sample_df)
        ratio_CD4_all, CD4_df, CD4_labels = ratioCalculation2(new_df, model_CD4)
        CD4_Pos_list.append(ratio_CD4_all[0])
        # CD57
        new_df = copy.deepcopy(sample_df)
        ratio_CD57_all, CD57_df, CD57_labels = ratioCalculation2(new_df, model_CD57)
        CD57_Pos_list.append(ratio_CD57_all[0])
        # CD56
        new_df = copy.deepcopy(sample_df)
        ratio_CD56_all, CD56_df, CD56_labels = ratioCalculation2(new_df, model_CD56)
        CD56_Pos_list.append(ratio_CD56_all[0])
        # gdTCR
        new_df = copy.deepcopy(sample_df)
        ratio_gdTCR_all, gdTCR_df, gdTCR_labels = ratioCalculation2(new_df, model_gdTCR)
        gdTCR_Pos_list.append(ratio_gdTCR_all[0])
        # CD8
        new_df = copy.deepcopy(sample_df)
        ratio_CD8_all, CD8_df, CD8_labels = ratioCalculation2(new_df, model_CD8)
        CD8_Pos_list.append(ratio_CD8_all[0])




    pre_df = pd.DataFrame(CD3_Pos_list, columns=['CD3_Auto'])
    pre_df['CD4_Auto'] = CD4_Pos_list
    pre_df['CD57_Auto'] = CD57_Pos_list
    pre_df['CD56_Auto'] = CD56_Pos_list
    pre_df['gdTCR_Auto'] = gdTCR_Pos_list
    pre_df['CD8_Auto'] = CD8_Pos_list
    pre_df['id'] = [i[:11] for i in os.listdir(new_samples_path)]
    pre_df = pre_df.reindex(columns=['id', 'CD3_Auto', 'CD4_Auto', 'CD57_Auto', 'CD56_Auto', 'gdTCR_Auto', 'CD8_Auto'])
    pre_df.to_excel(output_path+'pre_df.xlsx', index=False)