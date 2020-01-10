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
    CD14_Pos_list = list()
    Igd_Pos_list = list()
    CD123_Pos_list = list()
    CD85j_Pos_list = list()
    CD19_Pos_list = list()
    CD25_Pos_list = list()
    CD39_Pos_list = list()
    CD27_Pos_list = list()
    CD24_Pos_list = list()
    CD45RA_Pos_list = list()
    CD86_Pos_list = list()
    CD28_Pos_list = list()
    CD197_Pos_list = list()
    CD11c_Pos_list = list()
    CD33_Pos_list = list()
    CD152_Pos_list = list()
    CD161_Pos_list = list()
    CXCR5_Pos_list = list()
    CD183_Pos_list = list()
    CD94_Pos_list = list()
    CD127_Pos_list = list()
    PD1_Pos_list = list()
    CD20_Pos_list = list()
    CD16_Pos_list = list()
    

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
        # CD14
        new_df = copy.deepcopy(sample_df)
        ratio_CD14_all, CD14_df, CD14_labels = ratioCalculation2(new_df, model_CD14)
        CD14_Pos_list.append(ratio_CD14_all[0])
        # Igd
        new_df = copy.deepcopy(sample_df)
        ratio_Igd_all, Igd_df, Igd_labels = ratioCalculation2(new_df, model_Igd)
        Igd_Pos_list.append(ratio_Igd_all[0])
        # CD123
        new_df = copy.deepcopy(sample_df)
        ratio_CD123_all, CD123_df, CD123_labels = ratioCalculation2(new_df, model_CD123)
        CD123_Pos_list.append(ratio_CD123_all[0])
        # CD85j
        new_df = copy.deepcopy(sample_df)
        ratio_CD85j_all, CD85j_df, CD85j_labels = ratioCalculation2(new_df, model_CD85j)
        CD85j_Pos_list.append(ratio_CD85j_all[0])
        # CD19
        new_df = copy.deepcopy(sample_df)
        ratio_CD19_all, CD19_df, CD19_labels = ratioCalculation2(new_df, model_CD19)
        CD19_Pos_list.append(ratio_CD19_all[0])
        # CD25
        new_df = copy.deepcopy(sample_df)
        ratio_CD25_all, CD25_df, CD25_labels = ratioCalculation2(new_df, model_CD25)
        CD25_Pos_list.append(ratio_CD25_all[0])
        # CD39
        new_df = copy.deepcopy(sample_df)
        ratio_CD39_all, CD39_df, CD39_labels = ratioCalculation2(new_df, model_CD39)
        CD39_Pos_list.append(ratio_CD39_all[0])
        # CD27
        new_df = copy.deepcopy(sample_df)
        ratio_CD27_all, CD27_df, CD27_labels = ratioCalculation2(new_df, model_CD27)
        CD27_Pos_list.append(ratio_CD27_all[0])
        # CD24
        new_df = copy.deepcopy(sample_df)
        ratio_CD24_all, CD24_df, CD24_labels = ratioCalculation2(new_df, model_CD24)
        CD24_Pos_list.append(ratio_CD24_all[0])
        # CD45RA
        new_df = copy.deepcopy(sample_df)
        ratio_CD45RA_all, CD45RA_df, CD45RA_labels = ratioCalculation2(new_df, model_CD45RA)
        CD45RA_Pos_list.append(ratio_CD45RA_all[0])
        # CD86
        new_df = copy.deepcopy(sample_df)
        ratio_CD86_all, CD86_df, CD86_labels = ratioCalculation2(new_df, model_CD86)
        CD86_Pos_list.append(ratio_CD86_all[0])
        # CD28
        new_df = copy.deepcopy(sample_df)
        ratio_CD28_all, CD28_df, CD28_labels = ratioCalculation2(new_df, model_CD28)
        CD28_Pos_list.append(ratio_CD28_all[0])
        # CD197
        new_df = copy.deepcopy(sample_df)
        ratio_CD197_all, CD197_df, CD197_labels = ratioCalculation2(new_df, model_CD197)
        CD197_Pos_list.append(ratio_CD197_all[0])
        # CD11c
        new_df = copy.deepcopy(sample_df)
        ratio_CD11c_all, CD11c_df, CD11c_labels = ratioCalculation2(new_df, model_CD11c)
        CD11c_Pos_list.append(ratio_CD11c_all[0])
        # CD33
        new_df = copy.deepcopy(sample_df)
        ratio_CD33_all, CD33_df, CD33_labels = ratioCalculation2(new_df, model_CD33)
        CD33_Pos_list.append(ratio_CD33_all[0])
        # CD152
        new_df = copy.deepcopy(sample_df)
        ratio_CD152_all, CD152_df, CD152_labels = ratioCalculation2(new_df, model_CD152)
        CD152_Pos_list.append(ratio_CD152_all[0])
        # CD161
        new_df = copy.deepcopy(sample_df)
        ratio_CD161_all, CD161_df, CD161_labels = ratioCalculation2(new_df, model_CD161)
        CD161_Pos_list.append(ratio_CD161_all[0])
        # CXCR5
        new_df = copy.deepcopy(sample_df)
        ratio_CXCR5_all, CXCR5_df, CXCR5_labels = ratioCalculation2(new_df, model_CXCR5)
        CXCR5_Pos_list.append(ratio_CXCR5_all[0])
        # CD183
        new_df = copy.deepcopy(sample_df)
        ratio_CD183_all, CD183_df, CD183_labels = ratioCalculation2(new_df, model_CD183)
        CD183_Pos_list.append(ratio_CD183_all[0])
        # CD94
        new_df = copy.deepcopy(sample_df)
        ratio_CD94_all, CD94_df, CD94_labels = ratioCalculation2(new_df, model_CD94)
        CD94_Pos_list.append(ratio_CD94_all[0])
        # CD127
        new_df = copy.deepcopy(sample_df)
        ratio_CD127_all, CD127_df, CD127_labels = ratioCalculation2(new_df, model_CD127)
        CD127_Pos_list.append(ratio_CD127_all[0])
        # PD1
        new_df = copy.deepcopy(sample_df)
        ratio_PD1_all, PD1_df, PD1_labels = ratioCalculation2(new_df, model_PD1)
        PD1_Pos_list.append(ratio_PD1_all[0])
        # CD20
        new_df = copy.deepcopy(sample_df)
        ratio_CD20_all, CD20_df, CD20_labels = ratioCalculation2(new_df, model_CD20)
        CD20_Pos_list.append(ratio_CD20_all[0])
        # CD16
        new_df = copy.deepcopy(sample_df)
        ratio_CD16_all, CD16_df, CD16_labels = ratioCalculation2(new_df, model_CD16)
        CD16_Pos_list.append(ratio_CD16_all[0])


    pre_df = pd.DataFrame(CD3_Pos_list, columns=['CD3_Auto'])
    pre_df['CD4_Auto'] = CD4_Pos_list
    pre_df['CD57_Auto'] = CD57_Pos_list
    pre_df['CD56_Auto'] = CD56_Pos_list
    pre_df['gdTCR_Auto'] = gdTCR_Pos_list
    pre_df['CD8_Auto'] = CD8_Pos_list
    pre_df['CD14_Auto'] = CD14_Pos_list
    pre_df['Igd_Auto'] = Igd_Pos_list
    pre_df['CD123_Auto'] = CD123_Pos_list
    pre_df['CD85j_Auto'] = CD85j_Pos_list
    pre_df['CD19_Auto'] = CD19_Pos_list
    pre_df['CD25_Auto'] = CD25_Pos_list
    pre_df['CD39_Auto'] = CD39_Pos_list
    pre_df['CD27_Auto'] = CD27_Pos_list
    pre_df['CD24_Auto'] = CD24_Pos_list
    pre_df['CD45RA_Auto'] = CD45RA_Pos_list
    pre_df['CD86_Auto'] = CD86_Pos_list
    pre_df['CD28_Auto'] = CD28_Pos_list
    pre_df['CD197_Auto'] = CD197_Pos_list
    pre_df['CD11c_Auto'] = CD11c_Pos_list
    pre_df['CD33_Auto'] = CD33_Pos_list
    pre_df['CD152_Auto'] = CD152_Pos_list
    pre_df['CD161_Auto'] = CD161_Pos_list
    pre_df['CXCR5_Auto'] = CXCR5_Pos_list
    pre_df['CD183_Auto'] = CD183_Pos_list
    pre_df['CD94_Auto'] = CD94_Pos_list
    pre_df['CD127_Auto'] = CD127_Pos_list
    pre_df['PD1_Auto'] = PD1_Pos_list
    pre_df['CD20_Auto'] = CD20_Pos_list
    pre_df['CD16_Auto'] = CD16_Pos_list
    pre_df['id'] = [i[:11] for i in os.listdir(new_samples_path)]
    pre_df = pre_df.reindex(columns=['id', 'CD3_Auto', 'CD4_Auto', 'CD57_Auto', 'CD56_Auto', 'gdTCR_Auto', 'CD8_Auto',
                                     'CD14_Auto', 'Igd_Auto', 'CD123_Auto', 'CD85j_Auto', 'CD19_Auto', 'CD25_Auto',
                                     'CD39_Auto', 'CD27_Auto', 'CD24_Auto', 'CD54RA_Auto', 'CD86_Auto', 'CD28_Auto',
                                     'CD197_Auto', 'CD11c_Auto', 'CD33_Auto', 'CDC152_Auto', 'CD161_Auto', 'CXCR5_Auto',
                                     'CD183_Auto', 'CD94_Auto', 'CD127_Auto', 'PD1_Auto', 'CD20_Auto', 'CD16_Auto'])
    pre_df.to_excel(output_path+'pre_df.xlsx', index=False)