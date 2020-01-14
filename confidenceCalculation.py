# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2020/1/14


import pandas as pd
import numpy as np


def func(subset, names, raw_confidence):
    vals = list(subset.values)
    names = list(names.values)
    arrows_95 = list()
    for i in range(len(vals)):
        name = names[i]
        reference = raw_confidence[raw_confidence['subset']==name]
        if vals[i] > reference['upper_95'].values[0]:
            arrows_95.append('↑')
        elif vals[i] < reference['low_95'].values[0]:
            arrows_95.append('↓')
        else:
            arrows_95.append(' ')
    sample_df = pd.DataFrame([names, arrows_95]).T
    sample_df.columns = ['subset', 'confidence_95']
    return sample_df



def confidence_calculation(df):
    path = 'C:/Users/pc/OneDrive/PLTTECH/Project/00_immune_age_project/'
    raw_confidence = pd.read_excel(path+'Rawdata/confidence_output.xlsx').iloc[:, :4]
    select_subsets_df = pd.read_excel(path+'Rawdata/置信区间选择.xlsx')
    select_subsets = list(select_subsets_df['subset'].values)
    raw_confidence = raw_confidence[raw_confidence['subset'].isin(select_subsets)]
    df = df[df['subset'].isin(select_subsets)]
    subset = df['frequency']
    names = df['subset']
    sample_df = func(subset, names, raw_confidence)
    return sample_df