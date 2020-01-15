# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2020/1/14

import pandas as pd
import numpy as np
import os, sys


def predict_age(df):
    path = 'C:/Users/pc/OneDrive/PLTTECH/Project/00_immune_age_project/'
    formula_lvs = pd.read_csv(path+'Rawdata/formula_LVs.csv')
    subsets_34 = ['Lymphocytes/CD3-/B cells',
                  'Lymphocytes/CD3+/CD4+/CD161+',
                  'Lymphocytes/CD3+/CD8+/CD161+',
                  'Lymphocytes/CD3+/CD4+/Treg/Q1: 163Dy_CD161- , 155Gd_CD45RA+',
                  'Lymphocytes/CD3+/CD4+/Treg/Q4: 163Dy_CD161- , 155Gd_CD45RA-',
                  'Lymphocytes/CD3-/CD3-CD20-',
                  'Lymphocytes/CD3+/CD8+/CD28+',
                  'Lymphocytes/CD3+/CD4+',
                  'Lymphocytes/CD3+/CD4+/CD28+',
                  'Lymphocytes/CD3+/CD8+',
                  'Lymphocytes/CD3+/CD8+/CD85j+',
                  'Lymphocytes/CD3-/B cells/Q2: 145Nd_IgD+ , 153Eu_CD27+',
                  'Lymphocytes/CD3-/B cells/Q3: 145Nd_IgD+ , 153Eu_CD27-',
                  'Lymphocytes/CD3-/B cells/Q1: 145Nd_IgD- , 153Eu_CD27+',
                  'Lymphocytes/CD3-/B cells/Q4: 145Nd_IgD- , 153Eu_CD27-',
                  'Lymphocytes/CD3-/NK cells',
                  'Lymphocytes/NKT',
                  'Lymphocytes/CD3+',
                  'Lymphocytes/CD3+/CD4+/Treg',
                  'Lymphocytes/CD3+/CD4+/Q3: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-',
                  'Lymphocytes/CD3+/CD8+/Q3: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-',
                  'Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+',
                  'Lymphocytes/CD3+/CD8+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+',
                  'Lymphocytes/CD3+/CD4+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-',
                  'Lymphocytes/CD3+/CD8+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-',
                  'Lymphocytes/gd T-cells',
                  'Lymphocytes',
                  'Lymphocytes/CD3-/B cells/CD24+CD38-',
                  'Monocytes',
                  'Lymphocytes/CD3-/B cells/CD24-CD38+',
                  'Lymphocytes/CD3+/CD4+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
                  'Lymphocytes/CD3+/CD8+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
                  'Lymphocytes/CD3-/B cells/CD24+CD38+',
                  'Singlets/Viable',
                  ]
    df = df[df['subset'].isin(subsets_34)]
    df.index = list(df['subset'].values)
    df = df.loc[subsets_34]
    df.index = [i for i in range(df.shape[0])]

    # lv1 = np.sum(np.dot(df['frequency'].values, formula_lvs['LV1'].values))
    # lv2 = np.sum(np.dot(df['frequency'].values, formula_lvs['LV2'].values))
    # lv3 = np.sum(np.dot(df['frequency'].values, formula_lvs['LV3'].values))
    #
    # immune_age = 40.1322 - 0.6259*lv1 + 0.2941*lv2 - 0.0356*lv3

    coefs = [0.47146983, -0.3650005 , -0.24882258, -0.9051412 , -0.31502607,
            0.51286523,  0.2611913 ,  0.55553634, -0.13397269,  0.12581725,
            0.23091759, -0.05879372,  0.20945062, -0.02723427, -0.12342243,
            0.16356419, -0.07602652, -0.05547163, -0.23663755,  0.14209078,
            -0.53045574,  0.0091442 ,  0.37538138,  0.02249695,  0.21533253,
            0.30150205, -0.24412036,  0.87213848,  0.24412037,  6.68411177,
            -0.17373217, -0.06025842,  0.58863103,  0.]
    immune_age = -103.423110567636 + np.sum(np.dot(df['frequency'].values, coefs))

    immune_age = np.abs(immune_age)
    if immune_age <= 15:
        immune_age += 15
    if immune_age >= 100:
        immune_age = 100
    print(immune_age)
    age_df = pd.DataFrame([immune_age])
    ratio34_df = df

    return age_df, ratio34_df