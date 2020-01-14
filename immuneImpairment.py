# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2020/1/14

import pandas as pd


def immuneImpairmentMatrix(df, sample_id):
    '''
    提取免疫损伤原始矩阵
    :param df:
    :return:
    '''
    subsets_18 = [
        'Lymphocytes/CD3-/B cells',
        'Lymphocytes/CD3+/CD4+/Treg/Q1: 163Dy_CD161- , 155Gd_CD45RA+',
        'Lymphocytes/CD3-/NK cells/CD161+',
        'Lymphocytes/CD3+/CD8+/CD28-',
        'Lymphocytes/CD3+/CD8+/CD57+',
        'Lymphocytes/CD3-/NK cells/CD57+',
        'Lymphocytes/CD3+/CD8+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+',
        'Lymphocytes/CD3+/CD4+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-',
        'Lymphocytes/CD3+/CD8+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-',
        'Lymphocytes/CD3+/CD4+/Q5: 176Yb_HLA_DR- , 172Yb_CD38+',
        'Lymphocytes/CD3+/CD4+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
        'Lymphocytes/CD3+/CD8+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+',
        'Lymphocytes/CD3+/CD8+/PD1+',
        'Lymphocytes/CD3+',
        'Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5+',
        'Lymphocytes/CD3+/CD8+/CXCR5+',
        'Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/Th17 CXCR3-CCR6+',
        'Lymphocytes/CD3+/CD4+/Treg'
    ]
    df = df[df['subset'].isin(subsets_18)]
    df.index = list(df['subset'].values)
    df = df.loc[subsets_18]
    values = list(df['frequency'].values)
    values.extend([sample_id, 2019, 100, 'x'])
    impair_df = pd.DataFrame(values).T
    impair_df.columns = [
        'B.cells',
        'CD161negCD45RApos.Tregs',
        'CD161pos.NK.cells',
        'CD28negCD8pos.T.cells',
        'CD57posCD8pos.T.cells',
        'CD57pos.NK.cells',
        'effector.CD8pos.T.cells',
        'effector.memory.CD4pos.T.cells',
        'effector.memory.CD8pos.T.cells',
        'HLADRnegCD38posCD4pos.T.cells',
        'naive.CD4pos.T.cells',
        'naive.CD8pos.T.cells',
        'PD1posCD8pos.T.cells',
        'T.cells',
        'CXCR5+CD4pos.T.cells',
        'CXCR5+CD8pos.T.cells',
        'Th17 CXCR5-CD4pos.T.cells',
        'Tregs',
        'subject id',
        'year',
        'visit number',
        'age'
    ]
    return impair_df

