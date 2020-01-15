# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2020/1/15

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso


def predict_age(vals):
    path = 'C:/Users/pc/OneDrive/PLTTECH/Project/00_immune_age_project/'
    formula_lvs = pd.read_csv(path+'Rawdata/formula_LVs.csv')
    lv1 = np.sum(np.dot(vals, formula_lvs['LV1'].values))
    lv2 = np.sum(np.dot(vals, formula_lvs['LV2'].values))
    lv3 = np.sum(np.dot(vals, formula_lvs['LV3'].values))
    return lv1, lv2, lv3


if __name__ == '__main__':
    rawdata = pd.read_excel('C:/Users/pc/OneDrive/PLTTECH/Project/03_immuneAgeModel/rawdata/immune_age_raw.xlsx')
    # lv1_all = list()
    #     # lv2_all = list()
    #     # lv3_all = list()
    #     #
    #     # for i in range(rawdata.shape[0]):
    #     #     subsets = rawdata.iloc[i, 2:].values
    #     #     lv1, lv2, lv3 = predict_age(subsets)
    #     #     lv1_all.append(lv1)
    #     #     lv2_all.append(lv2)
    #     #     lv3_all.append(lv3)
    #     #
    #     # lv_df = pd.DataFrame(lv1_all, columns=['lv1'])
    #     # lv_df['lv2'] = lv2_all
    #     # lv_df['lv3'] = lv3_all
    #     # lv_df['id'] = rawdata['id']
    #     # lv_df['age'] = rawdata['age']
    #     #
    #     # X = lv_df.loc[:, ['lv1', 'lv2', 'lv3']].values
    # y = lv_df['age'].values

    X = rawdata.iloc[:, 2:].values
    y = rawdata['age'].values
    lr = Ridge(alpha=0.5)
    # lr = Lasso(alpha=6)
    lr.fit(X, y)
    lr.score(X, y)             # R^2 = 0.6779115006436693
    lr.coef_
    lr.intercept_

    # lv_df['immune age'] = list(lr.predict(X))
    # lv_df.to_excel('C:/Users/pc/OneDrive/PLTTECH/Project/03_immuneAgeModel/rawdata/lv_df.xlsx')

    pre_df = pd.DataFrame(list(lr.predict(X)), columns=['immune age'])
    pre_df['age'] = rawdata['age']
    pre_df.to_excel('C:/Users/pc/OneDrive/PLTTECH/Project/03_immuneAgeModel/rawdata/pre_df.xlsx')





