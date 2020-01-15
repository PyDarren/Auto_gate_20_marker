# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2020/1/15

import pandas as pd
from tkinter import filedialog
import os



if __name__ == '__main__':
    Fpath = filedialog.askdirectory()

    merge_df = pd.DataFrame()

    for info in os.listdir(Fpath):
        df = pd.read_excel(Fpath + '/' + info + '/rename_by_panelTable/Output/real_all.xlsx')
        merge_df = merge_df.append(df)

    merge_df.to_excel('E:/cd/0_Auto_Gate_20_marker/lung_cancer/subsets_95_112.xlsx', index=False)

