# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/12/30


from FCS import Fcs
from tkinter import filedialog
import pandas as pd
import numpy as np
import os, re, warnings, copy


if __name__ == '__main__':
    ###################################################
    ####    1.Read FCS file and convert to CSV     ####
    ###################################################
    # Choose File Path
    Fpath = filedialog.askdirectory()
    os.makedirs(Fpath+"/WriteFcs/")
    csv_path = Fpath + '/WriteFcs/'

    for filename in [filename for filename in os.listdir(Fpath) if os.path.splitext(filename)[1] == ".fcs"]:
        file = Fpath + '/' + filename
        fcs = Fcs(file)
        pars = fcs.pars
        new_filename = re.sub("-", "", filename)
        new_filename = re.sub("^.+?_", "", new_filename)
        # new_filename = re.sub("^.+?_", "", new_filename)
        new_file = Fpath + "/WriteFcs/" + new_filename + '.csv'
        fcs.write_to(new_file, pars, to='csv')

    path = Fpath + "/WriteFcs"
    all_df = pd.DataFrame()
    for file in os.listdir(path):
        df = pd.read_csv(path+'/'+file)
        all_df = all_df.append(df)
        print("File %s has finished." % file)
    all_df.iloc[:, :-1].to_csv(path + '/' + path.split('/')[-2] + '.csv', index=False)