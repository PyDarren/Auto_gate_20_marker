# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2019/12/30


from FCS import *
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

    csv_path = Fpath + '/WriteFcs/'
    output_path = Fpath + '/Output/'

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

    for filename in [filename for filename in os.listdir(Fpath) if os.path.splitext(filename)[1] == ".fcs"]:
        file = Fpath + '/' + filename
        fcs = Fcs(file)
        pars = fcs.pars

        save_channel = fcs.stain_channels_index
        save_channel.extend(fcs.preprocess_channels_index)
        # stain_channel_index.extend(add_index)
        pars = [pars[i] for i in range(0, len(pars)) if i + 1 in save_channel]

        pars = fcs.delete_channel(pars, 139, 162, 168, 169, 173)

        # pars = fcs.marker_rename(pars, ("Event_length", "Event_length"))

        # 根据当前的filename去查找新的name
        new_filename = re.sub("-", "", filename)
        # new_filename = re.sub("^.+?_", "gsH_", new_filename)
        new_file = Fpath + "WriteFcs/" + new_filename

        fcs.write_to(new_file, pars)

