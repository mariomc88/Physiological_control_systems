import os
import pandas as pd
from scipy import signal
import numpy as np
import os
import pickle

frequency = 60
directory = os.fsencode("/home/mario/PycharmProjects/Physiological_control_systems/Physiological_control_systems/Recordings")
list_participants = ["Part_1", "Part_2", "Part_3", "Part_4", "Part_5"]
base_df = pd.read_csv("Part_1_EC_Trial_1.csv", header=2, usecols=[0])
# base_df.drop(index=[0, 1], axis=0, inplace=True)
#Butterworth filter
lp_filter = signal.butter(4, 10, "lowpass", fs=frequency, output="sos")
dataframes_dict = {}
def CoP_centered(ML, AP):
    ML = np.asarray(ML)
    AP = np.asarray(AP)
    AP_corrected = AP - np.mean(AP)
    ML_corrected = ML - np.mean(ML)
    RD = np.linalg.norm(np.concatenate((ML_corrected, AP_corrected)))
    return RD
for participant in list_participants:
    dataframes_dict["df_"+participant] = base_df.copy()
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if participant in filename:
            participant_df = pd.read_csv(filename, header=2, usecols=[0,1,2])
            #Assign x and y columns and filter using the 4th order Butterworth filter
            x_COP = participant_df["x"]
            x_COP = signal.sosfilt(lp_filter, x_COP)
            y_COP = participant_df["y"]
            y_COP = signal.sosfilt(lp_filter, y_COP)
            RD = CoP_centered(x_COP, y_COP)
            dataframes_dict["df_" + participant].insert(1, "x_COP"+filename.strip(participant).strip(".csv"), x_COP)
            dataframes_dict["df_" + participant].insert(1, "y_COP"+filename.strip(participant).strip(".csv"), y_COP)
            dataframes_dict["df_" + participant].insert(1, "RD_COP"+filename.strip(participant).strip(".csv"), RD)
    df = dataframes_dict["df_" + participant]
    print(df.head())
    #Select time >5s and <25s
    df = df[(df["time"] >= 5) & (df["time"] <= 25)]
    dataframes_dict["df_" + participant] = df

with open('Preprocessed_studies_dict', 'wb') as f:
    pickle.dump(dataframes_dict, f)