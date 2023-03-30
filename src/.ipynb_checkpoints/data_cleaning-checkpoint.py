import pandas as pd
import numpy as np
import os
import json
import sys
# sys.path.insert(0,'..')
RESULTS_DIR = "../results/"
DATA_DIR = "../data/"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)   

def weight_to_int(row):
    try:
        return int(row)
    except:
        return int(row.split(" ")[0])


def height_to_inches(row):
    if type(row) == float:
        # 6.10 -> 72 inches
        feet, inches = str(row).split(".")
        return int(feet) * 12 + int(inches)

    if type(row) == str:
        # 6' 10" -> 72 inches
        feet, inches = str(row).split(" ")
        feet = feet.replace("'", '')
        inches = inches.replace('"', '')
        return int(feet) * 12 + int(inches)

    if type(row) == int:
        return row * 12

def clean_raw_data(filename=f"{DATA_DIR}Brdi_db_march.xlsx"):
    print(os.getcwd())
    players_df = pd.read_excel(filename, engine="openpyxl").drop(columns=[123, "id", "Data Initials", "Code Name", "draft status", ])

    # if no prev concussions "# of concussions" = 0
    players_df.loc[players_df["previous concussions?"] == "NO", '# of concussions'] = 0

    # "previous concussions?" YES/NO -> 0/1
    players_df["previous concussions?"] = players_df["previous concussions?"].apply(lambda x: 1 if x=="YES" else 0)

    # weight -> int
    players_df["weight"] = players_df["weight"].apply(weight_to_int)

    # height -> inches as int
    players_df["height"] = players_df["height"].apply(height_to_inches)

    # draft year -> int *not drafted == -1*
    players_df["draft year"] = players_df["draft year"].apply(lambda x: int(x) if pd.notnull(x) and x != 0 else -1)

    # draft number -> int *not drafted == -1*
    players_df["draft number"] = players_df["draft number"].apply(lambda x: int(x) if pd.notnull(x) and x != 0 else -1)

    # create drafted row
    players_df["drafted"] = players_df["draft number"].apply(lambda x: 0 if x == -1 else 1)
    column_to_move = players_df.pop("drafted")

    # players_df = players_df.drop(columns=["DR Errors: V", "DR Errors: HR"])

    # sumona_features = ['dob','weight','nConc','washer_score','RT_V','RT_HR','Delta_RT','MT_V','MT_HR','Delta_MT','TMT_V','TMT_HR','CMT_V','CMT_HR','cvRT_V','cvRT_HR','stdRT_V','stdRT_HR','Ball_Path_V','Ball_Path_HR','Delta_BallPath','FullPath_V','FullPath_HR','Delta_Fullpath','Corrective_V','Corrective_HR','PeakV_V','PeakV_HR','Delta_PV','AE_V','AE_HR','Delta_AE','VE_V','VE_HR','AbsOnAxis_HR','Delta_OnAxis','AbsOffAxis_V','AbsOffAxis_HR','Delta_OffAxis',] 
    sumona_features = ['age as of June 1','# of concussions','previous concussions?',"NHL",'weight','bimanual score: washer','RT_V','RT_HR','Delta_RT','MT_V','MT_HR','Delta_MT','TMT_V','TMT_HR','CMT: V','CMT: HR','cvRT_V','cvRT_HR','stdRT_V','stdRT_HR','Ball Path_V','Ball Path_HR','FullPath_V','FullPath_HR','Delta_Fullpath','Corrective_V','Corrective_HR','PeakV_V','PeakV_HR','Delta_PV','AE_V','AE_HR','Delta_AE','VE_V','VE_HR','AbsOnAxis_HR','Delta_OnAxis','AbsOffAxis_V','AbsOffAxis_HR','Delta_OffAxis', 'AbsOnAxis_V'] 
    # print(players_df.columns - sumona_features)
    
    # players_df = players_df.drop(columns=list(set(players_df.columns) - set(sumona_features)))
    players_df = players_df[sumona_features]
    players_df = players_df.dropna(axis="index")
    players_df.to_excel(f"{DATA_DIR}cleaned_data_test.xlsx", index=False)
    # players_df["Bimanual Score: Button"] = players_df["Bimanual Score: Button"].fillna(players_df["Bimanual Score: Button"].mean())
    # players_df["Delta_BallPath"] = players_df["Delta_BallPath"].fillna(players_df["Delta_BallPath"].mean())
    # players_df["Delta: VE"] = players_df["Delta: VE"].fillna(players_df["Delta: VE"].mean())
    # players_df["# of concussions"] = players_df["# of concussions"].fillna(players_df["# of concussions"].mean())
    # players_df["TMT_V"] = players_df["TMT_V"].fillna(players_df["TMT_V"].mean())
    # players_df["TMT_HR"] = players_df["TMT_HR"].fillna(players_df["TMT_HR"].mean())
    # players_df["FullPath_HR"] = players_df["FullPath_HR"].fillna(players_df["FullPath_HR"].mean())
    # players_df["FullPath_V"] = players_df["FullPath_V"].fillna(players_df["FullPath_V"].mean())
    # players_df["DR Errors: V"] = players_df["DR Errors: V"].fillna(players_df["DR Errors: V"].mean())
    # players_df["DR Errors: HR"] = players_df["DR Errors: HR"].fillna(players_df["DR Errors: HR"].mean())


    players_df.insert(8, "drafted", column_to_move)


    return players_df



def create_dataset(df, target_col="NHL"):

    if target_col == "NHL":
        # non_feature_cols = ["year","DOB", "draft year", "shoots", "Position", "drafted", "draft number"]
        non_feature_cols = ["drafted", "previous concussions?"]
    elif target_col == "previous concussions?":
        non_feature_cols = ["year","DOB", "draft year", "shoots", "Position", "drafted", "draft number", "NHL", "# of concussions"]



    y = df[target_col]
    X = df.drop(columns=non_feature_cols + [target_col])

    matlab_df = X.copy()
    matlab_df["target"] = y
    matlab_df.to_excel(f"{DATA_DIR}cleaned_data_matlab_{target_col}.xlsx")


    return X,y


# def get_all_results(filepath, include_cv=False):
#     results = pd.DataFrame()
#     for (dirpath, dirnames, filenames) in os.walk(filepath):
#         for filename in filenames:
#             if filename.endswith(".xlsx"):
#                 uuid = filename.split(".")[0].split("_")[-1]
#                 df = pd.read_excel(os.path.join(dirpath, filename), engine="openpyxl")
#                 df["uuid"] = uuid
                
#                 path_to_logs = f"training_logs/{uuid}.json"
#                 if os.path.exists(path_to_logs):
#                     with open(path_to_logs) as f:
#                         training_args = json.load(f)
                    
#                     if not include_cv:
#                         del training_args["cv"]
#                     df["training_args"] = str(training_args)
#                 results = pd.concat([results, df])

#     return results
                