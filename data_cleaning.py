import pandas as pd
import numpy as np

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

def clean_raw_data(filename="Brdi_db_march.xlsx"):
    players_df = pd.read_excel("Brdi_db_march.xlsx", engine="openpyxl").drop(columns=[123, "id", "Data Initials", "Code Name", "draft status", ])

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
    players_df.insert(8, "drafted", column_to_move)

    return players_df



def create_dataset(df, target_col="NHL"):

    non_feature_cols = ["year","DOB", "draft year", "shoots", "Position", "drafted", "draft number"]


    y = df[target_col]
    X = df.drop(columns=non_feature_cols + [target_col])

    X = X.fillna(X.mean())


    return X,y