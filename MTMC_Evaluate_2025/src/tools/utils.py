import pandas as pd
import os 
import re
            
def convert_txt_to_csv(lines): 
    """Convert MTMC txt file to csv"""
    columns = ['CameraId', 'FrameId', 'Id', 'X', 'Y', 'Width', 'Height']
    df_init = {k:[] for k in columns}
    for track_info in lines:
        if len(track_info) == 2: continue # Account for cases with empty gt/pred files
        for k, v in zip(columns, track_info):
            if(k != "CameraId"):
                df_init[k].append(float(v))
            else:
                df_init[k].append(v)
    df = pd.DataFrame(df_init)    
    return df

def convert_mot_txt_to_csv(lines): 
    """Convert MOT txt file to csv"""
    columns = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']
    df_init = {k:[] for k in columns}
    for track_info in lines:
        if len(track_info) == 1: continue # Account for cases with empty gt/pred files
        for k, v in zip(columns, track_info):
            df_init[k].append(float(v))
    df = pd.DataFrame(df_init)    
    return df

def convert_mot_to_mtmc(file):
    """Add camera id to MOT results file"""
    # camera_id = os.path.basename(file).split(".")[0]

    # here we are taking the file name as camera name
    camera_id = file.split(os.sep)[-1][:-4]
    lines = []
    with open(file, "rt") as f:
        for line in f:
            output = line.strip().split(",") 
            output = [camera_id] + output
            lines.append(output) 
    return lines

def convert_dataframe_to_txt(df, save_path, type="gt"):
    """Convert dataframe to MOT format text file"""
    df_val = df.values.tolist()
    tmp_list = [1,1,1] if type == "gt" else [1,-1,-1,-1]
    with open(save_path, "w+") as f:
        for row in df_val:
            line = row + tmp_list
            line = list(map(str, line))
            f.write(",".join(line) + "\n")
    
def merge_dataframe(dfs): 
    # csv_files = glob.glob(os.path.join(path, "*.csv"))
    # dfs = []
    # for file in csv_files:
    #     df = pd.read_csv(file) 
    #     dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
