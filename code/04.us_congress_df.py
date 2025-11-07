import pandas as pd
import os
import re

def integrate_files(type):
    """
    Makes a big dataframe out of a bunch of them in the data directory.

        - type (str): either "SpeakerMap" or "speeches"
    """
    path = "../data/hein-daily"
    lst = [os.path.join(path, f) for f in os.listdir(path) 
       if type in f and any(int(num) > 106 for num in re.findall(r'\d+', f))]
    
    dfs = [pd.read_csv(f, sep="|", encoding="cp1252", on_bad_lines="skip") for f in lst]

    full_df =  pd.concat(dfs, ignore_index=True)

    return full_df

if __name__ == "__main__":
    speeches = integrate_files("speeches")
    speakermap = integrate_files("SpeakerMap")

    speeches.merge(speakermap, on="speech_id", how="left")\
        .dropna(subset=["speakerid", "party"])\
        .drop(["lastname", "firstname", "district", "nonvoting", "gender", "state"], axis=1)\
        .to_csv("../outputs/interventions_us.csv")