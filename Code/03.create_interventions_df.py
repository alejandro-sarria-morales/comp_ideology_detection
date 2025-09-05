import pandas as pd
import os
import shutil
import ast 

#chunk the main sessions dataframe
sessions_path = r"C:\Users\asarr\Documents\Projects\comp_ideology_detection\outputs\sessions.csv"
sessions_df = pd.read_csv(sessions_path)

chunk_size = sessions_df.shape[0] // 10
chunks = [sessions_df[i:i + chunk_size] for i in range(0, sessions_df.shape[0], chunk_size)]

# create intervention for chunk
chunk_dir = r"C:\Users\asarr\Documents\Projects\comp_ideology_detection\outputs\temp_chunks"
os.makedir(chunk_dir)

intervention_id = 0
for chunk in chunks:
    chunk_ints = pd.DataFrame(columns=["session_id", "intervention_id", "speaker_text", "intervention_text"])
    for i, row in chunk.iterrows():
        session_id = row['id']
        intervention_pairs = ast.literal_eval(row['intervention_pairs'])
        
        for i in range(len(intervention_pairs)):
            speaker_text = intervention_pairs[i][0]
            intervention_text = intervention_pairs[i][1]
            
            new_row = {
                "session_id": session_id,
                "intervention_id": intervention_id,
                "speaker_text": speaker_text,
                "intervention_text": intervention_text
            }
            chunk_ints = pd.concat([chunk_ints, pd.DataFrame([new_row])], ignore_index=True)
            intervention_id += 1
    chunk_ints.to_csv(os.path.join(chunk_dir, f"interventions_chunk_{i}.csv"), index=False)

# join chunks, save intervention_df
intervention_df = pd.DataFrame(columns=["session_id", "intervention_id", "speaker_text", "intervention_text"])
for file_name in os.listdir(chunk_dir):
    file_path = os.path.join(chunk_dir, file_name)
    chunk = pd.read_csv(file_path)
    intervention_df = pd.concat([intervention_df, chunk], ignore_index=True)

intervention_df.to_csv(r"C:\Users\asarr\Documents\Projects\comp_ideology_detection\outputs\interventions.csv", index=False)

# delete temp files
shutil.rmtree(chunk_dir)