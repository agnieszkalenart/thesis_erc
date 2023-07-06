from IEMOCAP_reader import IEMOCAPReader
import pickle
import pandas as pd

DATA_PATH = '/Users/mareklukasikbackup/Aga/IEMOCAP_full_release'
OUTPUT_PATH = 'features/iemocap_with_history.csv'

transcripts, labels, own_historyID, other_historyID, own_historyID_rank, other_historyID_rank = pickle.load(open("CMN_wav2vec2/IEMOCAP/data/dataset.pkl",'rb'), encoding="latin1")

# add own history ids to dataframe
own_history_df = pd.DataFrame.from_dict(own_historyID, orient='index')
own_history_df['own_history'] = own_history_df.apply(lambda row: list(row[range(86)]), axis=1)
own_history_df.drop(columns=range(86), inplace=True)
print(own_history_df)

# add other history ids to dataframe
other_history_df = pd.DataFrame.from_dict(other_historyID, orient='index')
other_history_df['other_history'] = other_history_df.apply(lambda row: list(row[range(86)]), axis=1)
other_history_df.drop(columns=range(86), inplace=True)
print(other_history_df)
      

#read data
iemocap_reader = IEMOCAPReader()
df_data = iemocap_reader.process(data_path=DATA_PATH, min_duration=0.0, min_sad_frames_duration=0, sample=None,compute_speech_rate=True)

#concatenate dataframes
df_data = pd.concat([df_data, own_history_df, other_history_df], axis=1)

# add own history sentences to dataframe
def get_own_history_sentences(row):
    history_sentences = ""
    for i in range(86):
        previous_utterance_id = df_data.loc[row, 'own_history'][i]
        if previous_utterance_id != None:
            history_sentences += str(df_data.loc[previous_utterance_id, 'sentences']) + " "
    return history_sentences

# add other history sentences to dataframe
def get_other_history_sentences(row):
    history_sentences = ""
    for i in range(86):
        previous_utterance_id = df_data.loc[row, 'other_history'][i]
        if previous_utterance_id != None:
            history_sentences += str(df_data.loc[previous_utterance_id, 'sentences']) + " "
    return history_sentences

df_data['own_history_sentences'] = df_data.apply(lambda row: get_own_history_sentences(row.name), axis=1)
df_data['other_history_sentences'] = df_data.apply(lambda row: get_other_history_sentences(row.name), axis=1)


df_data.to_csv(OUTPUT_PATH)