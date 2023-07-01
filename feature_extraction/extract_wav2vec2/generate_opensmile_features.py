from IEMOCAP_reader import IEMOCAPReader
from dataframe_processing import Relabel, Filter, LabelEncoder, OutputMerger
from feature_extractor import Wav2Vec2Embeddings, OpensmileExtractor, Spectrogram
# from splits import Split, GroupSplit, LeaveOneOut
from normalize import NormalizationStatistics
# from generators import BatchGenerator
# from dienen_model import DienenModel
from data_processors import PadDP, ToNumpyDP, SqueezeDP, NormalizeDP
import pandas as pd
import pickle
import numpy as np
import ast

DATA_PATH = '/Users/agnieszkalenart/Documents/mannheim/master_thesis/IEMOCAP'

#read data
iemocap_reader = IEMOCAPReader()
df_data = iemocap_reader.process(data_path=DATA_PATH, min_duration=0.0, min_sad_frames_duration=0, sample=None,compute_speech_rate=True)
print(df_data.loc[['Ses01F_impro04_M010']])

#small data
# df_data= df_data.head(5)

#create opensmile features
opensmile = OpensmileExtractor()
data = opensmile.process(data=df_data, output_column='opensmile')

# to opensmile pickle
data['opensmile'] = data['opensmile'].apply(lambda x: ast.literal_eval(x))
data['opensmile'] = data['opensmile'].apply(np.array)
data['opensmile'] = data['opensmile'].apply(lambda x: x.reshape(6373,))
opensmile_dict = dict(zip(data.index, data['opensmile']))

with open('/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/features/cmn_audio_opensmile.pickle', 'wb') as file:
    pickle.dump(opensmile_dict, file)
