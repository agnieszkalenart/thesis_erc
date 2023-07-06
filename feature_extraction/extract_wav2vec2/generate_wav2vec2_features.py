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



DATA_PATH = '/Users/mareklukasikbackup/Aga/IEMOCAP_full_release'
MODEL_PATH = '/Users/mareklukasikbackup/Aga/pre-trained_models/wav2vec_small.pt'


#read data
iemocap_reader = IEMOCAPReader()
df_data = iemocap_reader.process(data_path=DATA_PATH, min_duration=0.0, min_sad_frames_duration=0, sample=None,compute_speech_rate=True)

#create wav2vec2 features
wav2vec2_model_path = MODEL_PATH
wav2vec2_dict_path = None
normalization_axis = 1
wav2vec2_padding_axis = 1
max_size = 240000
mode  = 'sequence'
output_column = 'wav2vec2'
layer = 'output'
save_feature_files = True

wav2vec2 = Wav2Vec2Embeddings()
model, cfg, task = wav2vec2.build_wav2vec_model(model_path=wav2vec2_model_path)
data = wav2vec2.process(data = df_data,
                model_path=wav2vec2_model_path,
                 max_size=max_size,
                 mode= mode,
                 output_column=output_column,
                 layer = layer,
                 save_feature_files= save_feature_files)

data['wav2vec2'] = data['wav2vec2'].apply(lambda x: [item for sublist in x for item in sublist])
data['wav2vec2'] = data['wav2vec2'].apply(np.array)
cmn_emb = dict(zip(data.index, data['wav2vec2']))


# Save the dictionary as a pickle file
with open('features/cmn_audio_wav2vec2.pickle', 'wb') as file:
    pickle.dump(cmn_emb, file)