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
data = iemocap_reader.process(data_path=DATA_PATH, min_duration=0.0, min_sad_frames_duration=0, sample=None,compute_speech_rate=True)

#small data
# df_data= df_data.head(5)

#create spectrogram features
spectrogram = Spectrogram()
data = spectrogram.process(data=data, output_column='spectrogram')
spect_dict = dict(zip(data.index, data['spectrogram']))

# Save the dictionary as a pickle file
with open('/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/features/audio_spectrogram.pickle', 'wb') as file:
    pickle.dump(spect_dict, file)
