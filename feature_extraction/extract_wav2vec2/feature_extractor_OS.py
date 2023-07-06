import librosa
from pathlib import Path
from tqdm import tqdm
import numpy as np
import joblib
import arff
import pandas as pd
import os

#from paips.utils import GenericFile
# import opensmile

MODEL_PATH = '/Users/mareklukasikbackup/Aga/pre-trained_models/wav2vec_small.pt'

CONF_PATH = "/Users/mareklukasik_1/Downloads/opensmile-3.0.0/config/is09-13/IS13_ComParE.conf"
TMPARFFPATH = "tmp/tmp.arff"

class OpensmileExtractor():
    def process(self, data, output_column, max_size = None):

        def smile_process_file(file_path):
            if os.path.exists(TMPARFFPATH):
                os.remove(TMPARFFPATH)
            os.system("SMILExtract -C " + CONF_PATH + " -I " + file_path + " -O " + TMPARFFPATH)
            data = arff.load(open(TMPARFFPATH, 'r'))
            data_list = []
            for item in data['data']:
                data_list.append(item[1:-1])
            return data_list 

        def extract_embedding(filename):
            y = smile_process_file(filename)
            if max_size:
                y = y[:max_size]
            return y
        
        tqdm.pandas()

        data[output_column] = data['filename'].progress_apply(extract_embedding)
        return data

class Spectrogram():
    def process(self, data, output_column, spec_type = 'magnitude', frame_size = 400, hop_size = 100, nfft = 400, window = 'hann', save_feature_files = True, log_offset = 1e-12):

        def extract_embedding(x):
            x, fs = librosa.core.load(x,sr=None)
            X = librosa.stft(x, n_fft=nfft, hop_length=hop_size, win_length=frame_size, window=window)
            X = X.T
            if spec_type == 'magnitude':
                y = np.abs(X)
            elif spec_type == 'complex':
                y = X
            elif spec_type == 'log_magnitude':
                y = np.log(np.abs(X)+log_offset)

            return y

        tqdm.pandas()

        data[output_column] = data['filename'].progress_apply(extract_embedding)
        return data
