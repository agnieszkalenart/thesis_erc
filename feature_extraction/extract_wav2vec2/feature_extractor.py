import fairseq
import torch
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2CtcConfig, Wav2VecCtc
from torchaudio.models.wav2vec2.utils import import_fairseq_model
import librosa
#from  paips.core import Task
from pathlib import Path
from tqdm import tqdm
import numpy as np
import joblib
import arff
import pandas as pd
import os

#from paips.utils import GenericFile
# import opensmile

MODEL_PATH = '/Users/agnieszkalenart/Documents/mannheim/master_thesis/wav2vec2/wav2vec_small_960h.pt'

CONF_PATH = "/Users/agnieszkalenart/Downloads/opensmile-3.0.0/config/is09-13/IS13_ComParE.conf"
TMPARFFPATH = "/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/tmp/tmp.arff"

class OpensmileExtractor():
    def process(self, data, output_column, max_size = None):

        def smile_process_file(file_path):
            if os.path.exists(TMPARFFPATH):
                os.remove(TMPARFFPATH)
            os.system("SMILExtract -C " + CONF_PATH + " -I " + file_path + " -O " + TMPARFFPATH)
            data = arff.load(open(TMPARFFPATH, 'r'))
            print(data)
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

class Wav2Vec2Embeddings():
    def build_wav2vec_model(self, model_path,  dict_path=None):
        
        arg_override = {'activation_dropout': 0.0,
                        'attention_dropout': 0.0,
                        'dropout': 0.0,
                        'dropout_features': 0.0,
                        'dropout_input': 0.0,
                        'encoder_layerdrop': 0.0,
                        'pooler_dropout': 0.0}

        if dict_path:
            arg_override.update({"data": dict_path})

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path], arg_overrides=arg_override)
        return model[0], cfg, task

    def process(self, data, model_path, max_size, mode, output_column, layer, save_feature_files):
        dict_path = None
        if isinstance(mode,list):
            mode = np.expand_dims(np.array(mode),axis=[0,2])
        if save_feature_files:
            # feature_output_path = GenericFile(self.global_parameters['cache_path'],self.task_hash,'features')
            # feature_output_path.mkdir(parents=True)
            feature_output_path = '/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/features/cache/'

        wav2vec_model,model_config,task = self.build_wav2vec_model(model_path,dict_path)
        imported = import_fairseq_model(wav2vec_model)
        #wav2vec_model.cuda()

        def extract_embedding(filename):
            output_feature_filename = feature_output_path+filename
            x,fs = librosa.core.load(filename, sr=16000)
            if max_size:
                x = x[:max_size]
            x = torch.tensor(np.expand_dims(x,axis=0))
            # print(x.shape)
            # .cuda()
            activations, _ = imported.extract_features(x, None)
# not sure about it
# keys in activations: 'x', 'padding_mask', 'features', 'layer_results'
            #activations = activations['features']
            # print(len(activations))
            # print(activations[0].shape)
            if layer == 'output':
                features = activations[-1].detach().numpy()
            elif layer == 'local_encoder':
                features = activations[0].cpu().detach().numpy()
            elif layer == 'transformer_layers':
                features = [activations[i].cpu().detach().numpy() for i in range(1,len(activations))]
            elif layer == 'enc_and_transformer':
                features = [activation.cpu().detach().numpy() for activation in activations]
            elif isinstance(layer,list):
                features = [activations[i].cpu().detach().numpy() for i in layer]
            elif isinstance(layer,dict):
                layer_from = layer.get('from',0)
                layer_to = layer.get('to',len(activations))
                features = activations[layer_from:layer_to]
                features = [f.cpu().detach().numpy() for f in features]
            
            if not isinstance(features,list):
                features = [features]

            features = np.concatenate(features,axis=1)

            if mode == 'mean':
                features = np.mean(features,axis=0).astype(np.float32)
            elif mode == 'sequence':
                pass
            elif type(mode).__name__=='ndarray':
                features = np.mean(mode*features,axis=1).astype(np.float32)

            # if save_feature_files:
            #     joblib.dump(features, output_feature_filename.local_filename)
            #     output_feature_filename.upload_from(output_feature_filename.local_filename)
            #     return output_feature_filename
            # else:
            return features

        tqdm.pandas()

        data[output_column] = data['filename'].progress_apply(extract_embedding)
        return data