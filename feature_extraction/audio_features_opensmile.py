#!/usr/bin/env python
import subprocess
import os
import pickle
#import opensmile
import arff

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

AUDIOS_FOLDER = "/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/MELD.Raw/meld_small_resampled/"
AUDIO_FEATURES_PATH = "/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/features/audio_features_opensmile.p"
CONF_PATH = "/Users/agnieszkalenart/Downloads/opensmile-3.0.0/config/is09-13/IS13_ComParE.conf"
TMPARFFPATH = "/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/MELD.Raw/tmp/tmp.arff"

# smile = opensmile.Smile(
#     feature_set=opensmile.FeatureSet.ComParE_2016,
#     feature_level=opensmile.FeatureLevel.Functionals,
# )

def get_librosa_features(path: str) -> np.ndarray:
    os.system("SMILExtract -C " + CONF_PATH + " -I " + path + " -O " + TMPARFFPATH)
    data = arff.load(open(TMPARFFPATH, 'r'))
    #data = data['data']
    data_list = []
    for i in len(data['data']):
        data_list.append((data['data'][i][1:-1]))
    #data = np.asarray(data['data'][0][1:-1])
    return data_list


def save_audio_features() -> None:
    audio_feature = {}
    for filename in tqdm(os.listdir(AUDIOS_FOLDER), desc="Computing the audio features"):
        if not filename.startswith('.'):
            id_ = filename.rsplit(".", maxsplit=1)[0]
            audio_feature[id_] = get_librosa_features(os.path.join(AUDIOS_FOLDER, filename))

    with open(AUDIO_FEATURES_PATH, "wb") as file:
        pickle.dump(audio_feature, file, protocol=2)


def get_audio_duration() -> None:
    filenames = os.listdir(AUDIOS_FOLDER)
    print(sum(librosa.core.get_duration(filename=os.path.join(AUDIOS_FOLDER, filename))
              for filename in tqdm(filenames, desc="Computing the average duration of the audios") if not filename.startswith('.')) / len([filename for filename in filenames if not filename.startswith('.')]))



def main() -> None:
    get_audio_duration()

    save_audio_features()
    
    with open(AUDIO_FEATURES_PATH, "rb") as file:
         pickle.load(file)


if __name__ == "__main__":
    main()