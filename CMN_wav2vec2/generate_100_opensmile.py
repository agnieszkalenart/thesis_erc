import pickle
from sklearn import model_selection, metrics
import numpy as np
import pandas as pd

AUDIO_EMBEDDINGS = '/Users/agnieszkalenart/Documents/mannheim/master_thesis/cmn_audio_opensmile_new.pickle'
trainID = pickle.load(open("/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/CMN_wav2vec2/IEMOCAP/data/trainID_new.pkl",'rb'), encoding="latin1")
testID = pickle.load(open("/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/CMN_wav2vec2/IEMOCAP/data/testID_new.pkl",'rb'), encoding="latin1")
valID,_ = model_selection.train_test_split(testID, test_size=.4, random_state=1227)


audio_emb = pickle.load(open(AUDIO_EMBEDDINGS, 'rb'), encoding="latin1")
