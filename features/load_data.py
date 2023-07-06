import pickle 

AUDIO_EMBEDDINGS = "CMN_wav2vec2/IEMOCAP/data/audio/IEMOCAP_audio_features.pickle"

audio_emb = pickle.load(open(AUDIO_EMBEDDINGS, 'rb'), encoding="latin1")


print(audio_emb['Ses03F_impro06_FXX0'])