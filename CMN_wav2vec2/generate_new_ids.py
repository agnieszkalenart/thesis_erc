import pickle
import random

AUDIO_EMBEDDINGS = 'features/cmn_audio_wav2vec2.pickle'
TEXT_EMBEDDINGS = 'features/cmn_text_bert.pickle'
TRAIN_IDS = 'CMN_wav2vec2/IEMOCAP/data/trainID.pkl'
TEST_IDS = 'CMN_wav2vec2/IEMOCAP/data/testID.pkl'

# load data
audio_emb = pickle.load(open(AUDIO_EMBEDDINGS, 'rb'), encoding="latin1")
audio_emb_ids = list(audio_emb.keys())
text_emb, _, _ = pickle.load(open(TEXT_EMBEDDINGS, 'rb'), encoding="latin1") 

train_ids = pickle.load(open(TRAIN_IDS, 'rb'), encoding="latin1")
test_ids = pickle.load(open(TEST_IDS, 'rb'), encoding="latin1")

# check lengths

print("old lengths:")
print(len(audio_emb))
print(len(text_emb))
print(len(train_ids))
print(len(test_ids))

# filter train ids and test ids that are in audio emb
train_ids = [id for id in train_ids if id in audio_emb.keys() and id in text_emb.keys()]
test_ids = [id for id in test_ids if id in audio_emb.keys() and id in text_emb.keys()]

# check lengths

print("new lengths:")
print(len(audio_emb))
print(len(text_emb))
print(len(train_ids))
print(len(test_ids))

# save ids
pickle.dump(train_ids, open('CMN_wav2vec2/IEMOCAP/data/trainID_new.pkl', 'wb'))
pickle.dump(test_ids, open('CMN_wav2vec2/IEMOCAP/data/testID_new.pkl', 'wb'))