import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import pickle
import ast

# Load the IEMOCAP dataset
df = pd.read_csv('features/iemocap_with_history.csv')

df = df.rename(columns={'Unnamed: 0': 'indices'})
df = df.rename(columns={'emotion': 'labels'})
df = df.rename(columns={'sentences': 'utterances'})
df['labels'] = df['labels'].fillna('')

# Drop all unnecessary columns
columns_to_keep = ['indices', 'labels', 'utterances', 'own_history_sentences', 'other_history_sentences', 'own_history', 'other_history']
df = df.drop(columns=[col for col in df.columns if col not in columns_to_keep])

# encode labels
label_idx = {'happiness':0, 'sadness':1, 'neutral':2, 'anger':3, 'excited':4, 'frustration':5, 'unassigned':6, 'surprise':6, 'other':6, 'fear':6, 'disgusted':6, '':6 }
df['labels'] = df['labels'].apply(lambda x: label_idx[x])

# drop rows with label 6
df = df[df['labels'] != 6]

# read list columns as list
df['own_history'] = df['own_history'].apply(lambda x: ast.literal_eval(x))
df['other_history'] = df['other_history'].apply(lambda x: ast.literal_eval(x))

# Create sentence and label lists
transcription = df.utterances.values
transcription_own_history = df.own_history_sentences.values
transcription_other_history = df.other_history_sentences.values
indices = df.indices.values
own_history = df.own_history.values
other_history = df.other_history.values
df.index = df.indices.values

# create own_history_rank and other_historyID_rank
def calculate_rank_own_history (row):
	rank = 0
	for i in row['own_history']:
		if i is not None:
			rank+=1
	return rank

df['own_history_rank'] = df.apply(calculate_rank_own_history, axis=1)

def calculate_rank_other_history (row):
	rank = 0
	for i in row['other_history']:
		if i is not None:
			rank+=1
	return rank

df['other_history_rank'] = df.apply(calculate_rank_other_history, axis=1)

other_history_rank = df.other_history_rank.values
own_history_rank = df.own_history_rank.values


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

transcription_tokenized = tokenizer(transcription.tolist(), padding = True, truncation = True, return_tensors="pt")
# transcription_own_history_tokenized = tokenizer(transcription.tolist(), padding = True, truncation = True, return_tensors="pt")
# transcription_other_history_tokenized = tokenizer(transcription.tolist(), padding = True, truncation = True, return_tensors="pt")

print("checkpoint: tokenization done")

with torch.no_grad():
  transcription_hidden = model(**transcription_tokenized) #dim : [batch_size(nr_sentences), tokens, emb_dim]
  # transcription_own_history_hidden = model(**transcription_own_history_tokenized) #dim : [batch_size(nr_sentences), tokens, emb_dim]
  # transcription_other_history_hidden = model(**transcription_other_history_tokenized) #dim : [batch_size(nr_sentences), tokens, emb_dim]

print("checkpoint: model done")


#get only the [CLS] hidden states
cls_transcription = transcription_hidden.last_hidden_state[:,0,:].numpy()
# cls_transcription_own_history = transcription_own_history_hidden.last_hidden_state[:,0,:].numpy()
# cls_transcription_other_history = transcription_other_history_hidden.last_hidden_state[:,0,:].numpy()

print("checkpoint: cls extraction done")


transcription_emb = dict(zip(indices, cls_transcription))
# transcription_own_history_emb = dict(zip(indices, cls_transcription_own_history))
# transcription_other_history_emb = dict(zip(indices, cls_transcription_other_history))

print("checkpoint: dict created")

#  if transciprtion_emb is already created, load file
# transcription_emb, _, _ = pickle.load(open('features/cmn_text_bert.pickle', 'rb'), encoding="latin1")


transcription_own_history_emb = []
transcription_other_history_emb = []

for iddx, ID in enumerate(indices):

		combined_historyID_rank = df['other_history_rank'].loc[ID] + df['own_history_rank'].loc[ID]

		if combined_historyID_rank > 0:
			textOwnHistoryEmb = np.asarray([transcription_emb[df['own_history'].loc[ID][idx]] if df['own_history'].loc[ID][idx] in transcription_emb else np.zeros(768) for idx in range(df['own_history_rank'].loc[ID])])
			textOtherHistoryEmb = np.asarray([transcription_emb[df['other_history'].loc[ID][idx]] if df['other_history'].loc[ID][idx] in transcription_emb else np.zeros(768) for idx in range(df['other_history_rank'].loc[ID])])
			transcription_own_history_emb.append(textOwnHistoryEmb)
			transcription_other_history_emb.append(textOtherHistoryEmb)
		else:
			textOwnHistoryEmb = np.empty((0))
			textOtherHistoryEmb = np.empty((0))
			transcription_own_history_emb.append(textOwnHistoryEmb)
			transcription_other_history_emb.append(textOtherHistoryEmb)

transcription_own_history_emb = dict(zip(indices, transcription_own_history_emb))
transcription_other_history_emb = dict(zip(indices, transcription_other_history_emb))

# Save the dictionary as a pickle file
with open('features/cmn_text_bert_with_history.pickle', 'wb') as file:
    pickle.dump([transcription_emb, 
                transcription_own_history_emb,
                transcription_other_history_emb],
                 file)

