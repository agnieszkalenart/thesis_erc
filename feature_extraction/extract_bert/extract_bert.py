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

# Load the IEMOCAP dataset
df = pd.read_csv('features/iemocap_with_history.csv')

df = df.rename(columns={'Unnamed: 0': 'indices'})
df = df.rename(columns={'emotion': 'labels'})
df = df.rename(columns={'sentences': 'utterances'})
df['labels'] = df['labels'].fillna('')

# Drop all unnecessary columns
columns_to_keep = ['indices', 'labels', 'utterances', 'own_history_sentences', 'other_history_sentences']
df = df.drop(columns=[col for col in df.columns if col not in columns_to_keep])

# encode labels
label_idx = {'happiness':0, 'sadness':1, 'neutral':2, 'anger':3, 'excited':4, 'frustration':5, 'unassigned':6, 'surprise':6, 'other':6, 'fear':6, 'disgusted':6, '':6 }
df['labels'] = df['labels'].apply(lambda x: label_idx[x])

# drop rows with label 6
df = df[df['labels'] != 6]

# Create sentence and label lists
transcription = df.utterances.values
transcription_own_history = df.own_history_sentences.values
transcription_other_history = df.other_history_sentences.values
indices = df.indices.values

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

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


# Save the dictionary as a pickle file
with open('/features/cmn_text_bert.pickle', 'wb') as file:
    pickle.dump([transcription_emb], 
                #  transcription_own_history_emb,
                #  transcription_other_history_emb],
                 file)


# # Function to extract words from a sentence
# def extract_words(sentence):
#     words = re.findall(r'\w+', sentence)  # Use regex to find all alphanumeric sequences
#     return words

# # Apply the function to create the 'words' column
# df['words'] = df['utterances'].apply(extract_words)

# # Check for maximum number of words in a sentence
# max_words = max(df['words'].apply(len))

# print('Maximum number of words in a sentence:', max_words)

# # Check for average number of words in a sentence
# avg_words = np.mean(df['words'].apply(len))

# print('Average number of words in a sentence:', avg_words)

# # Check for number of uttersances that have more than 30 words
# print('Number of utterances that have more than 30 words:', sum(df['words'].apply(len) > 30))


