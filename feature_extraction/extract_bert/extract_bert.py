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
df = pd.read_csv('/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/features/iemocap.csv')

# Load history of utterances
transcripts, labels, own_historyID, other_historyID, own_historyID_rank, other_historyID_rank = pickle.load(open("/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/CMN_wav2vec2/IEMOCAP/data/dataset.pkl",'rb'), encoding="latin1")

df = df.rename(columns={'Unnamed: 0': 'indices'})
df = df.rename(columns={'emotion': 'labels'})
df = df.rename(columns={'sentences': 'utterances'})

# Drop all unnecessary columns
columns_to_keep = ['indices', 'labels', 'utterances']
df = df.drop(columns=[col for col in df.columns if col not in columns_to_keep])

# encode labels
label_idx = {'happiness':0, 'sadness':1, 'neutral':2, 'anger':3, 'excited':4, 'frustration':5, 'unassigned':6, 'surprise':6, 'other':6, 'fear':6, 'disgusted':6 }
df['labels'] = df['labels'].apply(lambda x: label_idx[x])

# drop rows with label 6
df = df[df['labels'] != 6]

# Create sentence and label lists
sentences = df.utterances.values
indices = df.indices.values

# # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
# sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
# labels = df.labels.values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
# print ("Tokenize the first sentence:")
# print (tokenized_texts[0])

# MAX_LEN = 87

# # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
# input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# # Pad our input tokens
# input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# # Create attention masks
# attention_masks = []

# # Create a mask of 1s for each token followed by 0s for padding
# for seq in input_ids:
#   seq_mask = [float(i>0) for i in seq]
#   attention_masks.append(seq_mask)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

tokenized = tokenizer(sentences.tolist(), padding = True, truncation = True, return_tensors="pt")

with torch.no_grad():
  hidden = model(**tokenized) #dim : [batch_size(nr_sentences), tokens, emb_dim]

#get only the [CLS] hidden states
cls_token = hidden.last_hidden_state[:,0,:].numpy()

print(cls_token.shape)
print(cls_token)

bert_emb = dict(zip(indices, cls_token))


# Save the dictionary as a pickle file
with open('/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/features/cmn_text_bert.pickle', 'wb') as file:
    pickle.dump(bert_emb, file)


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


