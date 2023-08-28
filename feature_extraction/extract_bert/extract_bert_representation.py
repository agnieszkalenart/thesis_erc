import pickle
from sklearn import model_selection, metrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import ast

## LOAD DATA

trainID = pickle.load(open("CMN/IEMOCAP/data/trainID.pkl",'rb'), encoding="latin1")
testID = pickle.load(open("CMN/IEMOCAP/data/testID.pkl",'rb'), encoding="latin1")
trainID, valID = model_selection.train_test_split(trainID, test_size=.2, random_state=1227)

text_transcripts_emb, _, _ = pickle.load(open('features/text_bert_distilbert.pickle','rb'), encoding="latin1")


df = pd.DataFrame.from_dict(text_transcripts_emb, orient='index')
transcripts, labels, own_historyID, other_historyID, own_historyID_rank, other_historyID_rank = pickle.load(open("CMN/IEMOCAP/data/dataset.pkl",'rb'), encoding="latin1")

## TRAIN A FCN ON TEXT EMB

label_idx = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
labels_array = np.asarray([label_idx[labels[ID]] for ID in df.index])

# Split the data into features and labels
labels = pd.Series(labels_array)

# Split the data into training, validation and testing sets
X_train = df.loc[trainID]
X_test = df.loc[testID]
X_val = df.loc[valID]

df['labels'] = labels_array
df_new = df[['labels']]
y_train = df_new.loc[trainID]
y_test = df_new.loc[testID]
y_val = df_new.loc[valID]

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define the neural network model
model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(len(labels.unique()), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val))

# Predict the emotions on the testing set
predict_x = model.predict(X_test)
classes_x=np.argmax(predict_x,axis=1)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, classes_x)
print(f'Accuracy: {accuracy}')

# Create a new model that outputs the activations of the last layer
representation_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Extract the representations for each example
train_representations = representation_model.predict(X_train)
val_representations = representation_model.predict(X_val)
test_representations = representation_model.predict(X_test)

# Create dict representations
representations = {}
representations.update({idx: rep for idx, rep in zip(trainID, train_representations)})
representations.update({idx: rep for idx, rep in zip(valID, val_representations)})
representations.update({idx: rep for idx, rep in zip(testID, test_representations)})


## CREATE TEXT EMB OWN AND OTHER HISTORY

transcription_emb = representations
indices = representations.keys()


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



transcription_own_history_emb = []
transcription_other_history_emb = []

for iddx, ID in enumerate(indices):

		combined_historyID_rank = df['other_history_rank'].loc[ID] + df['own_history_rank'].loc[ID]

		if combined_historyID_rank > 0:
			textOwnHistoryEmb = np.asarray([transcription_emb[df['own_history'].loc[ID][idx]] if df['own_history'].loc[ID][idx] in transcription_emb else np.zeros(200) for idx in range(df['own_history_rank'].loc[ID])])
			textOtherHistoryEmb = np.asarray([transcription_emb[df['other_history'].loc[ID][idx]] if df['other_history'].loc[ID][idx] in transcription_emb else np.zeros(200) for idx in range(df['other_history_rank'].loc[ID])])
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
with open('features/representations_text_bert_200.pickle', 'wb') as file:
    pickle.dump([transcription_emb, 
                transcription_own_history_emb,
                transcription_other_history_emb],
                 file)


