import pickle
import numpy as np
from sklearn import model_selection, metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


wav2vec2_representations = pickle.load(open('features/audio_wav2vec2_representations_mean_none.pkl', 'rb'))
opensmile_representations = pickle.load(open('features/audio_opensmile.pickle', 'rb'))
trainID = pickle.load(open("CMN/IEMOCAP/data/trainID.pkl",'rb'), encoding="latin1")
testID = pickle.load(open("CMN/IEMOCAP/data/testID.pkl",'rb'), encoding="latin1")
trainID, valID = model_selection.train_test_split(trainID, test_size=.2, random_state=1227)

# Concatenate the representations
new_dict = {}  
intersection_keys = set(wav2vec2_representations.keys()).intersection(opensmile_representations.keys())

for key in intersection_keys:
    new_dict[key] = np.concatenate((wav2vec2_representations[key], opensmile_representations[key]))

audio_emb = new_dict

df = pd.DataFrame.from_dict(audio_emb, orient='index')
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
model.add(Dense(768, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(len(labels.unique()), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

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



# add missing representation
missing_dict = {"Ses03M_impro03_M001" : np.zeros(768)}
representations.update(missing_dict)


with open('features/audio_wav2vec2_opensmile_representations_768.pkl', 'wb') as f:
    pickle.dump(representations, f)