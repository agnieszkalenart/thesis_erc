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

SMALL = False
AUDIO_EMBEDDINGS = 'features/cmn_audio_wav2vec2.pickle'
if SMALL:
	trainID = pickle.load(open("CMN_wav2vec2/IEMOCAP/data/trainID_new_filtered.pkl",'rb'), encoding="latin1")
	testID = pickle.load(open("CMN_wav2vec2/IEMOCAP/data/testID_new_filtered.pkl",'rb'), encoding="latin1")
else:
	trainID = pickle.load(open("CMN_wav2vec2/IEMOCAP/data/trainID_new.pkl",'rb'), encoding="latin1")
	testID = pickle.load(open("CMN_wav2vec2/IEMOCAP/data/testID_new.pkl",'rb'), encoding="latin1")
trainID, valID = model_selection.train_test_split(trainID, test_size=.2, random_state=1227)

audio_emb = pickle.load(open(AUDIO_EMBEDDINGS, 'rb'), encoding="latin1")



df = pd.DataFrame.from_dict(audio_emb, orient='index')
df = df.rename(columns={0: 'feature_array'})
transcripts, labels, own_historyID, other_historyID, own_historyID_rank, other_historyID_rank = pickle.load(open("CMN_wav2vec2/IEMOCAP/data/dataset.pkl",'rb'), encoding="latin1")

# choose the desired method: 'mean',  'max', 'mean_max', 'no_pooling'
METHOD = 'mean'
# choose the desired model: 'fcn', 'none'
EXTRACTION = 'fcn'

def mean_pooling(tuple):
    sum = np.sum(tuple, axis=0)
    return sum/len(tuple)
def max_pooling(tuple):
    return np.max(tuple, axis=0)

# if METHOD == 'mean':
#     df['feature_array'] = df['feature_array'].apply(mean_pooling)

# if METHOD == 'max':
#     df['feature_array'] = df['feature_array'].apply(max_pooling)

# if METHOD == 'mean_max':
#     df['feature_array_mean'] = df['feature_array'].apply(mean_pooling)
#     df['feature_array_max'] = df['feature_array'].apply(max_pooling)
#     df['feature_array'] = df.apply(lambda x: np.concatenate((x['feature_array_mean'], x['feature_array_max'])), axis=1)
#     df.drop(['feature_array_mean', 'feature_array_max'], axis=1, inplace=True)

    
indices = df.index
# filter 'xxx', 'sur', 'oth', 'dis', 'fea' from df out
todrop_values = ['xxx', 'sur', 'oth', 'dis', 'fea']
todrop_keys = [key for key, value in labels.items() if value in todrop_values]
dict_filtered_out = {}
common_indexes = set(df.index).intersection(todrop_keys)
df = df.drop(common_indexes)
#filter out labels
for key in todrop_keys:
    if key in labels:
        del labels[key]
# create dict for empty arrays
for key in todrop_keys:
    if key in df.index:
        dict_filtered_out[key] = np.zeros(100)

desired_shape = (400, df['feature_array'][0].shape[1])
print(desired_shape)

# Function to pad or trim the array to the desired shape
def resize_array(arr):
    resized_arr = np.zeros(desired_shape)
    resized_arr[:arr.shape[0], :arr.shape[1]] = arr[:desired_shape[0], :desired_shape[1]]
    resized_arr = resized_arr.ravel()
    return resized_arr

df['feature_array'] = df['feature_array'].apply(resize_array)    


new_df = pd.DataFrame()

for i in range(df.shape[0]):
    row_df = pd.DataFrame(df['feature_array'][i])
    new_df = pd.concat([new_df, row_df], axis=1)


# Function to extract values from the array and create separate columns
def extract_values(row):
    return pd.Series(row[0])

# Apply the extract_values function to the column of arrays
new_df = df.apply(extract_values, axis=1)

# Concatenate the original DataFrame with the new DataFrame
df = pd.concat([df, new_df], axis=1)

# Drop the original column
df = df.drop('feature_array', axis=1)

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
model.add(Dense(100, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(len(labels.unique()), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

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
representations.update(dict_filtered_out)

# add missing representation
missing_dict = {"Ses03M_impro03_M001" : np.zeros(100)}
representations.update(missing_dict)

# Save the representations as a pickle file

with open('features/representations_wav2vec2_' + METHOD + '_' + EXTRACTION +'_new.pkl', 'wb') as f:
    pickle.dump(representations, f)