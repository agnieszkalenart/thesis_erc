import pickle
import numpy as np
import pandas as pd
from sklearn import model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

CROSSVALIDATION = False

AUDIO_EMBEDDINGS = '/Users/agnieszkalenart/Documents/mannheim/master_thesis/cmn_audio_opensmile_new.pickle'
trainID = pickle.load(open("/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/CMN_wav2vec2/IEMOCAP/data/trainID_new_filtered.pkl",'rb'), encoding="latin1")
testID = pickle.load(open("/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/CMN_wav2vec2/IEMOCAP/data/testID_new_filtered.pkl",'rb'), encoding="latin1")
valID,_ = model_selection.train_test_split(trainID, test_size=.2, random_state=1227)

audio_emb = pickle.load(open(AUDIO_EMBEDDINGS, 'rb'), encoding="latin1")
df = pd.DataFrame.from_dict(audio_emb, orient='index')
transcripts, labels, own_historyID, other_historyID, own_historyID_rank, other_historyID_rank = pickle.load(open("/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/CMN_wav2vec2/IEMOCAP/data/dataset.pkl",'rb'), encoding="latin1")

# filter 'xxx', 'sur', 'oth', 'dis', 'fea' from labels out
todrop_values = ['xxx', 'sur', 'oth', 'dis', 'fea']
todrop_keys = [key for key, value in labels.items() if value in todrop_values]
for key in todrop_keys:
    if key in labels:
        del labels[key]


# create embedding for filtered labels
dict_filtered_out = {}
for key in todrop_keys:
    if key in df.index:
        dict_filtered_out[key] = np.zeros(100)

# Find the common indexes
common_indexes = set(df.index).intersection(todrop_keys)
# Drop the specified indexes
df = df.drop(common_indexes)
for key in todrop_keys:
    if key in audio_emb:
        del audio_emb[key]

label_idx = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
labels_array = np.asarray([label_idx[labels[ID]] for ID in audio_emb.keys()])

# Split the data into features and labels
features = df
labels = pd.Series(labels_array)
indices = df.index

# trainID = [ID for ID in trainID if ID not in todrop_keys]
# testID = [ID for ID in testID if ID not in todrop_keys]
# valID = [ID for ID in valID if ID not in todrop_keys]

# Define the neural network model
def create_model():
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(100, activation='relu'))    
    model.add(Dense(len(labels.unique()), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if CROSSVALIDATION:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, indices, test_size=0.2, random_state=42)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)  

    # Standardize the feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)



    # Perform cross-validation
    num_folds = 10
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    cv_scores = []
    best_model = None
    best_score = 0

    for train_index, val_index in skf.split(X_train, y_train):
        train_features, val_features = X_train[train_index], X_train[val_index]
        train_labels, val_labels = y_train[train_index], y_train[val_index]
        
        model = create_model()
        model.fit(train_features, train_labels, epochs=20, batch_size=32, verbose=0)
        
        val_pred = model.predict(val_features)
        classes_x=np.argmax(val_pred,axis=1)
        fold_score = accuracy_score(val_labels, classes_x)

        if fold_score > best_score:
            best_score = fold_score
            best_model = model

        cv_scores.append(fold_score)

    # Print the cross-validation scores
    print('Cross-Validation Scores:', cv_scores)
    print('Average Accuracy:', np.mean(cv_scores))


else:
    # Split the data into training, validation and testing sets
    # X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, indices, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

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
    best_model = create_model()

    # Compile the model
    best_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    best_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predict the emotions on the testing set
predict_x = best_model.predict(X_test)
classes_x=np.argmax(predict_x,axis=1)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, classes_x)
print(f'Accuracy: {accuracy}')


# Create a new model that outputs the activations of the last layer
representation_model = Model(inputs=best_model.input, outputs=best_model.layers[-2].output)


# Extract the representations for each example
train_representations = representation_model.predict(X_train)
test_representations = representation_model.predict(X_test)

# Create dict representations
representations = {}
representations.update({idx: rep for idx, rep in zip(trainID, train_representations)})
if CROSSVALIDATION==False:
    val_representations = representation_model.predict(X_val)
    representations.update({idx: rep for idx, rep in zip(valID, val_representations)})
representations.update({idx: rep for idx, rep in zip(testID, test_representations)})
representations.update(dict_filtered_out)


# Save the representations as a pickle file
with open('/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/features/representations_opensmile.pkl', 'wb') as f:
    pickle.dump(representations, f)