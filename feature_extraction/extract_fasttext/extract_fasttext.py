import fasttext
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Load the pre-trained FastText model
model = fasttext.load_model('cc.en.300.bin')

# Load the IEMOCAP dataset
df = pd.read_csv('data/iemocap.csv')
print(df.head())

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

# Function to extract words from a sentence
def extract_words(sentence):
    words = re.findall(r'\w+', sentence)  # Use regex to find all alphanumeric sequences
    return words

# Apply the function to create the 'words' column
df['words'] = df['utterances'].apply(extract_words)

# Check for maximum number of words in a sentence
max_words = max(df['words'].apply(len))

print('Maximum number of words in a sentence:', max_words)

# Check for average number of words in a sentence
avg_words = np.mean(df['words'].apply(len))

print('Average number of words in a sentence:', avg_words)

# Check for number of uttersances that have more than 30 words
print('Number of utterances that have more than 30 words:', sum(df['words'].apply(len) > 30))


# Create FastText embeddings for each utterance
embeddings_column = []
for utterance in df['words']:
    embedding_matrix = []
    for word in utterance:           
        # Obtain the FastText embedding for the utterance
        embedding = model.get_sentence_vector(word).tolist()
        embedding_matrix.append(embedding)
    embeddings_column.append(embedding_matrix)


# Change the shape of the embeddings column to (30, 300)

# Cut off the extra words if the number of words in an utterance is more than 30
for i in embeddings_column:
    if len(i) > 30:
        i = i[:30]

# Pad the embeddings with zeros if the number of words in an utterance is less than 30
for i in embeddings_column:
    if len(i) < 30:
        for j in range(30 - len(i)):
            i.append(np.zeros(300, dtype="float32").tolist())

# check if all embeddings have the right shape
for i in range(len(embeddings_column)):
    if len(embeddings_column[i]) == 30:
        continue
    for j in embeddings_column[i]:
        if len(j) == 300:
            continue
        else:
            print('error', i, j)

# flatten the embeddings column
# def flatten(l):
#     return [item for sublist in l for item in sublist]

# embeddings_column = [flatten(item) for item in embeddings_column]


emb_tensor = tf.constant(embeddings_column)

# labels = tf.convert_to_tensor(labels)

# features = tf.convert_to_tensor(embeddings_column[:1363])

# print(type(embeddings_column))
# print(embeddings_column[0])
# print(type(embeddings_column[0]))
# print(len(embeddings_column[0]))
# print(type(embeddings_column[0][0]))
# print(len(embeddings_column[0][0]))
# print(type(embeddings_column[0][0][0]))



# indices = np.asarray(df['indices'])
# features = np.asarray(embeddings_column)
# labels = np.asarray(df['labels'])

indices = df['indices']
features = pd.DataFrame(embeddings_column)
print(features.head())
labels = df['labels']


f= len(features)
l = len(labels)
i = len(indices)

# Split the data into training, validation and testing sets
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, indices, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(X_train, y_train, indices_train, test_size=0.2, random_state=42)


# Define the model architecture
input_shape = (30, 300)
num_filters = 50

# Define the input layer
inputs = tf.keras.Input(shape=input_shape)

# Convolutional layers with different filter sizes
filter_sizes = [3, 4, 5]
conv_outputs = []

for filter_size in filter_sizes:
    conv_layer = tf.keras.layers.Conv1D(num_filters, filter_size, activation='relu',  padding='same')(inputs)
    conv_outputs.append(conv_layer)

# Concatenate the convolutional outputs
concatenated = tf.keras.layers.concatenate(conv_outputs, axis=-1)

# Max pooling
pooled = tf.keras.layers.MaxPooling1D(2)(concatenated)

# Flatten the pooled output
flattened = tf.keras.layers.Flatten()(pooled)

# Fully connected layer
dense = tf.keras.layers.Dense(100, activation='relu')(flattened)

# Output layer
outputs = tf.keras.layers.Dense(6, activation='softmax')(dense)

# Create the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
