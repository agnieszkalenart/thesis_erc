import pickle
import numpy as np


wav2vec2_representations = pickle.load(open('features/audio_wav2vec2_representations_mean_max_fcn.pkl', 'rb'))
opensmile_representations = pickle.load(open('features/audio_opensmile_representations.pkl', 'rb'))

# Concatenate the representations
new_dict = {}  
intersection_keys = set(wav2vec2_representations.keys()).intersection(opensmile_representations.keys())

for key in intersection_keys:
    new_dict[key] = np.concatenate((wav2vec2_representations[key], opensmile_representations[key]))

# Save the new dictionary
pickle.dump(new_dict, open('features/audio_wav2vec2_opensmile_representations_200.pkl', 'wb'))
