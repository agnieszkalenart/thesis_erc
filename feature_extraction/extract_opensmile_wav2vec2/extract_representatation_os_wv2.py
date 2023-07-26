import pickle
import numpy as np


wav2vec2_representations = pickle.load(open('features/representations_wav2vec2_mean_max_fcn.pkl', 'rb'))
opensmile_representations = pickle.load(open('features/representations_opensmile.pkl', 'rb'))

# Concatenate the representations
new_dict = {}  
intersection_keys = set(wav2vec2_representations.keys()).intersection(opensmile_representations.keys())

for key in intersection_keys:
    new_dict[key] = np.concatenate((wav2vec2_representations[key], opensmile_representations[key]))

# Save the new dictionary
pickle.dump(new_dict, open('features/representations_wav2vec2_opensmile.pkl', 'wb'))