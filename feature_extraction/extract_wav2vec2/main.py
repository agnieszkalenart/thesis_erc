from IEMOCAP_reader import IEMOCAPReader
from dataframe_processing import Relabel, Filter, LabelEncoder, OutputMerger
from feature_extractor import Wav2Vec2Embeddings, OpensmileExtractor, Spectrogram
# from splits import Split, GroupSplit, LeaveOneOut
from normalize import NormalizationStatistics
# from generators import BatchGenerator
# from dienen_model import DienenModel
from data_processors import PadDP, ToNumpyDP, SqueezeDP, NormalizeDP
import pandas as pd
import pickle
import numpy as np
import ast

DATA_PATH = '/Users/agnieszkalenart/Documents/mannheim/master_thesis/IEMOCAP'

#read data
iemocap_reader = IEMOCAPReader()
df_data = iemocap_reader.process(data_path=DATA_PATH, min_duration=0.0, min_sad_frames_duration=0, sample=None,compute_speech_rate=True)

df_data.to_csv('/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/features/iemocap.csv')
# print(df_data.loc[['Ses01F_impro04_M010']])
#pre-process data
#relabel
# relabel = Relabel()
# relabels = [{'column': 'emotion', 'old_name': 'excited', 'new_name': 'happiness'}]
# df_data = relabel.process(enable=True, data=df_data, relabels=relabels)
#filter
# include_values =  ['anger','happiness','sadness','neutral']
# column = 'emotion'
# filter = Filter()
# df_data = filter.process(df_data=df_data, column_filter=column, include_values=include_values)
#label encoder
# column = 'emotion'
# new_column = 'classID'
# label_encoder = LabelEncoder()
# df_data, possible_labels = label_encoder.process(df_data=df_data, column=column, new_column=new_column)



#small data
# df_data= df_data.head(5)

# #create wav2vec2 features
# wav2vec2_model_path = '/Users/agnieszkalenart/Documents/mannheim/master_thesis/wav2vec2/wav2vec_small.pt'
# wav2vec2_dict_path = None
# normalization_axis = 1
# wav2vec2_padding_axis = 1
# max_size = 240000
# mode  = 'sequence'
# output_column = 'wav2vec2'
# layer = 'output'
# save_feature_files = True

# wav2vec2 = Wav2Vec2Embeddings()
# model, cfg, task = wav2vec2.build_wav2vec_model(model_path=wav2vec2_model_path)
# data = wav2vec2.process(data = df_data,
#                 model_path=wav2vec2_model_path,
#                  max_size=max_size,
#                  mode= mode,
#                  output_column=output_column,
#                  layer = layer,
#                  save_feature_files= save_feature_files)

# data['wav2vec2'] = data['wav2vec2'].apply(lambda x: x.tolist())


# #create opensmile features
# opensmile = OpensmileExtractor()
# data = opensmile.process(data=df_data, output_column='opensmile')


# #create spectrogram features
# spectrogram = Spectrogram()
# data = spectrogram.process(data=data, output_column='spectrogram')
# data['spectrogram'] = data['spectrogram'].apply(lambda x: x.tolist())


# #preprocess WAV2VEC2 sequence
# data.to_csv('iemocap_with_features_full.csv')

# # to opensmile pickle
# data = pd.read_csv('/Users/agnieszkalenart/Documents/mannheim/master_thesis/iemocap_with_features_full.csv')
# data['opensmile'] = data['opensmile'].apply(lambda x: ast.literal_eval(x))
# data['opensmile'] = data['opensmile'].apply(np.array)
# data['opensmile'] = data['opensmile'].apply(lambda x: x.reshape(6373,))
# opensmile_dict = dict(zip(data['Unnamed: 0'], data['opensmile']))

# with open('cmn_audio_opensmile.pickle', 'wb') as file:
#     pickle.dump(opensmile_dict, file)


# PadDP = PadDP()
# data = PadDP.process(data=data, col_in = 'wav2vec2', col_out = 'wav2vec2', max_length=400)
# SqueezeDP = SqueezeDP()
# data = SqueezeDP.process(data=data, col_in = 'wav2vec2', col_out = 'wav2vec2', axis=0)
# Normalizer = NormalizeDP
# normalization_statistics = NormalizationStatistics()
# statistics = normalization_statistics.process(data=data, normalization_by='subject', column ='wav2vec2', axis=normalization_axis)
# data = Normalizer.process(data=data, col_in = wav2vec2, col_out = wav2vec2, statistics=statistics)
# ToNumpyDP = ToNumpyDP()
# data = ToNumpyDP.process(data=data, col_in = wav2vec2)

# data.to_csv('wav2vec2_emb.csv')


# # Preprocess for cmn 
# data['wavfile'] = data['wavfile'].str.replace('.wav', '')
# df_data['wav2vec2'] = data['wav2vec2'].apply(lambda x: [item for sublist in x for item in sublist])
# data['wav2vec2'] = data['wav2vec2'].apply(np.array)
# cmn_emb = dict(zip(data['wavfile'], data['wav2vec2']))

# # Save the dictionary as a pickle file
# with open('cmn_audio_wav2vec2.pickle', 'wb') as file:
#     pickle.dump(cmn_emb, file)

# #leave-one-out
# data = data 
# group_col = 'session'
# leave_one_out = LeaveOneOut()
# train_folds, test_folds = leave_one_out.process(data, group_col = group_col)

# #split data - PartitionDevTest
# split = Split()
# split_col = 'session'
# group_outputs = {'dev': train_folds, 'test': test_folds}
# PartitionDevTest = split.process(data=data, split_column=split_col, group_outputs=group_outputs)

# #split data - PartitionTrainValColumn
# group_split = GroupSplit()
# group_column = 'session'
# out_column = 'partition'
# splits = {'train': 0.9, 'val': 0.1}
# #to do: check element of tuple which is dev and which is test
# PartitionTrainValColumn = group_split.process(data=PartitionDevTest[0], group_column=group_column, out_column=out_column, splits=splits)

# #split data - PartitionTrainVal
# split = Split()
# split_col = 'partition'
# PartitionTrainVal = split.process(data=PartitionTrainValColumn, split_column=split_col)

# #merge output
# outputs = {'train': PartitionTrainVal[0], 'validation': PartitionTrainVal[1], 'test': PartitionDevTest[1]}
# output_merger = OutputMerger()
# merged_data = output_merger.process(outputs=outputs)

# ### normalize: NormalizationStatistics
# normalizer = NormalizationStatistics()
# train_statistics = normalizer.process(data=merged_data[0], normalization_by='subject', column ='wav2vec2', axis=normalization_axis)
# val_statistics = normalizer.process(data=merged_data[1],  normalization_by='subject', column ='wav2vec2', axis=normalization_axis)
# test_statistics = normalizer.process(data=merged_data[2],  normalization_by='subject', column ='wav2vec2', axis=normalization_axis)

# #batch generation
# batch_generator = BatchGenerator()
# batch_x = ['audio','mask']
# batch_y = 'targets'
# train_batch = batch_generator.process(data = merged_data[0], batch_task=None, batch_x = batch_x, batch_y = batch_y, extra_data=None, shuffle=True, batch_size=16, seed=1234)
# val_batch = batch_generator.process(data = merged_data[1], batch_task=None, batch_x = batch_x, batch_y = batch_y, extra_data=None, shuffle=True, batch_size=16, seed=1234)
# test_batch = batch_generator.process(data = merged_data[2], batch_task=None, batch_x = batch_x, batch_y = batch_y, extra_data=None, shuffle=True, batch_size=16, seed=1234)

# #model 
# ## from dinen import Model???

# print(data)
# # print(possible_labels)
# # print(df_data.head(30))
# #print(df_data['emotion'].head(1000))
# #print(df_data.loc[df_data['emotion'] == "happiness"]['emotion'])