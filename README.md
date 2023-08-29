# Welcome to the ERC Repository!
### Agnieszka Lenart
This repository contains code used for Master's thesis "Audio and Textual Feature Extraction for Emotion Recognition in Conversations" written at the Chair of Artificial Intelligence at the University of Mannheim.

## Contents of this repository:

#### Top-level directory
    .
    ├── CMN                     # Files to run CMN model 
    ├── ICON                    # Files to run ICON model 
    ├── feature_extraction      # Files for extraction of text and audio features
    ├── features                # Pickle files with extracted features
    ├── pre-processing          # Files to preprocess IEMOCAP Dataset
    ├── .gitignore
    ├── README.md
    └── requirements.txt

#### Feature extraction methods
    .
    ├── ...
    ├── feature_extraction              # Directory with files for extraction of the following features:
    │   ├── extract_bert                # BERT-based text features
    │   ├── extract_fasttext            # FastText text features
    │   └── extract_opensmile           # OpenSMILE audio features
    │   └── extract_opensmile_wav2vec2  # OpenSMILE and Wav2Vec2 audio features
    │   └── extract_wav2vec2            # Wav2Vec2 audio features
    └── ...
#### Features descriptions
    .
    ├── ...
    ├── features                                          
    │   ├── audio_opensmile_representations.pkl                 # OpenSMILE representations 
    │   ├── audio_opensmile.pickle                              # OpenSMILE features
    │   |── audio_wav2vec2_opensmile_representations_200.pkl    # OpenSMILE + Wav2Vec2 representations of dimension 200
    │   |── audio_wav2vec2_opensmile_representations_768.pkl    # OpenSMILE + Wav2Vec2 representations of dimension 768
    |   |── audio_wav2vec2_representations_max_fcn.pkl          # Wav2Vec2 max-pooling representations       
    |   |── audio_wav2vec2_representations_mean_fcn.pkl         # Wav2Vec2 mean-pooling representations  
    |   |── audio_wav2vec2_representations_mean_max_fcn.pkl     # Wav2Vec2 max- and mean-pooling representations  
    |   |── audio_wav2vec2_representations_mean_none.pkl        # Wav2Vec2 mean-pooling 
    |   |── audio_wav2vec2.pickle                               # Wav2Vec2 features
    |   |── text_bert_distilbert_representations_200.pickle     # DistilBERT representations
    |   |── text_bert_distilbert.pickle                         # DistilBERT features
    |   |── text_bert_roberta.pickle                            # RoBERTa features
    │   └── text_fasttext.pickle                                # FastText representations
    └── ...
Files in the directories that do not have the word 'representation' extract features and with it in the name first extract features and then use a fully connected network to learn a latent representation of these features that is smaller than the original dimension of the features.

  
## How to run experiments?
1. Install all the required Python packages by:
```
pip install -r requirements.txt
```
2. Download features and put them in the folder 'features'.
3. To run CMN with DistilBERT and Wav2Vec2 (with mean-pooling) features, run the file 'CMN/train_iemocap.py'.
   To run ICON with DistilBERT and Wav2Vec2 (with mean-pooling) features, run the file 'ICON/train_iemocap.py'.

## How to run experiments with custom features?
To run CMN with custom features:
1. Change the path to selected features (description in the section below) in 'CMN/IEMOCAP/utils_cmn.py'.
2. If necessary, change the TEXT_DIM and AUDIO_DIM in 'CMN/IEMOCAP/utils_cmn.py'.
3. Run the file 'CMN/train_iemocap.py'.

To run ICON with custom features:
1. Change the path to selected features (description in the section below) in 'ICON/IEMOCAP/utils.py'.
2. If necessary, change the TEXT_DIM and AUDIO_DIM in 'CMN/IEMOCAP/utils.py'.
3. Run the file 'ICON/train_iemocap.py'.



## How to extract features on your own?
1. Download IEMOCAP Dataset and put it in the '/data' folder.
2. Run the 'pre-processing/pre-processing.py'.
3. To extract BERT text features:
   - run files in 'feature_extraction/extract_bert'
4. To extract BERT text features:
   - run files in 'feature_extraction/extract_fasttext'
5. To extract OpenSMILE audio features:
   - install [OpenSMILE software](https://github.com/audeering/opensmile/releases/tag/v3.0.0)
   - update the file 'feature_extraction/extract_opensmile/feature_extractor_OS.py' so it can access OpenSMILE installed on your device
   - run 'feature_extraction/extract_opensmile/generate_opensmile_features.py'
6. To extract Wav2Vec2 features:
  - download [Wav2Vec2 Base (no fine-tuning)](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md) and put it in the folder 'pre-trained_models'
  - run 'feature_extraction/extract_wav2vec2/generate_wav2vec2_features.py'


## Sources:
- [Emotion Recognition in Conversations GitHub](https://github.com/declare-lab/conv-emotion/tree/master)
- [Official implementation of 'Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings'](https://github.com/habla-liaa/ser-with-w2v2/tree/master)

