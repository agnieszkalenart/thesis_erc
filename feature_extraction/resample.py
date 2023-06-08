import librosa
import os
import soundfile as sf

FILES_PATH_OLD = "/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/MELD.Raw/meld_small/"
FILES_PATH_NEW= "/Users/agnieszkalenart/Documents/mannheim/master_thesis/thesis_erc/MELD.Raw/meld_small_resampled/"
SAMPLE_RATE = 16000

for f in os.listdir(FILES_PATH_OLD):
    if not f.startswith('.'):
        # Load the audio file
        y, sr = librosa.load(FILES_PATH_OLD + f, sr=None)
        # Resample the audio to a new sample rate
        y_resampled = librosa.resample(y, sr, SAMPLE_RATE)
        # Save the resampled audio to a new file
        sf.write(FILES_PATH_NEW + f, y_resampled, SAMPLE_RATE)