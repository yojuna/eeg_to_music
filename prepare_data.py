import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from utils import get_spectrograms
import hyperparams as hp
import librosa
from utils import stim_event_dict
from preprocess import get_stimulus_id

class PrepareDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self, npz_file, root_dir):
        """
        Args:
            npz_file (string): Path to the .npz file with saved eeg signal data.
            root_dir (string): Directory with all the wavs.

        """
        self.loaded_arrays = np.load(npz_file)
        self.array_indexes = self.loaded_arrays.files
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def get_wav_path(self, array_index):
        exp_id = array_index.split('_')[0]
        stimulus_id = get_stimulus_id(int(exp_id))
        wav_filename = stim_event_dict[stimulus_id]
        wav_filepath = os.path.join(self.root_dir, wav_filename)
        return wav_filepath

    def __len__(self):
        return len(self.array_indexes)

    def __getitem__(self, idx):
        wav_name = self.get_wav_path(self.array_indexes[idx])       
        mel, mag = get_spectrograms(wav_name)
        
        np.save(wav_name[:-4] + '.pt', mel)
        np.save(wav_name[:-4] + '.mag', mag)

        sample = {'mel':mel, 'mag': mag}

        return sample
    
if __name__ == '__main__':
    dataset = PrepareDataset(os.path.join(hp.data_path, hp.npz_filename), os.path.join(hp.data_path, 'processed_wavs'))
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
