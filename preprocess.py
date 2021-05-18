import hyperparams as hp
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from text import text_to_sequence
import collections
from scipy import signal
import torch as t
import math
from utils import stim_event_dict

print('preprocess.py packages imported...')

class OpenMIIRDataset(Dataset):
    """Process the prepared OpenMIIR dataset."""

    def __init__(self, loaded_npz, root_dir):
        """
        Args:
            npz_file (string): Path to the .npz file with saved eeg signal data.
            root_dir (string): Directory with all the wavs.
        """
#         self.loaded_arrays = np.nditer(np.load(npz_file))
        self.loaded_arrays = loaded_npz
        self.array_indexes = self.loaded_arrays.files
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __len__(self):
        return len(self.array_indexes)

    def get_wav_path(self, array_index):
        exp_id = array_index.split('_')[0]
        stimulus_id = get_stimulus_id(int(exp_id))
        wav_filename = stim_event_dict[stimulus_id]
        wav_filepath = os.path.join(self.root_dir, wav_filename)
        return wav_filepath
        
    def __getitem__(self, idx):
        wav_name = self.get_wav_path(self.array_indexes[idx])
        # print('wav_name', wav_name)
        # eeg_array = self.loaded_arrays[str(self.array_indexes[idx])]
        try:
            eeg_array = self.loaded_arrays[self.array_indexes[idx]]
        except:
            print('error ', idx, ' || ',self.array_indexes[idx])
            pass
        mel = np.load(wav_name[:-4] + '.pt.npy')
        mel_input = np.concatenate([np.zeros([1, hp.num_mels], np.float32), mel[:-1,:]], axis=0)
        eeg_signal_dimensions = eeg_array.shape
        eeg_signal_length = eeg_array.shape[1]
        # text_length = len(text)
        pos_eeg_signal= np.arange(1, eeg_signal_length + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)

        sample = {
            'eeg_array': eeg_array, 
            'mel': mel, 
            'mel_input':mel_input, 
            'pos_mel': pos_mel, 
            'pos_eeg_signal': pos_eeg_signal,
            'eeg_signal_dimensions': eeg_signal_dimensions, 
            'eeg_signal_length': eeg_signal_length
        }

        return sample    

def collate_fn_transformer(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        eeg_array = [d['eeg_array'] for d in batch]
        mel = [d['mel'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_eeg_signal= [d['pos_eeg_signal'] for d in batch]
        eeg_signal_length = [d['eeg_signal_length'] for d in batch]
        
        eeg_array = [i for i,_ in sorted(zip(eeg_array, eeg_signal_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(zip(mel, eeg_signal_length), key=lambda x: x[1], reverse=True)]
        mel_input = [i for i, _ in sorted(zip(mel_input, eeg_signal_length), key=lambda x: x[1], reverse=True)]
        pos_eeg_signal = [i for i, _ in sorted(zip(pos_eeg_signal, eeg_signal_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, eeg_signal_length), key=lambda x: x[1], reverse=True)]
        eeg_signal_length = sorted(eeg_signal_length, reverse=True)
        # PAD sequences with largest length of the batch
        eeg_array = _prepare_eeg_data(eeg_array)
        mel = _pad_mel(mel)
        mel_input = _pad_mel(mel_input)
        pos_mel = _prepare_data(pos_mel).astype(np.int32)
        pos_eeg_signal = _prepare_data(pos_eeg_signal).astype(np.int32)

        print('eeg prep arr shape:  ', eeg_array.shape)
        print('mel prep arr shape:  ', mel.shape)
        print('mel_input prep arr shape:  ', mel_input.shape)
        print('pos_mel prep arr shape:  ', pos_mel.shape)
        print('pos_eeg prep arr shape:  ', pos_eeg_signal.shape)
        # print('eeg prep arr shape:  ', eeg_array)

        return t.FloatTensor(eeg_array), t.FloatTensor(mel), t.FloatTensor(mel_input), t.LongTensor(pos_eeg_signal), t.LongTensor(pos_mel), t.LongTensor(eeg_signal_length)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

class PostDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.loc[idx, 0]) + '.wav'
        mel = np.load(wav_name[:-4] + '.pt.npy')
        mag = np.load(wav_name[:-4] + '.mag.npy')
        sample = {'mel':mel, 'mag':mag}

        return sample
    
def collate_fn_postnet(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        mel = [d['mel'] for d in batch]
        mag = [d['mag'] for d in batch]
        
        # PAD sequences with largest length of the batch
        mel = _pad_mel(mel)
        mag = _pad_mel(mag)

        return t.FloatTensor(mel), t.FloatTensor(mag)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))


def _pad_eeg_data(input_array, length):
    print('input_array shape: ', input_array.shape[1])
    _pad = 0
    return np.stack([np.pad(x, (0, length - input_array.shape[1]), mode='constant', constant_values=_pad) for x in input_array])

def _prepare_eeg_data(inputs):
    # note: dont need to calculate max for all shapes
    #       as the input list is already sorted by length
    max_len = max((x.shape[1] for x in inputs))
    print('at _prepare_eeg_data() within collate_fn_transformer()')
    print('max length: ', max_len)
    return np.stack([_pad_eeg_data(x, max_len) for x in inputs])

def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, hp.outputs_per_step - (timesteps % hp.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

# def get_dataset():
#     print('at get_dataset()')
#     # return LJDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))
#     return OpenMIIRDataset(os.path.join(hp.data_path, hp.npz_filename), os.path.join(hp.data_path, 'processed_wavs'))

def get_dataset():
    print('at get_dataset()')
    loaded_npz = np.load(os.path.join(hp.data_path, hp.npz_filename))
    # print(loaded_npz.files[0])
    # return LJDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))
    return OpenMIIRDataset(loaded_npz, os.path.join(hp.data_path, 'processed_wavs'))



def get_post_dataset():
    return PostDatasets(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

def get_stimulus_id(event_id):
    if event_id < 1000:
        return int(event_id / 10)
    else:
        return event_id




# --------------------------------


# class LJDatasets(Dataset):
#     """LJSpeech dataset."""

#     def __init__(self, csv_file, root_dir):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the wavs.

#         """
#         self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
#         self.root_dir = root_dir

#     def load_wav(self, filename):
#         return librosa.load(filename, sr=hp.sample_rate)

#     def __len__(self):
#         return len(self.landmarks_frame)

#     def __getitem__(self, idx):
#         wav_name = os.path.join(self.root_dir, self.landmarks_frame.loc[idx, 0]) + '.wav'
#         text = self.landmarks_frame.loc[idx, 1]

#         text = np.asarray(text_to_sequence(text, [hp.cleaners]), dtype=np.int32)
#         mel = np.load(wav_name[:-4] + '.pt.npy')
#         mel_input = np.concatenate([np.zeros([1,hp.num_mels], np.float32), mel[:-1,:]], axis=0)
#         text_length = len(text)
#         pos_text = np.arange(1, text_length + 1)
#         pos_mel = np.arange(1, mel.shape[0] + 1)

#         sample = {'text': text, 'mel': mel, 'text_length':text_length, 'mel_input':mel_input, 'pos_mel':pos_mel, 'pos_text':pos_text}
#         return sample


# def collate_fn_transformer(batch):

#     # Puts each data field into a tensor with outer dimension batch size
#     if isinstance(batch[0], collections.Mapping):

#         text = [d['text'] for d in batch]
#         mel = [d['mel'] for d in batch]
#         mel_input = [d['mel_input'] for d in batch]
#         text_length = [d['text_length'] for d in batch]
#         pos_mel = [d['pos_mel'] for d in batch]
#         pos_text= [d['pos_text'] for d in batch]
        
#         text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
#         mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
#         mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
#         pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
#         pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
#         text_length = sorted(text_length, reverse=True)
#         # PAD sequences with largest length of the batch
#         text = _prepare_data(text).astype(np.int32)
#         mel = _pad_mel(mel)
#         mel_input = _pad_mel(mel_input)
#         pos_mel = _prepare_data(pos_mel).astype(np.int32)
#         pos_text = _prepare_data(pos_text).astype(np.int32)


#         return t.LongTensor(text), t.FloatTensor(mel), t.FloatTensor(mel_input), t.LongTensor(pos_text), t.LongTensor(pos_mel), t.LongTensor(text_length)

#     raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
#                      .format(type(batch[0]))))
#  