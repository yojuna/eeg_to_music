# Audio
num_mels = 80
# num_freq = 1024
n_fft = 2048
sr = 22050
# frame_length_ms = 50.
# frame_shift_ms = 12.5
preemphasis = 0.97
frame_shift = 0.0125 # seconds
frame_length = 0.05 # seconds
hop_length = int(sr*frame_shift) # samples.
win_length = int(sr*frame_length) # samples.
n_mels = 80 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20
hidden_size = 256
# embedding_size = 512
max_db = 100
ref_db = 20
    
n_iter = 60
# power = 1.5
outputs_per_step = 1

epochs = 10000
lr = 0.001
save_step = 2000
image_step = 500

# batch_size > 8 => OOM error for my system.
batch_size = 2

embedding_size = 69

# cleaners='english_cleaners'

# data_path = './data/LJSpeech-1.1'
# data_path = '/home/archeron/dev/data/eeg_to_music_data/LJSpeech-1.1'
data_path = '/home/archeron/dev/data/eeg_to_music_data'
npz_filename = 'P01_extracted.npz'
checkpoint_path = data_path + '/checkpoint'
sample_path = data_path + '/samples'

checkpoint_file_transformer = data_path + '/checkpoint/checkpoint_transformer_eeg_2000.pth.tar'