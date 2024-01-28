#https://github.com/zimmerrol/emomatch-pytorch/blob/master/src/preprocessing/convert_mp4_to_hdf5.py
from __future__ import print_function
from __future__ import absolute_import

#import time
import tqdm
import os
import numpy as np
from glob import glob
import h5py as h5

import librosa
import cv2

import warnings
warnings.filterwarnings('ignore')


#***********************************************************AUDIO PROCESSING*****************************************#  
#https://github.com/qiuqiangkong/dcase2019_task3/tree/master/utils
audio_duration = 30   
sample_rate = 22050 #16000
window_size = 2048 # or frame size
hop_size = 512     # So that there are 64 frames per second
mel_bins = 128

frames_per_sec = sample_rate // hop_size 
frames_num = frames_per_sec * audio_duration #Total temporal frames = 64*10 =640    
audio_samples = int(sample_rate * audio_duration)


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def read_audio(audio_path, target_fs=None):
    #(audio, fs) = soundfile.read(audio_path)
    audio, fs = librosa.load(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]

#***********************************************************AUDIO PROCESSING *****************************************#  
def compute_mel_spec(audio_name):    
    # Compute short-time Fourier transform
    stft_matrix = librosa.core.stft(y=audio_name, n_fft=window_size, hop_length=hop_size, window=np.hanning(window_size), center=True, dtype=np.complex64, pad_mode='reflect').T
    melW = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins).T
    # Mel spectrogram
    mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, melW)
    # Log mel spectrogram
    logmel_spc = librosa.core.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=None)
    logmel_spc = np.expand_dims(logmel_spc, axis=0)
    logmel_spc = logmel_spc.astype(np.float32)
    #print("The shape of logmel_spc:", logmel_spc.shape)
    logmel_spc = np.array(logmel_spc).transpose((2, 1, 0))
    logmel_spc = logmel_spc[0 : frames_num]
    
    return logmel_spc

#***********************************************************Audio PROCESSING START*****************************************#  
def music_feature(audio_path):
    # Read audio
    (audio, _) = read_audio(audio_path=audio_path, target_fs=sample_rate)
    #multichannel_audio, fs = read_multichannel_audio(audio_path=audio_path, target_fs=sample_rate)
    
    if audio.shape[0] < audio_samples: 
        #audio = repeat_array(audio, audio_samples) #Repeat the same audio until 10sec length
        audio = pad_truncate_sequence(audio, audio_samples) ## Pad or truncate audio recording
    
    #If audio length is more than 10second then clip it to 10Sec
    elif audio.shape[0] > audio_samples:
        audio = audio[int((audio.shape[0]-audio_samples)/2):int((audio.shape[0]+audio_samples)/2)]
        
    logmel_spc = compute_mel_spec(audio)
    
    return logmel_spc
    

#***********************************************************VIDEO PROCESSING START*****************************************#  
def video3d_frames(filename, width, height, depth, color=False, skip=True):
    cap = cv2.VideoCapture(filename)
    nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    if skip:
        frames = [x * nframe / depth for x in range(depth)]
    else:
        frames = [x for x in range(depth)]
    
    framearray = []
    for i in range(depth):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, frame = cap.read()
        frame = cv2.resize(frame, (height, width), interpolation = cv2.INTER_CUBIC)            
        if color:
            framearray.append(frame)
        else:
            framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()

    X = np.array(framearray)    #vid3d.video3d(video_dir, color=color, skip=skip)

    if color:
        return np.array(X).transpose((1, 2, 0, 3))
    else:
        return np.array(X).transpose((1, 2, 0, 0))
    
#***********************************************************VIDEO PROCESSING END*****************************************# 

#******************** Shift to Other file extenson with same name ****************************************#

def audio_to_video_path_real(audio_path):
        path_names = audio_path.split('\\')
#         print(path_names)
        if path_names[-2] != 'Audio':
#             print(path_names[-2])
            raise Exception('invalid audio-video path conversion')

        path_names[-2] = 'Video'
#         print(os.path.splitext('\\'.join(path_names))) 
#         print('The audio path real:', path_names)
        ext = r'.mp4'
        v_path = os.path.splitext('\\'.join(path_names))[0] + ext
        print(v_path)
#         return glob(os.path.splitext('\\'.join(path_names))[0] + ".*")[0]
        return v_path

#********************************************* LABEL THE GENRE *******************************************#
# Prepare multi-hot-encoded-labels for the various genres
def multi_hot_encoded_labels(set_of_genre):
    col_names =  ['African','Arabic','Chinese','French','Indian','Mongolian',
                   'Nepali','Spanish','Positive','Neutral','Negative']    
    row=[]    
    for i in range(len(col_names)):
        found = 0
        for j in range (len(set_of_genre)):
            if (set_of_genre[j]==col_names[i]):
                found = 1
                break
        row.append(found)
    return row


def main(mp4_directory, hdf5_directory, utterance):
    
    color = True #use RGB image (True) or grayscale image (False)
    skip = True #Get frames at interval(True) or contenuously (False)
    #depth_fast = 64 #number of frames to use 
    img_rows, img_cols, frames = 128, 128, 64

    hdf5_file_mp4 = hdf5_directory + os.path.sep + utterance[len(mp4_directory):] 
    base_hdf5_file = os.path.splitext(hdf5_file_mp4)[0]  #Remove the .mp4 extension
    #utterance_hdf5_file = base_hdf5_file + '.h5'
    utterance_npz_file = base_hdf5_file + '.npz'
    
    if os.path.exists(utterance_npz_file):
        return True

    os.makedirs(os.path.dirname(utterance_npz_file), exist_ok=True)
    
    #********************** For target generation ***********************************#
    # encode the class labels using one hot coding (use index as label)
    
    #For multi-label one-hot
    multi_ganre = []
    
    #G:\RUNNING_PROJECTS\ASHIM_MultiCulture_Emo\Revised_Data\Audio\Test_Data\African\Negative\audio\_African_Music_111.wav

    #Making Folder as Label
    dir_sub, _ = os.path.split(utterance)
    basepath, _ = os.path.split(dir_sub)
    Country_emo, pnn_label = os.path.split(basepath)
    _, geo_label = os.path.split(Country_emo)
    multi_ganre.extend([geo_label, pnn_label])
    print("The multi_ganre is:", multi_ganre)
    
    label_onehot_multi = multi_hot_encoded_labels(multi_ganre)
    print("The label_onehot_multi is:", label_onehot_multi)
      
    
    #**************************Mix Audio processing***************************************************#
    logmel_spc  = music_feature(utterance)
    #print("The mel_phasegram shape is:",mel_phasegram.shape)
    
    #*********Video Features ********************#
    video_path = audio_to_video_path_real(utterance)
    video3D = video3d_frames(video_path, width=img_rows, height=img_cols, depth=frames, color=color, skip=skip)
    #print("The video_data shape is:",video_data.shape)
    
    np.savez(utterance_npz_file, logmel_spc=logmel_spc, video3D=video3D, target = label_onehot_multi)
    
    '''
    #Use HDF5 file format (Increase some data size)
    try:
        with h5.File(utterance_hdf5_file, 'w') as hf:
            hf.create_dataset('mel_phasegram', data=mel_phasegram, dtype=np.float32)
            hf.create_dataset('hcqt_phasegram', data=hcqt_phasegram, dtype=np.float32)
            
            hf.create_dataset('video3D', data=video3D, dtype=np.float32)
            
            hf.create_dataset('target', data=target, dtype=np.float32)
            
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return False
    return True
    '''
if __name__ == '__main__':
    
    mp4_directory = 'F:\\Project_data\\Test_Data'
    hdf5_directory = 'F:\\Project_data\\Test_Out'
    
    if not os.path.exists(hdf5_directory):
        os.makedirs(hdf5_directory)
    
    
    for culture_dir in tqdm.tqdm(sorted(list(glob(os.path.join(mp4_directory, '*')))), position=0):
        # print(culture_dir)
        for Sent_dir in tqdm.tqdm(sorted(list(glob(os.path.join(culture_dir, '*')))), position=1, leave=False):
            # print(Sent_dir)
            for sub_dir in tqdm.tqdm(list(glob(os.path.join(Sent_dir, '*'))), position=2, leave=False):
                # print(sub_dir)
                for utterance in tqdm.tqdm(list(glob(os.path.join(sub_dir, '*.wav'))), position=3, leave=False):
                    print("The input utterance dir", utterance)
                    main(mp4_directory, hdf5_directory, utterance)       
