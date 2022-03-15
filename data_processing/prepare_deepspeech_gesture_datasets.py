import numpy as np
import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)
from audio_features import extract_melspec1
from sklearn.preprocessing import StandardScaler
import pyarrow
import lmdb as lmdb
import torch
import pickle
from model.vocab import Vocab

def fit_and_standardize(data):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler
    
def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled
    
def motion_to_dict(fulls, vid, text):
    """
    fulls: a list of [T, xxx + 1] - motion and phase
    style: a *number*
    meta: a dict, e.g. {"style": "angry", "content": "walk"}
    """
    output = []
    for full in fulls:
        motion = full[:, :]
        output.append({
            "motion": motion,
            "style": vid,
            "text": text 
        })
    return output

def divide_clip_bfa(input, window, window_step, divide):
    if not divide:  # return the whole clip
        t = ((input.shape[0]) // 4) * 4 + 4
        t = max(t, 12)
        return [input]
    windows = []
    for j in range(0, len(input) - window + 1, window_step):
        slice = input[j: j + window].copy()  # remember to COPY!!
        windows.append(slice)
    return windows


def train_data_use(lang_model, data_root, datapath, processed_dir):

    lmdb_env = lmdb.open(data_root, readonly=True, lock=False)

    motion_path = processed_dir + 'motion_feat.npy'
    speech_path = processed_dir + 'speech_feat.npy'
    vid_path = processed_dir + 'vid_feat.npz'
    text_path = processed_dir + 'text_feat.npz.npy'
    if os.path.exists(motion_path) and os.path.exists(speech_path) and os.path.exists(vid_path) and os.path.exists(text_path):
        mel_new = np.load(speech_path, allow_pickle=True)
        motion_new = np.load(motion_path, allow_pickle=True)
        vid_new =  np.load(vid_path, allow_pickle=True)['vid_all'].item()
        text_new = np.load(text_path, allow_pickle=True)
    else:
        with lmdb_env.begin() as txn:
            n_samples = txn.stat()['entries']

            mel_new = []
            text_new = []
            motion_new = []
            vid_all = []
            data_dict = {}
            for i in range(n_samples):
                key = '{:010}'.format(i).encode('ascii')
                sample = txn.get(key)
                sample = pyarrow.deserialize(sample)
                word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
                vid = aux_info['vid']
                duration = aux_info['end_time'] - aux_info['start_time']
                
                mel = extract_melspec1(audio, pose_resampling_fps, 16000)
                mel_new.append(mel)
                vec_seq = vec_seq.reshape(-1, 27)
                motion_new.append(vec_seq)
                vid_all.append(vid)

                def extend_word_seq(lang, words, end_time=None):
                    n_frames = 42
                    if end_time is None:
                        end_time = aux_info['end_time']
                    frame_duration = (end_time - aux_info['start_time']) / n_frames

                    extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
                    prev_idx = 0
                    for word in words:
                        idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                        if idx < n_frames:
                            extended_word_indices[idx] = lang.get_word_index(word[0])
                            # extended_word_indices[prev_idx:idx+1] = lang.get_word_index(word[0])
                            prev_idx = idx
                    return torch.Tensor(extended_word_indices).long()

                def words_to_tensor(lang, words, end_time=None):
                    indexes = [lang.SOS_token]
                    for word in words:
                        if end_time is not None and word[1] > end_time:
                            break
                        indexes.append(lang.get_word_index(word[0]))
                    indexes.append(lang.EOS_token)
                    return torch.Tensor(indexes).long()
                
                duration = aux_info['end_time'] - aux_info['start_time']

                sample_end_time = aux_info['start_time'] + duration * 42 / vec_seq.shape[0]

                # to tensors
                word_seq_tensor = words_to_tensor(lang_model, word_seq, sample_end_time)
                extended_word_seq = extend_word_seq(lang_model, word_seq, sample_end_time)
                text_new.append(extended_word_seq)

            
            vid_new = {"vid": vid_all}
            data_dict['vid_all'] = vid_new

            # np.savez_compressed(vid_path, **data_dict)

            mel_new = np.array(mel_new)
            motion_new = np.array(motion_new)
            text_new = np.array(text_new)
            # np.save(speech_path, mel_new)
            # np.save(motion_path, motion_new)

            print("Done.")

    train_inputs = []
    window = 42
    window_step = 10

    vid = vid_new['vid']
    for fidx in range(len(mel_new)):
        audio = mel_new[fidx]
        motion = motion_new[fidx]
        text = text_new[fidx]
        if audio.shape[0] >= window:
            if audio.shape[0] - motion.shape[0] > 3:
                print("wrong")
            else:
                audio = audio[:motion.shape[0]]

            raw = np.concatenate([audio, motion], axis=-1)

            train_clips = motion_to_dict(
                divide_clip_bfa(raw, window=window, window_step=window_step, divide=True), vid[fidx], text)

            train_inputs += train_clips

    return train_inputs

def convert_dir_vec_to_pose(vec):
    dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                     (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length
    vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    else:
        assert False

    return joint_pos



def build_vocab(name, cache_path, word_vec_path=None, feat_dim=None):
    with open(cache_path, 'rb') as f:
        lang_model = pickle.load(f)

    if word_vec_path is None:
        lang_model.word_embedding_weights = None
    elif lang_model.word_embedding_weights.shape[0] != lang_model.n_words:
        assert False

    return lang_model


if __name__ == "__main__":
    '''
    Converts bvh and wav files into features, slices in equal length intervals and divides the data
    into training, validation and test sets. Adding an optional style argument ("MG-R", "MG-V", "MG-H" or "MS-S") 
    will add features for style control.
    '''

    
    vocab_cache_path = '../ted_dataset/vocab_cache.pkl'
    lang_model = build_vocab('words', vocab_cache_path, 'data/fasttext/crawl-300d-2M-subword.bin',
                             300)

    fps = 30
    pose_resampling_fps = 15

    current_root = os.getcwd()
    parent_root = os.path.dirname(current_root)

    data_root_val = os.path.join(parent_root, 'ted_dataset/lmdb_val_cache')
    datapath_val = os.path.join(data_root_val, 'data.mdb')
    processed_dir_val = os.path.join(parent_root, 'ted_dataset/gesture_processed/val_')

    val_inputs = train_data_use(lang_model, data_root_val, datapath_val, processed_dir_val)

    data_root_test = os.path.join(parent_root, 'ted_dataset/lmdb_test_cache')
    datapath_test = os.path.join(data_root_test, 'data.mdb')
    processed_dir_test = os.path.join(parent_root, 'ted_dataset/gesture_processed/test_')

    test_inputs = train_data_use(lang_model, data_root_test, datapath_test, processed_dir_test)

    data_root = os.path.join(parent_root, 'ted_dataset/lmdb_train_cache')
    datapath = os.path.join(data_root, 'data.mdb')
    processed_dir = os.path.join(parent_root, 'ted_dataset/gesture_processed/train_')

    train_inputs = train_data_use(lang_model, data_root, datapath, processed_dir)

    data_dict = {}
    data_info = {}


    for subset, inputs in zip(["train", "test", "val"], [train_inputs, test_inputs, val_inputs]):
        motions = []
        styles = []
        texts = []
        for input in inputs:
            styles.append(input["style"])
            motions.append(input["motion"])
            texts.append(input["text"])
        
        data_dict[subset] = {"motion": motions, "style": styles, "text": texts}


    print(len(data_dict["train"]["motion"]))
    print(len(data_dict["test"]["motion"]))
    print(len(data_dict["val"]["motion"]))
    different_style = list(set(data_dict["train"]["style"]))
    print(len(different_style))
    different_style = list(set(data_dict["test"]["style"]))
    print(len(different_style))
    different_style = list(set(data_dict["val"]["style"]))
    print(len(different_style))
    np.savez_compressed(os.path.join(current_root, 'data/locomotion/all_multi_context.npz'), **data_dict)
    print("Done.")
    