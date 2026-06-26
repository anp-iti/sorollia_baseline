import numpy as np
import h5py
import csv
import time
import logging

from utilities import int16_to_float32
import pandas as pd


def read_black_list(black_list_csv):
    """Read audio names from black list. 
    """
    with open(black_list_csv, 'r') as fr:
        reader = csv.reader(fr)
        lines = list(reader)

    black_list_names = ['Y{}.wav'.format(line[0]) for line in lines]
    return black_list_names


class AudioSetDataset(object):
    def __init__(self, sample_rate=32000):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        """
        
        self.sample_rate = sample_rate
    
    def __getitem__(self, meta):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': strAA, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        hdf5_path = meta['hdf5_path']
        index_in_hdf5 = meta['index_in_hdf5']

        with h5py.File(hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][index_in_hdf5].decode()
            waveform = int16_to_float32(hf['waveform'][index_in_hdf5])
            waveform = self.resample(waveform)
            target = hf['target'][index_in_hdf5].astype(np.float32)

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
            
        return data_dict

    def resample(self, waveform):
        """Resample.

        Args:
          waveform: (clip_samples,)

        Returns:
          (resampled_clip_samples,)
        """
        if self.sample_rate == 32000:
            return waveform
        elif self.sample_rate == 16000:
            return waveform[0 :: 2]
        elif self.sample_rate == 8000:
            return waveform[0 :: 4]
        else:
            raise Exception('Incorrect sample rate!')


class Base(object):
    def __init__(self, indexes_hdf5_path, batch_size, black_list_csv, random_seed):
        """Base class of train sampler.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Black list
        if black_list_csv:
            self.black_list_names = read_black_list(black_list_csv)
        else:
            self.black_list_names = []

        logging.info('Black list samples: {}'.format(len(self.black_list_names)))

        # Load target
        load_time = time.time()

        with h5py.File(indexes_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.hdf5_paths = [hdf5_path.decode() for hdf5_path in hf['hdf5_path'][:]]
            self.indexes_in_hdf5 = hf['index_in_hdf5'][:]
            self.targets = hf['target'][:].astype(np.float32)
        
        (self.audios_num, self.classes_num) = self.targets.shape
        logging.info('Training number: {}'.format(self.audios_num))
        logging.info('Load target time: {:.3f} s'.format(time.time() - load_time))


class TrainSampler(Base):
    def __init__(self, indexes_hdf5_path, batch_size, black_list_csv=None, 
        random_seed=1234, path_to_csv_labels=None):
        """Balanced sampler. Generate batch meta for training.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(TrainSampler, self).__init__(indexes_hdf5_path, batch_size, 
            black_list_csv, random_seed)
        
        self.indexes = np.arange(self.audios_num)
        self.samples_num_per_class = np.sum(self.targets, axis=0)
        df = pd.read_csv(path_to_csv_labels)
        self.labels = list(df["display_name"])
        # Shuffle indexes
        self.random_state.shuffle(self.indexes)
        
        self.pointer = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'hdf5_path': string, 'index_in_hdf5': int}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)
                
                # If audio in black list then continue
                if self.audio_names[index] in self.black_list_names:
                    continue
                else:
                    batch_meta.append({
                        'hdf5_path': self.hdf5_paths[index], 
                        'index_in_hdf5': self.indexes_in_hdf5[index]})
                    i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes': self.indexes,
            'pointer': self.pointer}
        return state
            
    def load_state_dict(self, state):
        self.indexes = state['indexes']
        self.pointer = state['pointer']


class EvaluateSampler(object):
    def __init__(self, indexes_hdf5_path, batch_size):
        """Evaluate sampler. Generate batch meta for evaluation.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
        """
        self.batch_size = batch_size

        with h5py.File(indexes_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.hdf5_paths = [hdf5_path.decode() for hdf5_path in hf['hdf5_path'][:]]
            self.indexes_in_hdf5 = hf['index_in_hdf5'][:]
            self.targets = hf['target'][:].astype(np.float32)
            
        self.audios_num = len(self.audio_names)

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'hdf5_path': string, 
             'index_in_hdf5': int}
            ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer, 
                min(pointer + batch_size, self.audios_num))

            batch_meta = []

            for index in batch_indexes:
                batch_meta.append({
                    'audio_name': self.audio_names[index], 
                    'hdf5_path': self.hdf5_paths[index], 
                    'index_in_hdf5': self.indexes_in_hdf5[index], 
                    'target': self.targets[index]})

            pointer += batch_size
            yield batch_meta


def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...}, 
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
    np_data_dict = {}

    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])

    return np_data_dict