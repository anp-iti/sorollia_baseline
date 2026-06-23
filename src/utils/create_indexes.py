import numpy as np
import argparse
import csv
import os
import glob
import datetime
import time
import logging
import h5py
import librosa

from utils.utilities import create_folder


def create_indexes(args):
    """Create indexes a for dataloader to read for training. When users have 
    a new task and their own data, they need to create similar indexes. The 
    indexes contain meta information of "where to find the data for training".
    """

    # Arguments & parameters
    waveforms_hdf5_path = args.waveforms_hdf5_path
    indexes_hdf5_path = args.indexes_hdf5_path
    
    # Paths
    create_folder(os.path.dirname(indexes_hdf5_path))
    
    with h5py.File(waveforms_hdf5_path, 'r') as hr:
        with h5py.File(indexes_hdf5_path, 'w') as hw:
            audios_num = len(hr['audio_name'])
            hw.create_dataset('audio_name', data=hr['audio_name'][:], dtype='S20')
            hw.create_dataset('target', data=hr['target'][:], dtype=bool)
            hw.create_dataset('hdf5_path', data=[waveforms_hdf5_path.encode()] * audios_num, dtype='S200')
            hw.create_dataset('index_in_hdf5', data=np.arange(audios_num), dtype=np.int32)
            creation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            hw.attrs.create('creation_date', data=creation_date)
            hw.attrs.create('sample_rate', data=hr.attrs['sample_rate'], dtype = np.int32)
            hw.close()
        hr.close()
    print('Write to {}'.format(indexes_hdf5_path))
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_indexes = subparsers.add_parser('create_indexes')
    parser_create_indexes.add_argument('--waveforms_hdf5_path', type=str, required=True, help='Path of packed waveforms hdf5.')
    parser_create_indexes.add_argument('--indexes_hdf5_path', type=str, required=True, help='Path to write out indexes hdf5.')

  
    args = parser.parse_args()
    
    if args.mode == 'create_indexes':
        create_indexes(args)


    else:
        raise Exception('Incorrect arguments!')