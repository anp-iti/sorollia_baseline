import numpy as np
import argparse
import csv
import os
import glob
from datetime import datetime
import time
import logging
import h5py
import librosa
import torch
import config
import sys
sys.path.append(config.pytorch_path)
import pandas as pd

from utilities import (create_folder, get_filename, create_logging, 
    float32_to_int16, pad_or_truncate, read_metadata)

def read_audio_panns(audio_path, sample_rate):
    """
    Loads an audio file from a local or remote path and returns it as an array.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        numpy.ndarray: Loaded audio array.
    """

    (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    
    audio = audio[None, :]  # (batch_size, segment_samples)
    return audio

def create_dataset_logging(path_to_logs):
    
    datetime_name = (datetime.now()).strftime("%Y_%m_%d_%H_%M")

    logs_dir = os.path.join(path_to_logs, 'dataset', datetime_name)
    os.makedirs(logs_dir, exist_ok=True)
    logger = create_logging(logs_dir, filemode='w', filename='dataset.log',
                            level='INFO')
    
    return logger, logs_dir


def obtain_number_classes(csv_path):
    
    # Cargar el CSV
    df = pd.read_csv(csv_path)

    # Separar categorías y obtener valores únicos
    unique_categories = set()
    df["category"].dropna().str.split(";").apply(unique_categories.update)

    # Número de categorías únicas
    print(f"Number of unique categories: {len(unique_categories)}")
    print("Unique categories:", unique_categories)
    
    return len(unique_categories), unique_categories


def create_mappings(csv_path):
    
    df = pd.read_csv(csv_path) 
    
    labels = list(df["class_id"])
    ids =  list(df["index"])

    # classes_num = len(labels)

    lb_to_ix = {label : i for i, label in enumerate(labels)}
    ix_to_lb = {i : label for i, label in enumerate(labels)}

    id_to_ix = {id : i for i, id in enumerate(ids)}
    ix_to_id = {i : id for i, id in enumerate(ids)}

    return [id_to_ix,ix_to_lb,lb_to_ix,ix_to_id]


def pack_waveforms_to_hdf5(args):
    """Pack waveform and target of several audio clips to a single hdf5 file. 
    This can speed up loading and training.
    """
    
    logger, logs_dir = create_dataset_logging()
    
    logger.info(f'Log path: {logs_dir}')
    
    logger.info('-'*50)
    logger.info('Pack waveforms to h5 files')
    logger.info('-'*50)
    
    logger.info("Script executed with arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Arguments & parameters
    audios_dir = args.audios_dir
    csv_path = args.csv_path
    csv_label = args.csv_label
    waveforms_hdf5_path = args.waveforms_hdf5_path

    clip_samples = args.fsamp * args.duration
    classes_num, labels = obtain_number_classes(csv_path)
    sample_rate = args.fsamp
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax

    meta_dict = read_metadata(csv_path, classes_num, xx_to_xi)

    audios_num = len(meta_dict['audio_name'])
    logger.info(f'Number of audio files: {audios_num}')
    # Pack waveform to hdf5
    total_time = time.time()
    create_folder(os.path.dirname(waveforms_hdf5_path))
    logger.info(f"Creating folder to store data: {os.path.dirname(waveforms_hdf5_path)}" )

    with h5py.File(waveforms_hdf5_path, 'w') as hf:

        hf.create_dataset('audio_name', shape= ((audios_num,)), dtype = 'S30')
        hf.create_dataset('waveform', shape = ((audios_num, clip_samples)), dtype = np.int16)
        hf.create_dataset('target', shape = ((audios_num, classes_num)), dtype = bool)
        hf.attrs.create('sample_rate', data = sample_rate, dtype = np.int32)
        creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hf.attrs.create('creation_date', data=creation_date)
        
        # Pack waveform & target of several audio clips to a single hdf5 file
        waveforms = []
        audio_names = []
        targets = []

        for n in range(audios_num):
            logger.info(f'{n+1}/{len(audios_num)}')
            logger.info(audios_dir)
            audio_path = os.path.join(audios_dir, meta_dict['audio_name'][n])
            logger.info(audio_path)

            if os.path.isfile(audio_path):
                audio = read_audio_panns(audio_path = audio_path, sample_rate=sample_rate)

                hf['audio_name'][n] = meta_dict['audio_name'][n].encode()
                hf['waveform'][n] = float32_to_int16(audio)
                hf['target'][n] = meta_dict['target'][n]
                logger.info(meta_dict['audio_name'][n])

            else:
                logger.info('{} File does not exist! {}'.format(n, audio_path))
        
        hf.close()
    
    logger.info('Write to {}'.format(waveforms_hdf5_path))
    logger.info('Pack hdf5 time: {:.3f}'.format(time.time() - total_time))
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_pack_wavs = subparsers.add_parser('pack_waveforms_to_hdf5')
    parser_pack_wavs.add_argument('--csv_path', type = str, required = True, help = 'Path of csv file containing audio info to be downloaded.')
    parser_pack_wavs.add_argument('--audios_dir', type = str, required = False, help = 'Directory to save out downloaded audio.')
    parser_pack_wavs.add_argument('--waveforms_hdf5_path', type = str, required = True, help = 'Path to save out packed hdf5.')
    parser_pack_wavs.add_argument('--cuda', action = 'store_true', default = False, help = 'Use GPU to do different calculations')
    parser_pack_wavs.add_argument('--csv_label', type = str, required= True, help = 'Path to csv to map labels')
    parser_pack_wavs.add_argument('--fsamp', type=int, required=False, default=44100, help='Sample rate of audio')


    args = parser.parse_args()

    if args.mode == 'pack_waveforms_to_hdf5':
        pack_waveforms_to_hdf5(args)

    else:
        raise Exception('Incorrect arguments!')