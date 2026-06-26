import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from datetime import datetime
import numpy as np
import argparse
import time
import logging
import h5py
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
from torchview import draw_graph
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau
 
from utilities import (create_folder, delete_folder, get_filename, create_logging, Mixup, 
    StatisticsContainer)
from models import (Cnn14, Cnn14_no_specaug, Cnn14_no_dropout, 
    Cnn6, Cnn10, ResNet22, ResNet38, ResNet54, Cnn14_emb512, Cnn14_emb128, 
    Cnn14_emb32, MobileNetV1, MobileNetV2, LeeNet11, LeeNet24, DaiNet19, 
    Res1dNet31, Res1dNet51, Wavegram_Cnn14, Wavegram_Logmel_Cnn14, 
    Wavegram_Logmel128_Cnn14, Cnn14_16k, Cnn14_8k, Cnn14_mel32, Cnn14_mel128, 
    Cnn14_mixup_time_domain, Cnn14_DecisionLevelMax, Cnn14_DecisionLevelAtt)
from pytorch_utils import (move_data_to_device, count_parameters, count_flops, 
    do_mixup)
from data_generator import (AudioSetDataset, TrainSampler, EvaluateSampler, collate_fn)
from evaluate import Evaluator
from losses import get_loss_func

import pandas as pd

from argparse import Namespace

def train(args):
    
    """
    Args:
        workspace: str
        data_type: 'balanced_train' | 'full_train'
        sample_rate: int
        window_size: int
        hop_size: int
        mel_bins: int
        fmin: int
        fmax: int
        model_type: list[str]
        loss_type: 'clip_bce'
        augmentation: 'none' | 'mixup'
        batch_size: int
        learning_rate: float
        resume_iteration: int
        early_stop: int
        patience_lr: int
        factor_lr: float
        total_iterations: int
        cuda: bool
        full_summary: bool
        train_data: str
        val_data: str
        test_data: str
        save_model_iter: int
        val_iter: int
        log_level: str
        csv_label: str
        run_info: str
    """

    # Arugments & parameters
    workspace = args.workspace
    data_type = args.data_type
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    models = args.model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    total_iterations = args.total_iterations
    patience_lr = args.patience_lr
    factor_lr = args.factor_lr
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename
    train_indexes_hdf5_path = args.train_data
    eval_test_indexes_hdf5_path = args.test_data
    val_data = args.val_data
    save_model_iter = args.save_model_iter
    val_iter = args.val_iter
    full_summary = args.full_summary
    
    num_workers = 8
    run_info = args.run_info
 
    datetime_name = (datetime.now()).strftime("%Y_%m_%d_%H_%M")
    
    with h5py.File(train_indexes_hdf5_path, "r") as f:
        target = f['target'][:]
        
    classes_num = target.shape[1]
    
    loss_func = get_loss_func(loss_type)
    
    if not isinstance(models, list):
        models = [models]
    # Paths
    black_list_csv = None

    if val_data:
        eval_bal_indexes_hdf5_path = os.path.join(workspace,'hdf5s', 'indexes', val_data)
    else:
        eval_bal_indexes_hdf5_path = eval_test_indexes_hdf5_path


    checkpoints_dir = os.path.join(workspace, datetime_name, 'checkpoints')
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, datetime_name,'statistics.pkl')
    logs_dir = os.path.join(workspace, datetime_name)
    
    
    for model_type in models:

        logger = create_logging(logs_dir, filemode='w', filename='train.log',
                            level=args.log_level)
      
        logger.info(f"""Parameters used to train the model
        workspace = {workspace}
        data_type = {data_type}
        sample_rate = {sample_rate}
        window_size = {window_size}
        hop_size = {hop_size}
        mel_bins = {mel_bins}
        fmin = {fmin}
        fmax = {fmax}
        model_type = {model_type}
        loss_type = {loss_type}
        augmentation = {augmentation}
        batch_size = {batch_size}
        learning_rate = {learning_rate}
        resume_iteration = {resume_iteration}
        early_stop = {early_stop}
        filename = {filename}
        total_iterations = {total_iterations}""")

        if 'cuda' in str(device):
            logger.info('Using GPU.')
            device = 'cuda'
        else:
            logger.info('Using CPU. Set --cuda flag to use GPU.')
            device = 'cpu'
            
        # Model
        Model = eval(model_type)
        
        model = Model(sample_rate=sample_rate, window_size=window_size, 
            hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
            classes_num=classes_num)
        # Text summary of the model
        model_summary = summary(model,tuple([320000]), device='cpu', verbose=0)
        with open(os.path.join(logs_dir, 'network_arch.txt'), 'w') as net_file:
            net_file.write(str(model_summary))
       

        params_num = count_parameters(model)
        
        logger.info('Parameters num: {}'.format(params_num))
        
        # Dataset will be used by DataLoader later. Dataset takes a meta as input 
        # and return a waveform and a target.
        dataset = AudioSetDataset(sample_rate=sample_rate)
        
        
        
        train_sampler = TrainSampler(
            indexes_hdf5_path=train_indexes_hdf5_path, 
            batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size,
            black_list_csv=black_list_csv,
            path_to_csv_labels=args.csv_label)
        
        # Evaluate sampler
        eval_bal_sampler = EvaluateSampler(indexes_hdf5_path=eval_bal_indexes_hdf5_path, batch_size=batch_size)
        eval_test_sampler = EvaluateSampler(indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size)

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=dataset, 
            batch_sampler=train_sampler, collate_fn=collate_fn, 
            num_workers=num_workers, pin_memory=True)
        
        eval_bal_loader = torch.utils.data.DataLoader(dataset=dataset, 
            batch_sampler=eval_bal_sampler, collate_fn=collate_fn, 
            num_workers=num_workers, pin_memory=True)

        eval_test_loader = torch.utils.data.DataLoader(dataset=dataset, 
            batch_sampler=eval_test_sampler, collate_fn=collate_fn, 
            num_workers=num_workers, pin_memory=True)

        # Pie plot showing the samples for training
        total = sum(train_sampler.samples_num_per_class)
        percentages = [(label, sample / total * 100) for label, sample in zip(train_sampler.labels, train_sampler.samples_num_per_class)]
        percentages.sort(key=lambda x: x[1], reverse=True)
        formatted_legend = [f"{label}: {percentage:.1f}%" for label, percentage in percentages]
        sorted_samples = [x[1] for x in percentages]
        fig_pie = plt.pie(sorted_samples,labels = None)
        plt.legend(formatted_legend, title = 'Categories',loc = 'best', bbox_to_anchor = (1.25, 0.8))
        plt.title('Samples of each class in the training set')
        plt.savefig(os.path.join(logs_dir,'pie_chart_samples.png'), bbox_inches = 'tight')
        plt.close()

        if 'mixup' in augmentation:
            mixup_augmenter = Mixup(mixup_alpha = 1.)

        # Evaluator
        evaluator = Evaluator(model=model)
            
        # Statistics
        statistics_container = StatisticsContainer(statistics_path)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, 
            betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0., amsgrad = True)
        scheduler = ReduceLROnPlateau(optimizer, mode = 'max',
                                    patience = patience_lr, factor = factor_lr, verbose = True)

        train_bgn_time = time.time()
        
        # Resume training
        if resume_iteration > 0:
            resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
                'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
                sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
                'data_type={}'.format(data_type), model_type, 
                'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
                'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
                '{}_iterations.pth'.format(resume_iteration))

            logger.info('Loading checkpoint {}'.format(resume_checkpoint_path))
            checkpoint = torch.load(resume_checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            train_sampler.load_state_dict(checkpoint['sampler'])
            statistics_container.load_state_dict(resume_iteration)
            iteration = checkpoint['iteration']

        else:
            iteration = 0
        
        # Parallel
        logger.info('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

        if 'cuda' in str(device):
            model.to(device)
        
        time1 = time.time()
        
        best_val_map = float('-inf') # init validation metrics 
        no_improvement_count = 0 # init early stopping count        

        
        run_name = f"{model_type}_{run_info}_{datetime_name}"
        logger.info(f" Run name: {run_name}")
            
        for batch_data_dict in train_loader:
            """batch_data_dict: {
                'audio_name': (batch_size [*2 if mixup],), 
                'waveform': (batch_size [*2 if mixup], clip_samples), 
                'target': (batch_size [*2 if mixup], classes_num), 
                (ifexist) 'mixup_lambda': (batch_size * 2,)}
            """
            
            if no_improvement_count >= early_stop:
                torch.cuda.empty_cache()
                break
        
            # Evaluate
            if (iteration % val_iter == 0 and iteration > resume_iteration) or (iteration == 0) or (iteration == args.total_iterations):
                
                if iteration == 0:
                    logger.info('Obtaining metrics of first iteration')
                elif iteration == args.total_iterations:
                    logger.info(f'Obtaining metrics of last iteration: {args.total_iterations}')
                else:
                    logger.info(f'Obtaining metrics of iteration: {iteration}')
                
                train_fin_time = time.time()
                
                test_statistics = evaluator.evaluate(eval_bal_loader, logger=logger)

                logger.info('Validate test mAP: {:.3f}'.format(
                    np.mean(test_statistics['average_precision'])))
                map_aux = np.mean(test_statistics['average_precision'])
                if iteration > 0:
                    scheduler.step(map_aux)
                    if map_aux > best_val_map:
                        checkpoint = {
                            'iteration': iteration, 
                            'model': model.module.state_dict(), 
                            'sampler': train_sampler.state_dict()}

                        checkpoint_path = os.path.join(
                            checkpoints_dir, 'best_checkpoint.pth'.format(iteration))
                        # saving dict checkpoint
                        torch.save(checkpoint, checkpoint_path)
                        # saving model weigths
                        torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best_model.pth'))
                        best_val_map = map_aux
                        no_improvement_count = 0
                        logger.info("(*) New best model - mAP = {:.3f}".format(map_aux))
                    else:
                        no_improvement_count += 1
                        logger.info("(*) No improvement iteration {} - patience {}".format(iteration, no_improvement_count))

                statistics_container.append(iteration, test_statistics, data_type='test')
                statistics_container.dump()

                train_time = train_fin_time - train_bgn_time
                validate_time = time.time() - train_fin_time

                logger.info(
                    'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                        ''.format(iteration, train_time, validate_time))

                logger.info('------------------------------------')

                train_bgn_time = time.time()
            
            # Save model
            if iteration % save_model_iter == 0:
                checkpoint = {
                    'iteration': iteration, 
                    'model': model.module.state_dict(), 
                    'sampler': train_sampler.state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_iterations.pth'.format(iteration))
                    
                torch.save(checkpoint, checkpoint_path)
                logger.info('Model saved to {}'.format(checkpoint_path))
            
            # Mixup lambda
            if 'mixup' in augmentation:
                batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                    batch_size=len(batch_data_dict['waveform']))

            # Move data to device
            for key in batch_data_dict.keys():
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
            
            # Forward
            model.train()

            if 'mixup' in augmentation:
                batch_output_dict = model(batch_data_dict['waveform'], 
                    batch_data_dict['mixup_lambda'])
                """{'clipwise_output': (batch_size, classes_num), ...}"""

                batch_target_dict = {'target': do_mixup(batch_data_dict['target'], 
                    batch_data_dict['mixup_lambda'])}
                """{'target': (batch_size, classes_num)}"""
            else:
                batch_output_dict = model(batch_data_dict['waveform'], None)
                """{'clipwise_output': (batch_size, classes_num), ...}"""

                batch_target_dict = {'target': batch_data_dict['target']}
                """{'target': (batch_size, classes_num)}"""

            # Loss
            loss = loss_func(batch_output_dict, batch_target_dict)

            # Backward
            loss.backward()
            logger.info(f"Iteration {iteration} - loss {loss.item()}")
            
            optimizer.step()
            optimizer.zero_grad()
            
            if iteration % 10 == 0:
                logger.info('--- Iteration: {}, train time: {:.3f} s / 10 iterations ---'\
                    .format(iteration, time.time() - time1))
                time1 = time.time()
            
            # Stop learning
            if iteration == total_iterations:
                torch.cuda.empty_cache()
                break

            iteration += 1
    
        # final test
        checkpoint_path = os.path.join(checkpoints_dir, 'best_checkpoint.pth')
        execution_folder = os.path.dirname(checkpoints_dir)
        model.load_state_dict(torch.load(os.path.join(checkpoints_dir,'best_model.pth'))) # loading best model for metrics
        if full_summary:
            logger.info('Final metrics')
            test_statistics = evaluator.evaluate(eval_test_loader,
                                            extra_analysis=True,
                                            path=execution_folder,
                                            path_to_label=args.csv_label)
            
            
    
        if args.run_info:    
            persistent_dir = os.path.join(workspace, args.run_info)
            os.makedirs(persistent_dir, exist_ok=True)

          
            src_metrics = os.path.join(execution_folder, "outputs", "metrics.txt")
            dst_metrics = os.path.join(persistent_dir, "metrics.txt")
            shutil.copyfile(src_metrics, dst_metrics)

            # (Opcional) copiar otras métricas si quieres
            for fname in ["confusion_matrix.png", "roc_curve.png"]:
                src = os.path.join(execution_folder, "outputs", fname)
                if os.path.exists(src):
                    shutil.copyfile(src, os.path.join(persistent_dir, fname))

        logger.info(f'{model_type} training finished!')


  

        

if __name__ == '__main__':
    
    
    available_models = ['Cnn14','Cnn10','Cnn6','MobileNetV1','MobileNetV2',
                        'DaiNet19', 'LeeNet11', 'LeeNet24', 'ResNet22', 'ResNet38',
                        'ResNet54', 'Res1dNet31', 'Res1dNet51', 'Wavegram_Cnn14',
                        'Wavegram_Logmel_Cnn14'] # Cnn14_DecisionLevelMax
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, default='full_train', choices=['balanced_train', 'full_train'])
    parser_train.add_argument('--sample_rate', type=int, default=32000)
    parser_train.add_argument('--window_size', type=int, default=1024)
    parser_train.add_argument('--hop_size', type=int, default=320)
    parser_train.add_argument('--mel_bins', type=int, default=64)
    parser_train.add_argument('--fmin', type=int, default=50)
    parser_train.add_argument('--fmax', type=int, default=14000) 
    parser_train.add_argument('--model_type', nargs = '*', default = available_models)
    parser_train.add_argument('--loss_type', type=str, default='clip_bce', choices=['clip_bce'])
    parser_train.add_argument('--augmentation', type=str, default='mixup', choices=['none', 'mixup'])
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--resume_iteration', type=int, default=0)
    parser_train.add_argument('--early_stop', type=int, default=10)
    parser_train.add_argument('--patience_lr', type=int, default=5)
    parser_train.add_argument('--factor_lr', type=float, default=0.5)
    parser_train.add_argument('--total_iterations', type=int, default=10000000)
    parser_train.add_argument('--cuda', action='store_true', default=True)
    parser_train.add_argument('--full_summary', action='store_true', default=False)
    parser_train.add_argument('--train_data', type=str, required=True)
    parser_train.add_argument('--val_data', type=str, required=False)
    parser_train.add_argument('--test_data', type=str, required=True)
    parser_train.add_argument('--save_model_iter', required=False, type=int, default=10000,
                              help='number of iterations to save the model')
    parser_train.add_argument('--val_iter', required=False, type=int, default=1000,
                              help='number of iterations to run a validation process')
    parser_train.add_argument('--log_level', required=False, type=str, default='INFO')
    parser_train.add_argument('--csv_label', required=True, type=str)

    parser_train.add_argument('--run_info', type=str, help='additional info to identify the run')
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')