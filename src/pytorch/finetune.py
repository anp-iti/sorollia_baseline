import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from datetime import datetime
import numpy as np
import argparse
import h5py
# import math
import time
# import logging

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
from torchview import draw_graph
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau
 
from utilities import (create_folder, get_filename, create_logging, Mixup, 
    StatisticsContainer)
# from models import *
from models_finetune import Transfer_Cnn14, Transfer_Cnn10, Transfer_Cnn6, Transfer_MobileNetV2, Transfer_MobileNetV1
from pytorch_utils import (move_data_to_device, count_parameters, count_flops, 
    do_mixup)
from data_generator import (AudioSetDataset, TrainSampler, EvaluateSampler, collate_fn)
from evaluate import Evaluator
from losses import get_loss_func

def train_loop(train_loader, eval_bal_loader,
               checkpoints_dir, statistics_container, loss_func,
               evaluator, scheduler, optimizer, train_sampler, model, args, logger=None):
    
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    iteration = 0
    no_improvement_count = 0 # init early stopping count 
    best_val_map = float('-inf') # init validation metrics
    train_bgn_time = time.time()
    time1 = time.time()
    if 'mixup' in args.augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)
    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """
        
        if no_improvement_count >= args.early_stop:
            torch.cuda.empty_cache()
            break
    
        # Evaluate
        if (iteration % args.val_iter == 0 and iteration > args.resume_iteration) or (iteration == 0) or (iteration == args.total_iterations):
            if iteration == 0:
                logger.info('Obtaining metrics of first iteration')
            elif iteration == args.total_iterations:
                logger.info(f'Obtaining metrics of last iteration: {args.total_iterations}')
            else:
                logger.info(f'Obtaining metrics of iteration: {iteration}')
            train_fin_time = time.time()
            
            test_statistics = evaluator.evaluate(eval_bal_loader, logger=logger)

            map_aux = np.mean(test_statistics['average_precision'])
            logger.info(f'Validate test mAP | iteration: {iteration} = {map_aux:.3f}')
            
           
            if iteration > 0:
                scheduler.step(map_aux)
                if map_aux > best_val_map:
                    checkpoint = {
                        'iteration': iteration, 
                        'model': model.module.state_dict(), 
                        'sampler': train_sampler.state_dict()}

                    checkpoint_path = os.path.join(
                        checkpoints_dir, 'best.pth'.format(iteration))
                    torch.save(checkpoint, checkpoint_path)
                    best_weights_path = os.path.join(
                        checkpoints_dir, f'best_w.pth')
                    torch.save(model.state_dict(), best_weights_path)

                    best_val_map = map_aux
                    no_improvement_count = 0
                    logger.info(f"(*) New best model - mAP = | iteration: {iteration} = {map_aux:.3f}")
                else:
                    no_improvement_count += 1
                    logger.info("(*) No improvement iteration {} - patience {}".format(iteration, no_improvement_count))
                                        
            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            if iteration != 0:
                logger.info(
                    'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                        ''.format(iteration, train_time, validate_time))
            else:
                logger.info(
                    'iteration: {}, validate time: {:.3f} s'
                        ''.format(iteration, validate_time))

            logger.info('------------------------------------')

            train_bgn_time = time.time()
        
        # Save model
        if iteration % args.save_model_iter == 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logger.info('Model saved to {}'.format(checkpoint_path))
        
        # Mixup lambda
        if 'mixup' in args.augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict['waveform']))

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Forward
        model.train()
        if 'mixup' in args.augmentation:
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
        if iteration == args.total_iterations:
            torch.cuda.empty_cache()
            break

        iteration += 1 ##


def net_summary(model,clip_samples,logs_dir,batch_size,full_summary):
    model_summary = summary(model,tuple([clip_samples]), device='cpu', verbose=0)
    with open(os.path.join(logs_dir, 'network_arch.txt'), 'w') as net_file:
        net_file.write(str(model_summary))
    # Graph visualization of the model used in the experiment
    if full_summary:
        model_graph = draw_graph(model, input_size=(batch_size,clip_samples), save_graph=True, 
                                 directory=logs_dir, filename = 'model_arch',
                                 show_shapes=True,expand_nested=True)
    else:
        model_graph = draw_graph(model, input_size=(batch_size,clip_samples), save_graph=True,
                                 directory=logs_dir, filename = 'model_arch',
                                 show_shapes=True, expand_nested=True, depth = 1)


def sampler(train_indexes_hdf5_path,eval_bal_indexes_hdf5_path,eval_test_indexes_hdf5_path,
            batch_size,augmentation,black_list_csv,
            path_to_csv_labels):

   
     
    train_sampler = TrainSampler(
        indexes_hdf5_path=train_indexes_hdf5_path, 
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size,
        black_list_csv=black_list_csv,
        path_to_csv_labels=path_to_csv_labels)
    
    # Evaluate sampler
    eval_bal_sampler = EvaluateSampler(indexes_hdf5_path=eval_bal_indexes_hdf5_path, batch_size=batch_size)
    eval_test_sampler = EvaluateSampler(indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size)
    return train_sampler,eval_bal_sampler,eval_test_sampler

def data_loader(dataset,train_sampler,eval_bal_sampler,eval_test_sampler):
    num_workers = 8
     

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
    return train_loader,eval_bal_loader,eval_test_loader

def pie_plot(train_sampler,logs_dir):
    total = sum(train_sampler.samples_num_per_class)
    percentages = [(label, sample / total * 100) for label, sample in zip(train_sampler.labels, train_sampler.samples_num_per_class)]
    percentages.sort(key=lambda x: x[1], reverse=True)
    formatted_legend = [f"{label}: {percentage:.1f}%" for label, percentage in percentages]
    plt.pie(train_sampler.samples_num_per_class,labels=None)
    plt.legend(formatted_legend, title='Categories',loc = 'best',bbox_to_anchor=(1.25, 0.8))
    plt.title('Samples of each class in the training set')
    plt.savefig(os.path.join(logs_dir,'pie_chart_samples.png'),bbox_inches='tight')

def train(args):

    # Arugments & parameters
    workspace = args.workspace
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    train_indexes_hdf5_path = args.train_data
    eval_test_indexes_hdf5_path = args.test_data
    val_data = args.val_data
    model_type = args.model_type
    loss_type = args.loss_type
    augmentation = args.augmentation
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    total_iterations = args.total_iterations
    patience_lr = args.patience_lr
    factor_lr = args.factor_lr
    save_model_iter = args.save_model_iter
    val_iter = args.val_iter
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    
    filename = args.filename
    pretrain = True if pretrained_checkpoint_path else False
    full_summary = args.full_summary
    
    datetime_name = (datetime.now()).strftime("%Y_%m_%d_%H_%M")

    logs_dir = os.path.join(workspace, datetime_name)
    # First, create the loggder
    logger = create_logging(logs_dir, filemode='w', filename='finetune.log', level=args.log_level)

    logger.info('Logger created')

    if 'cuda' in str(device):
        logger.info('Using GPU.')
        device = 'cuda'
    else:
        logger.critical('Using CPU. Set --cuda flag to use GPU. Stopping training')
        # device = 'cpu'
        sys.exit()

    
    loss_func = get_loss_func(loss_type)
    black_list_csv = None
    logger.info(f'Path to training data: {train_indexes_hdf5_path}')
    logger.info(f'Path to test data: {train_indexes_hdf5_path}')
    
    if val_data:
        eval_bal_indexes_hdf5_path = args.val_data
    else:
        eval_bal_indexes_hdf5_path = eval_test_indexes_hdf5_path
    logger.info(f'Path to validation data: {train_indexes_hdf5_path}')
    

    # Read number of labels/targets according to target dataset, number of columns in one-hot encoding
    with h5py.File(train_indexes_hdf5_path, "r") as f:
        target = f['target'][:]
        
    classes_num = target.shape[1]
    
    logger.info(f'Number of classes: {classes_num}')

    checkpoints_dir = os.path.join(workspace, datetime_name, 'checkpoints')
    create_folder(checkpoints_dir)
    logger.info(f'Path to checkpoints: {checkpoints_dir}')

    statistics_path = os.path.join(workspace, datetime_name,'statistics.pkl')
    logger.info(f'Path to statistics according to PANNs framework: {statistics_path}\n')
    
    
    logger.info(f"""Parameters used to train the model
    workspace = {workspace}
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
    
    # Model, Transfer_Cnn14
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num, freeze_base,loss_type)
    
    logger.info("Model loaded")
    
   
    params_num = count_parameters(model)
    logger.info('Parameters num: {}'.format(params_num))
    
    if pretrain:
        model.load_from_pretrain(pretrained_checkpoint_path) 
        logger.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        logger.info('Load pretrained model successfully')

    dataset = AudioSetDataset(sample_rate=sample_rate)
    train_sampler,eval_bal_sampler,eval_test_sampler =sampler(train_indexes_hdf5_path,
                                                                eval_bal_indexes_hdf5_path, eval_test_indexes_hdf5_path,
                                                                batch_size, augmentation,black_list_csv,
                                                                path_to_csv_labels=args.csv_label)
    train_loader,eval_bal_loader,eval_test_loader = data_loader(dataset,train_sampler,eval_bal_sampler,eval_test_sampler)
    
    logger.info("Datasets loaded")

    # Pie plot showing the samples for training
    pie_plot(train_sampler,logs_dir)
    logger.info(f"Training pie distribution stored at {logs_dir}")
    
    # Evaluator
    evaluator = Evaluator(model=model)
        
    # Statistics
    statistics_container = StatisticsContainer(statistics_path)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='max',
                                  patience=patience_lr, factor=factor_lr)

    
    # Parallel
    logger.info('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    if 'cuda' in device:
        model.to(device)
   
    

    run_name = args.run_name

    logger.info(f" Run name: {run_name}")
    
    logger.info("Starting training loop...\n")
    train_loop(train_loader, eval_bal_loader,checkpoints_dir, statistics_container,
                loss_func,evaluator, scheduler, optimizer, train_sampler, model,
                args= args, logger=logger)
    # final test
    logger.info('Obtaining extra metrics')
    checkpoint_path = os.path.join(checkpoints_dir, 'best.pth')
    best_weights_path = os.path.join(
                checkpoints_dir, f'best_w.pth')
    model.load_state_dict(torch.load(best_weights_path))
    execution_folder = os.path.dirname(checkpoints_dir)
    test_statistics = evaluator.evaluate(eval_test_loader,
                                            extra_analysis=True,
                                            path=execution_folder,
                                            path_to_label=args.csv_label)
    
    logger.info('Training finished!')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace',type= str,required=True)
    parser_train.add_argument('--sample_rate', type=int, required=True)
    parser_train.add_argument('--window_size', type=int, required=True)
    parser_train.add_argument('--hop_size', type=int, required=True)
    parser_train.add_argument('--mel_bins', type=int, required=True)
    parser_train.add_argument('--fmin', type=int, required=True)
    parser_train.add_argument('--fmax', type=int, required=True) 
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, default='clip_bce', choices=['clip_bce','clip_classification'])
    parser_train.add_argument('--augmentation', type=str, default='mixup', choices=['none', 'mixup'])
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--resume_iteration', type=int, default=0)
    parser_train.add_argument('--early_stop', type=int, default=10)
    parser_train.add_argument('--patience_lr', type=int, default=5)
    parser_train.add_argument('--factor_lr', type=float, default=0.5)
    parser_train.add_argument('--total_iterations', type=int, default=10000000)
    parser_train.add_argument('--train_data', type=str, required=True)
    parser_train.add_argument('--test_data', type=str, required=True)
    parser_train.add_argument('--val_data', type=str, required=False)
    parser_train.add_argument('--save_model_iter', required=False, type=int, default=10000,
                              help='number of iterations to save the model')
    parser_train.add_argument('--val_iter', required=False, type=int, default=1000,
                              help='number of iterations to run a validation process')
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--full_summary', action='store_true', default=False)
    parser_train.add_argument('--csv_label', required=True, type=str)
    parser_train.add_argument('--log_level', required=False, type=str, default='INFO')
    parser_train.add_argument('--run_name', required=True, type=str)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')
    

