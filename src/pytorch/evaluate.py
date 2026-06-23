
#FROM METRICS? metrics también, no?





import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import config
from data_generator import (AudioSetDataset, EvaluateSampler, collate_fn)
from utilities import create_folder

from sklearn import metrics
import pandas as pd
import numpy as np
import argparse

from pytorch_utils import forward
from models import *
sys.path.insert(1, os.path.join(sys.path[0], '../evaluation'))
from metrics import get_roc_curve,get_f1_score,get_confusion_matrix

class Evaluator(object):
    def __init__(self, model):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        
    def evaluate(self, data_loader, extra_analysis = False, path = None, path_to_label=None, logger=None):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict, 
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward(
            model = self.model, 
            generator = data_loader, 
            return_target = True,
            logger=logger)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)
        # print(target.shape)
        # print(clipwise_output.shape)

        # class_label_indices_problem_path = path_to_label
        
        valid_columns = [i for i in range(target.shape[1]) if len(np.unique(target[:, i])) > 1]
        new_target = target[:, valid_columns]
        new_clipwise_output = clipwise_output[:, valid_columns]
        # mapping_columns = {new: original for new, original in enumerate(valid_columns)}

        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None)
        # The fact is that there will be classes that will not belong to any audio. Then, the calculation must be just for the classes present
        auc = metrics.roc_auc_score(new_target, new_clipwise_output, average = None) # multi_class = ovr/ovo???? Using micro because, unbalanced.
        
        statistics = {'average_precision': average_precision, 'auc': auc}
        
        if extra_analysis:

            checkpoint_dir_outputs = os.path.join(path,'outputs')
            create_folder(checkpoint_dir_outputs)
            y_target = new_target
            # this needs to be removed as hardcoded
            class_label_indices = pd.read_csv(path_to_label) 
            classes = class_label_indices['display_name'].tolist()
            y_pred = list()

            for audio in clipwise_output:
                y_pred.append((audio > 0.5) + 0)

            y_pred = np.array(y_pred)

            get_roc_curve(new_target.astype(int),new_clipwise_output,classes,path_to_cm = os.path.join(checkpoint_dir_outputs,'roc_curve.png'))
            get_f1_score(new_target.astype(int), y_pred ,new_clipwise_output, classes, path_to_cm = os.path.join(checkpoint_dir_outputs,'metrics.txt'))
            get_confusion_matrix(new_target.astype(int), y_pred,classes, path_to_cm = os.path.join(checkpoint_dir_outputs, 'confusion_matrix.png'))

        return statistics
  