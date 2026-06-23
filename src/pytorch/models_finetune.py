'''
Models supported to be finetuned
'''

import torch
import torch.nn as nn
from models import Cnn14, Cnn10, Cnn6, MobileNetV1, MobileNetV2, init_layer

# CNN14
class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base,loss_type):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527
        # print(f'Classes num: {classes_num}')
        
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)
        self.loss_type = loss_type
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False
            
            # Descongelar f1 - HARDCODED FOR PAPER
            # for param in self.base.fc1.parameters():
            #     param.requires_grad = True

        self.init_weights()

        for name, param in self.named_parameters():
            print(name, param.requires_grad)

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path, weights_only=False)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
        # print(embedding.shape)
        if self.loss_type == 'clip_bce':
                clipwise_output =  torch.sigmoid(self.fc_transfer(embedding))
        else:
            clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict

# CNN10: Based on CNN14
class Transfer_Cnn10(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base,loss_type):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn10, self).__init__()
        audioset_classes_num = 527
        # print(f'Classes num: {classes_num}')
        
        self.base = Cnn10(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, classes_num, bias=True)
        self.loss_type = loss_type
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
        # print(embedding.shape)
        if self.loss_type == 'clip_bce':
                clipwise_output =  torch.sigmoid(self.fc_transfer(embedding))
        else:
            clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict

class Transfer_Cnn6(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base,loss_type):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn6, self).__init__()
        audioset_classes_num = 527
        # print(f'Classes num: {classes_num}')
        
        self.base = Cnn6(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(512, classes_num, bias=True)
        self.loss_type = loss_type
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
        # print(embedding.shape)
        if self.loss_type == 'clip_bce':
                clipwise_output =  torch.sigmoid(self.fc_transfer(embedding))
        else:
            clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict

# Mobilenetv2: using Cnn14 as template
class Transfer_MobileNetV2(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base,loss_type):
        """Classifier for a new task using pretrained MobileNetV2 as a sub module.
        """
        super(Transfer_MobileNetV2, self).__init__()
        audioset_classes_num = 527
        # print(f'Classes num: {classes_num}')
        
        self.base = MobileNetV2(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(1024, classes_num, bias=True)
        self.loss_type = loss_type
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
        # print(embedding.shape)
        if self.loss_type == 'clip_bce':
                clipwise_output =  torch.sigmoid(self.fc_transfer(embedding))
        else:
            clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict
    
# Mobilenetv1: using Mobilenetv2 as template
class Transfer_MobileNetV1(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base,loss_type):
        """Classifier for a new task using pretrained MobileNetV2 as a sub module.
        """
        super(Transfer_MobileNetV1, self).__init__()
        audioset_classes_num = 527
        # print(f'Classes num: {classes_num}')
        
        self.base = MobileNetV1(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(1024, classes_num, bias=True)
        self.loss_type = loss_type
        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']
        # print(embedding.shape)
        if self.loss_type == 'clip_bce':
                clipwise_output =  torch.sigmoid(self.fc_transfer(embedding))
        else:
            clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict
