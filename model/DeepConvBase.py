"""
PyTorch implementation of DeepConvLSTM found at: https://github.com/dspanah/Sensor-Based-Human-Activity-Recognition-DeepConvLSTM-Pytorch.
"""

import torch
from torch import nn
import torch.nn.functional as F

class DeepConvBase(nn.Module):
    def __init__(self, input_length, n_sensor_channel, n_hidden=128, n_filters=64, filter_size=5, drop_prob=0.5):

        super().__init__()

        self.drop_prob = drop_prob
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_sensor_channel = n_sensor_channel
        self.input_length = input_length
             
        self.conv1 = nn.Conv1d(n_sensor_channel, n_filters, filter_size) 
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)
        
        self.fc1 = nn.Linear((input_length - 16)*n_filters, n_hidden) # 16 = 4 conv layers * 4
        
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, x):
        batch_size = x.shape[0]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.contiguous().view(batch_size, -1)
        x = self.dropout(x)

        x = self.fc1(x)

        return x
    
    def modify_model(self, nb_sensor_channels):
        self.conv1 = nn.Conv1d(nb_sensor_channels, self.n_filters, self.filter_size)