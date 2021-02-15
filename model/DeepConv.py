import torch
from torch import nn
import torch.nn.functional as F
from model.DeepConvBase import DeepConvBase

class DeepConv(DeepConvBase):
    def __init__(self, n_classes=18, n_hidden=128, *args, **kwargs):

        super().__init__(n_hidden=n_hidden, *args, **kwargs)

        self.n_classes = n_classes
        self.fc = nn.Linear(n_hidden, n_classes)
    
    def forward(self, x):
        x = super().forward(x)
        x = self.fc(x)
        
        return x
    
    def modify_model(self, nb_sensor_channels, n_classes):
        super().modify_model(nb_sensor_channels=nb_sensor_channels)

        self.fc = nn.Linear(self.n_hidden, n_classes)
        self.n_classes = n_classes