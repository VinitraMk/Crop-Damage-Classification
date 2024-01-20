import torch.nn as nn
import torch.optim as optim

from models.ots_models import get_model

class Alexnet(nn.Module):
    def __init__(self, num_classes, get_weights = False):
        super().__init__()
        self.model, _ = get_model("alexnet", get_weights)
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)