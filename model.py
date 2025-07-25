import torch.nn as nn

class SalaryModel(nn.Module):
    def __init__(self, input_features):
        super(SalaryModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)