import torch.nn as nn

class ANN(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.model(x)