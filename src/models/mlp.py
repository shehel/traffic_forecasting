import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64, dropout=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(hidden_size, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x