import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :])
        return out

def train_windows(df, ref_day=5, predict_day=1):
    X_train, Y_train = [], []
    for i in range(df.shape[0]-predict_day-ref_day+1):
        X_train.append(np.array(df.iloc[i:i+ref_day,:]))
        Y_train.append(np.array(df.iloc[i+ref_day:i+ref_day+predict_day]["Close"]))
    return np.array(X_train), np.array(Y_train)


# Load the data
data = pd.read_csv('AABA_2006-01-01_to_2018-01-01.csv')
data = data.drop(columns=['Date', 'Name'])

sequence_length = 10
X_train, Y_train = train_windows(data, sequence_length)
X_train_2D = X_train.reshape(-1, X_train.shape[-1])

# Initialize the MinMaxScaler
scaler = MinMaxScaler()
X_train_normalized_2D = scaler.fit_transform(X_train_2D)
X_train = X_train_normalized_2D.reshape(X_train.shape)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

input_dim = X_train_tensor.shape[2]
hidden_dim = 64
num_layers = 2
output_dim = Y_train_tensor.shape[1]

model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    optimizer.zero_grad()
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

print('Finished Training')