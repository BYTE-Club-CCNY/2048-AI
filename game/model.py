import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(F.linear(self.linear1(x)))
        x = F.linear(self.linear2(x))
        return x

    def save(self, epoch, file_name="model"):
        # create a directory to store the mode
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, f"{file_name}_epoch_{epoch}.pth")
        torch.save(self.state_dict(), file_name)


class QTrainer:
# train_step function, init function
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            # add a batch dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        # 1: predicted Q values with current state
        pred = self.model(state)
        
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        
        target = pred.clone()
        
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        
        # empty the gradient
        self.optimizer.zero_grad()
        self.criterion(target, pred).backward()
        self.optimizer.step()