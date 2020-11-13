from sklearn.datasets import load_boston
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

#X, y_data = load_boston(return_X_y=True)
X = load_boston()
y_data = X.target
feature_names = X.feature_names
X = X.data
m = np.mean(X, axis = 0)
st = np.std(X, axis = 0)
X = (X - m) / st
list_train = [(torch.from_numpy(X[iii,:]).float(),torch.tensor([y_data[iii]]).float()) for iii in range(X.shape[0])]


class Network(nn.Module):

  def __init__(self, init_dim, out_dim):
    super().__init__()
    self.fc1 = nn.Linear(init_dim, 5)
    self.fc2 = nn.Linear(5, out_dim)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

my_net = Network(13, 1)
opt = torch.optim.SGD(my_net.parameters(), lr=1e-4, momentum=0.9)
loss_fn = nn.MSELoss()

num_epochs = 10
for epoch in range(num_epochs):

  random.shuffle(list_train)
  tot_loss = 0.
  for x,y in list_train:
    my_net.zero_grad()
    x = torch.unsqueeze(x, dim=0)
    y = torch.unsqueeze(y, dim=0)
    out = my_net(x)
    #print(x,y)
    loss = loss_fn(out,y)
    tot_loss = tot_loss + loss.item()
    loss.backward()
    opt.step()
  





X_PI, y_data_PI = X.copy(), y_data.copy()

import matplotlib.pyplot as plt
figure, ax = plt.subplots(1,2, figsize= (18,6))
feat_id_list = [0,-1]
for idx,feat_id in enumerate(feat_id_list):

  x_min = np.min(X_PI[:, feat_id])
  x_max = np.max(X_PI[:, feat_id])
  x_pi = np.linspace(x_min, x_max, 100)

  mean_pred_list = []
  for iii in range(x_pi.shape[0]):
    new_X_PI = X_PI.copy()
    new_X_PI[:, feat_id] = x_pi[iii]
    list_train = [(torch.from_numpy(new_X_PI[jjj,:]).float(),
                torch.tensor([y_data[jjj]]).float()) for jjj in range(new_X_PI.shape[0])]
    mean_pred = 0.
    for x,y in list_train:
      my_net.zero_grad()
      x = torch.unsqueeze(x, dim=0)
      y = torch.unsqueeze(y, dim=0)
      out = my_net(x)
      mean_pred = mean_pred + out.item()
    mean_pred = mean_pred / len(list_train)
    mean_pred_list.append(mean_pred)


  ax[idx].plot(x_pi, mean_pred_list)
  ax[idx].set_ylabel('Partial Dependence')
  ax[idx].set_xlabel(feature_names[feat_id])
  ax[idx].grid(axis='y')
