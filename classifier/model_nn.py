import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class NN(nn.Module):
	def __init__(self, input_dim=300, hidden_dim=64, batch_size=1, output_dim=1, num_layers=2, bidirectional=True):
		super(NN, self).__init__()

		self.net = nn.Sequential(
		nn.Linear(360, 128),
		# nn.Dropout(0.4),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Linear(128, 16),
		# nn.Dropout(0.2),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Linear(16, 1),
		nn.Sigmoid(),
			)

	def forward(self, feat):
		out = self.net(feat)
		return out

class RMSELoss(nn.Module):
	def __init__(self):
		super(RMSELoss, self).__init__()
		self.mse = nn.MSELoss()
		
	def forward(self,yhat,y):
		return torch.sqrt(self.mse(yhat,y))
