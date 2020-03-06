import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class LSTM(nn.Module):
	def __init__(self, input_dim=300, hidden_dim=64, batch_size=1, output_dim=1, num_layers=2, bidirectional=True):
		super(LSTM, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.output_dim = output_dim

		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, self.bidirectional)
		# if(bidirectional):
		# 	self.linear = nn.Linear(self.hidden_dim+360, 64)
		# else:
		# 	self.linear = nn.Linear(self.hidden_dim+360, 64)

		self.linear = nn.Linear(self.hidden_dim, 64)

		self.linear2 = nn.Linear(64, 1)
		self.relu = nn.LeakyReLU(0.2)

	def forward(self, input, market_features):
		# input : (seq_len, batch, input_dim)

		out, (h, c) = self.lstm(input)
		out = self.linear(out[-1])
		# out = self.linear(torch.cat((out[-1],market_features), 1))
		out = self.relu(out)
		out = self.linear2(out)
		return out

class RMSELoss(nn.Module):
	def __init__(self):
		super(RMSELoss, self).__init__()
		self.mse = nn.MSELoss()
		
	def forward(self,yhat,y):
		return torch.sqrt(self.mse(yhat,y))
