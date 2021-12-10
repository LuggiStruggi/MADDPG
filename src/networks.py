import torch
from torch import nn

class Actor(nn.Module):

	def __init__(self, act_dim: int, obs_dim: int, history: int = 0, hidden_dim: int = 32):
		
		self.linear1 = nn.Linear(in_features=obs_dim*(history+1), out_features=hidden_dim)
		self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
		self.linear3 = nn.Linear(in_features=hidden_dim, out_features=act_dim)
		self.activation = nn.ReLU()
		self.softmax = nn.Softmax()
		self.max_action_values = max_action_values

	def forward(self, obs: torch.Tensor) -> torch.Tensor:
		"""
		obs -> (batch_size, n_agents, history+1, obs_dim)
		"""
		x = torch.flatten(obs, start_dim=2)
		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)
		x = self.activation(x)
		x = self.linear3(x)
		return self.softmax(x)


class MADDPGCritic(nn.Module):
	"""
	Critic which takes observation-action pairs of all agents and returns specific q values for each 
	"""
	
	def __init__(self, n_agents: int, act_dim: int, obs_dim: int, history: int = 0, hidden_dim: int = 256):
		super(MADDPGCritic, self).__init__()	
		in_features = n_agents*((history+1)*obs_dim+act_dim)
		self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
		self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
		self.Q_output = nn.Linear(in_features=hidden_dim, out_features=n_agents)
		self.activation = nn.ReLU()


	def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
		"""
		obs -> (batch_size, n_agents, history+1, obs_dim)
		act -> (batch_size, n_agents, act_dim) 
		"""
		# flatten obervation and action then concatenate -> (batch_size, n_agents*((history+1)*obs_dim + act_dim))
		x = torch.cat((torch.flatten(obs, start_dim=1), torch.flatten(act, start_dim=1)), dim=1)
		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)
		x = self.activation(x)
		x = self.linear3(x)

		return x

class MADDPGCritic_2(nn.Module):
	"""
	Critic which takes observation-action pairs of all agents and an index and returns q value for that specific agent
	"""
	
	def __init__(self, n_agents: int, act_dim: int, obs_dim: int, history: int = 0, hidden_dim: int = 256):
		super(MADDPGCritic, self).__init__()	
		in_features = n_agents*((history+1)*obs_dim+act_dim)
		self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
		self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
		self.Q_output = nn.Linear(in_features=hidden_dim, out_features=n_agents)
		self.activation = nn.ReLU()


	def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
		"""
		obs -> (batch_size, n_agents, history+1, obs_dim)
		act -> (batch_size, n_agents, act_dim) 
		"""
		# flatten obervation and action then concatenate -> (batch_size, n_agents*((history+1)*obs_dim + act_dim))
		x = torch.cat((torch.flatten(obs, start_dim=1), torch.flatten(act, start_dim=1)), dim=1)
		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)
		x = self.activation(x)
		x = self.linear3(x)

		return x
