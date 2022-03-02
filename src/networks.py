import torch
from torch import nn

class Actor(nn.Module):

	def __init__(self, act_dim: int, obs_dim: int, history: int = 0, hidden_dim: int = 32, min_act = 0.0, max_act  = 1.0):
		super(Actor, self).__init__()	
		
		self.linear1 = nn.Linear(in_features=obs_dim*(history+1), out_features=hidden_dim)
		self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
		self.linear3 = nn.Linear(in_features=hidden_dim, out_features=act_dim)
		self.activation = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, obs: torch.Tensor) -> torch.Tensor:
		"""
		obs -> (batch_size, n_agents, history+1, obs_dim)
		returns -> (batch_size, n_agents, act_dim)
		"""
		x = torch.flatten(obs, start_dim=-2)
		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)
		x = self.activation(x)
		x = self.linear3(x)

		return self.sigmoid(x)


class Actor2(nn.Module):
	"""
	Multiple indipendent actor networks (one for each agent)
	"""
	
	def __init__(self, act_dim: int, obs_dim: int, n_agents: int,  history: int = 0, hidden_dim: int = 32, min_act: float  = 0.0, max_act: float  = 1.0):
		super(Actor2, self).__init__()	
		
		self.actors = nn.ModuleList([Actor(act_dim=act_dim, obs_dim=obs_dim, history=history, hidden_dim=hidden_dim, min_act=min_act, max_act=max_act) for i in range(n_agents)])

	def forward(self, obs: torch.Tensor) -> torch.Tensor:

		return torch.stack([actor(obs[..., i, :, :]) for i, actor in enumerate(self.actors)], dim = -2)
		
			

class MADDPGCritic(nn.Module):
	"""
	Critic which takes observation-action pairs of all agents and returns specific q values for each 
	"""
	
	def __init__(self, n_agents: int, act_dim: int, obs_dim: int, history: int = 0, hidden_dim: int = 32):
		super(MADDPGCritic, self).__init__()	
		in_features = n_agents*((history+1)*obs_dim+act_dim)
		self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
		self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
		self.linear3 = nn.Linear(in_features=hidden_dim, out_features=n_agents)
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


class MADDPGCritic2(nn.Module):
	"""
	Critic which takes observation-action pairs of all agents but of actual agent seperated and returns q value for that specific agent
	"""
	
	def __init__(self, n_agents: int, act_dim: int, obs_dim: int, history: int = 0, hidden_dim: int = 256, shuffle: bool = True):
		super(MADDPGCritic2, self).__init__()	
		
		in_features_other = (n_agents-1)*((history+1)*obs_dim+act_dim)
		in_features_self = ((history+1)*obs_dim+act_dim)
		
		hidden_self = hidden_dim//n_agents
		hidden_other = hidden_dim - hidden_self
	
		self.linear1_other = nn.Linear(in_features=in_features_other, out_features=hidden_other)
		self.linear1_self = nn.Linear(in_features=in_features_self, out_features=hidden_self)

		self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
		self.activation = nn.ReLU()
		
		self.linear3 = nn.Linear(in_features=hidden_dim, out_features=1)

		self.shuffle = shuffle


	def forward_single(self, obs_self: torch.Tensor, act_self: torch.Tensor, obs_other: torch.Tensor, act_other: torch.Tensor) -> torch.Tensor:
		"""
		obs_other -> (batch_size, n_agents-1, history+1, obs_dim)
		act_other -> (batch_size, n_agents-1, act_dim) 
		obs_self -> (batch_size, 1, history+1, obs_dim)
		act_self -> (batch_size, 1, act_dim)
		"""
		if self.shuffle:
			bs = obs_other.shape[0]
			n_agents = obs_other.shape[1]
			perm = torch.stack([torch.randperm(n_agents) for i in range(bs)])
			obs_other = torch.stack([obs_other[i, perm[i]] for i in range(bs)])
			act_other = torch.stack([act_other[i, perm[i]] for i in range(bs)])

		# flatten obervation and action then concatenate -> (batch_size, n_agents*((history+1)*obs_dim + act_dim))
		x_other = torch.cat((torch.flatten(obs_other, start_dim=1), torch.flatten(act_other, start_dim=1)), dim=1)
		x_self = torch.cat((torch.flatten(obs_self, start_dim=1), torch.flatten(act_self, start_dim=1)), dim=1)
		x_other = self.linear1_other(x_other)
		x_self = self.linear1_self(x_self)
		x_other = self.activation(x_other)
		x_self = self.activation(x_self)
		x = torch.cat((x_self, x_other), dim=1)
		x = self.linear2(x)
		x = self.activation(x)
		x = self.linear3(x)
		
		return x

	def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:

		Q_values = torch.zeros(obs.shape[:2])

		for i in range(obs.shape[1]):
			indicies = torch.tensor([j for j in range(obs.shape[1]) if j != i])
			single = self.forward_single(torch.index_select(obs, 1, torch.tensor([i])), torch.index_select(act, 1, torch.tensor([i])),
										 	  torch.index_select(obs, 1, indicies), torch.index_select(act, 1, indicies))
			single = torch.squeeze(single)
			Q_values[:, i] = single
		return Q_values


class MADDPGCritic3(nn.Module):
	"""
	Critic which takes observation-action pairs of all agents and returns one q value for all
	"""
	
	def __init__(self, n_agents: int, act_dim: int, obs_dim: int, history: int = 0, hidden_dim: int = 32):
		super(MADDPGCritic3, self).__init__()	
		in_features = n_agents*((history+1)*obs_dim+act_dim)
		self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_dim)
		self.linear2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
		self.linear3 = nn.Linear(in_features=hidden_dim, out_features=1)
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



class MADDPGCritic4(nn.Module):
	"""
	one critic for each agent
	"""

	def __init__(self, n_agents: int, act_dim: int, obs_dim: int, history: int = 0, hidden_dim: int = 32):
		super(MADDPGCritic4, self).__init__()
	
		self.critics = nn.ModuleList([MADDPGCritic3(n_agents=n_agents, act_dim=act_dim, obs_dim=obs_dim, history=history, hidden_dim=hidden_dim) for i in range(n_agents)])

	def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:

		return torch.stack([critic(obs, act) for i, critic in enumerate(self.critics)], dim=-2)

