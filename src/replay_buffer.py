from collections import deque
import torch
import random
import pickle

class ReplayBuffer:
	"""
	ReplayBuffer to store transitions of the agents
	"""

	def __init__(self, obs_shape: tuple, act_shape: tuple, history: int = 0, max_length: int = 100000):
	
		self.obs_shape = obs_shape
		self.act_shape = act_shape
		self.memory = deque([], maxlen=max_length)
		self.history = HistoryBuffer(obs_shape, history)

	def store(self, act: torch.Tensor, reward: int, next_obs: torch.Tensor, done: bool):
			
		if next_obs.shape != self.obs_shape:
			raise ValueError(f"Observation has wrong shape. Expected shape of {self.obs_shape} but got {next_obs.shape}")
	
		if act.shape != self.act_shape:
			raise ValueError(f"Action has wrong shape. Expected shape of {self.act_shape} but got {act.shape}")

		if not isinstance(done, bool) and done not in [0, 1]:
			raise ValueError(f"done needs to be bool or int (0, 1). Got {type(done)}.")
	
		if not isinstance(reward, int) and  not isinstance(reward, float) and reward != None:
			raise ValueError(f"Reward needs to be either int, float or None (for first observation). Got {type(reward)}.")
			
		if reward == None:
			raise ValueError("Reward can only be None at first observation of episode.")

		self.history.store(next_obs)
		
		obs = self.history.get_prev_obs()
		next_obs = self.history.get_new_obs()
	
		self.memory.appendleft((obs, act, reward, next_obs, done))
		
		if done:
			self.history.clear()


	def sample_batch(self, batchsize: int) -> torch.Tensor:
		
		if batchsize > len(self.memory):
			raise ValueError(f"Batchsize of {batchsize} too big. ReplayBuffer currently only holds {len(self.memory)} elements.")
		batch = random.sample(self.memory, k=batchsize)
		ordered = list(zip(*batch))
		states, actions, next_states = [torch.stack(elem) for i, elem in enumerate(ordered) if (i != 2 and i != 4)]
		rewards = torch.Tensor(list(ordered[2]))
		dones = torch.Tensor(list(ordered[4]))
		return states, actions, rewards, next_states, dones

	def __getitem__(self, key):
		return self.memory[key]

	def __len__(self):
		return len(self.memory)


class HistoryBuffer:

	def __init__(self, obs_shape: tuple, history: int = 0):

		self.memory = deque([], maxlen=history+2) # to rember previous and current state
		self.obs_shape = obs_shape

	def store(self, obs: torch.Tensor):
		
		if obs.shape != self.obs_shape:
			raise ValueError(f"Observation has wrong shape. Expected shape of {self.obs_shape} but got {obs.shape}")

		# fill up history if first observation
		if len(self.memory) == 0:
			while(len(self.memory) < self.memory.maxlen):
				self.memory.appendleft(obs)
			return
	
		self.memory.appendleft(obs)
		 
	def get_new_obs(self) -> torch.Tensor:
		return torch.cat([t.unsqueeze(0) for t in list(self.memory)[:-1]])
		

	def get_prev_obs(self) -> torch.Tensor:
		return torch.cat([t.unsqueeze(0) for t in list(self.memory)[1:]])
	
	def clear(self):
		self.memory.clear()
