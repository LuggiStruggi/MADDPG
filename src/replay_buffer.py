from collections import deque
import torch
import random
import pickle

class ReplayBuffer:
	"""
	ReplayBuffer to store transitions of the 
	"""

	def __init__(self, obs_shape: tuple, act_shape: tuple, history: int = 0, max_length: int = 100000):
	
		self.obs_shape = obs_shape
		self.act_shape = act_shape
		self.history = deque([], maxlen=history+1)
		self.memory = deque([], maxlen=max_length)


	def store(self, obs: torch.Tensor, act: torch.Tensor, reward: int, done: bool):
		
		if obs.shape != self.obs_shape:
			raise ValueError(f"Observation has wrong shape. Expected shape of {self.obs_shape} but got {obs.shape}")

		if act.shape != self.act_shape:
			raise ValueError(f"Action has wrong shape. Expected shape of {self.act_shape} but got {act.shape}")

		if not isinstance(done, bool):
			raise ValueError(f"done needs to be bool. Got {type(done)}.")
	
		if not isinstance(reward, int) and reward != None:
			raise ValueError(f"Reward needs to be either int or None (for first observation). Got {type(reward)}.")
		
		# fill up history if first observation
		if len(self.history) == 0:
			while(len(self.history) < self.history.maxlen):
				self.history.appendleft(obs)
			return
		
		if reward == None:
			raise ValueError("Reward can only be None at first observation of episode.")		

		# get state and next state	
		state = self.get_history()
		self.history.appendleft(obs)
		next_state = self.get_history()
			
		self.memory.appendleft((state, act, reward, next_state, done))
		
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


	def get_history(self) -> torch.Tensor:
		return torch.cat([t.unsqueeze(0) for t in self.history])


	def __getitem__(self, key):
		return self.memory[key] 
		

"""	def save_file(self, path, save_history: bool = False):
		with open(path, "wb") as f:
			pickle.dump({"memory" : list(self.memory), "history" : self.history if save_history else })

	def load_file(self, path):
		pass"""

if __name__ == "__main__":
	n_agents = 2
	obs_dim = 3
	act_dim = 2
	history = 200
	rb = ReplayBuffer(obs_shape=(n_agents, obs_dim), act_shape=(n_agents, act_dim), history=history)
	done = False
	for i in range(100000):
		
		obs = torch.rand((n_agents, obs_dim))
		act = torch.rand((n_agents, act_dim))
		if done == True:
			reward = None
		else:
			reward = random.randint(0, 10)
		done = False
		if i % 20 == 20:
			done = True
		
		rb.store(obs, act, reward, done)

	while(True):
		s, a, r, ns, d = rb.sample_batch(10)

		print(s.shape)
		print(ns.shape)
		print(a.shape)
		print(d.shape)
		print(r.shape)
		input("\n")
