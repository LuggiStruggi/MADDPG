import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from replay_buffer import ReplayBuffer, HistoryBuffer
from networks import Actor, MADDPGCritic
import os

class Agents:

	def __init__(self, n_agents: int, obs_dim: int, act_dim: int, sigma: float, lr_critic: float, lr_actor: float, gamma: float,
				 tau: float = 1.0, history: int = 0, batch_size: int = 32, continuous: bool = False):
		
		self.actor = Actor(act_dim=act_dim, obs_dim=obs_dim, history=history, hidden_dim=64)
		self.actor_target = Actor(act_dim=act_dim, obs_dim=obs_dim, history=history, hidden_dim=64)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_target.eval()

		self.critic = MADDPGCritic(n_agents=n_agents, act_dim=act_dim, obs_dim=obs_dim, history=history, hidden_dim=100)
		self.critic_target = MADDPGCritic(n_agents=n_agents, act_dim=act_dim, obs_dim=obs_dim, history=history, hidden_dim=100)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_target.eval()
		self.critic_loss = nn.MSELoss()

		self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

		self.buffer = ReplayBuffer(obs_shape=(n_agents, obs_dim), act_shape=(n_agents, act_dim) if continuous else (n_agents,), history=history)	
		self.sigma = sigma
		self.tau = tau
		self.gamma=gamma
		self.batch_size = batch_size

		# the shapes of the actual tensors (without batchsize in dim 0)
		self.obs_shape = (history+1, n_agents, obs_dim)
		self.act_shape = (n_agents, act_dim)
	
		self.eval()


	def act(self, obs: torch.Tensor, deterministic: bool = True, discretized: bool = False, target: bool = False) -> torch.Tensor:
		
		noise = torch.randn(self.act_shape)*self.sigma

		act = (self.actor_target(obs) if target else self.actor(obs)) + (0 if deterministic else noise)
		act = torch.clamp(act, min=0, max=1)
		#act = F.softmax(act, dim=-1)
		
		if discretized:
			act = torch.argmax(act, dim=-1)

		return act


	def train_batch(self, optim_actor: bool = False, update_targets: bool = False):

		s, a, r, ns, d = self.buffer.sample_batch(self.batch_size)
	
		# (batch_size, history+1, n_agents, obs_dim) -> (batch_size, n_agents, history+1, obs_dim)
		s = torch.swapaxes(s, 1, 2)
		ns = torch.swapaxes(ns, 1, 2)

		self.train()

		critic_loss = self._train_critic(s, a, r, ns, d)
		actor_loss = None
	
	
		if optim_actor:	
			actor_loss = self._train_actor(s)

		if update_targets:
			self._update_target(self.critic, self.critic_target)
			self._update_target(self.actor, self.actor_target)

		self.eval()

		return critic_loss, actor_loss

	
	def eval(self):
		self.critic.eval()
		self.actor.eval()

	def train(self):
		self.critic.train()
		self.actor.train()

	def save(self, foldername):
		torch.save(self.critic.state_dict(), os.path.join(foldername, "critic"))
		torch.save(self.actor.state_dict(), os.path.join(foldername, "actor"))

	def load(self, foldername):
		self.critic.load_state_dict(torch.load(os.path.join(foldername, "critic")))
		self.critic_target.load_state_dict(torch.load(os.path.join(foldername, "critic")))
		self.actor.load_state_dict(torch.load(os.path.join(foldername, "actor")))
		self.actor_target.load_state_dict(torch.load(os.path.join(foldername, "actor")))


	def _train_critic(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, ns: torch.Tensor, d: torch.Tensor):
		
		with torch.no_grad():
			na = self.act(obs=ns, target=True)
			nQ = self.critic_target(ns, na)
			
			# s -> (bs, n_agents, his+1, obs_dim)
			# a -> (bs, n_agents, act_dim)
			# Q -> (bs, n_agents)
			# d -> (bs)
			# r -> (bs)
			y = (r + self.gamma*(1.0-d)*nQ.T).T

			# y -> (bs, n_agents)
			
		self.critic_optim.zero_grad()
		Q = self.critic(s, a)
		loss = self.critic_loss(Q, y)
		loss.backward()
		self.critic_optim.step()
		
		return loss.detach().item()

	def _train_actor(self, s):
	
		self.actor_optim.zero_grad()
		pi = self.actor(s)
		Q = self.critic(s, pi)
		loss = -Q.mean()
		loss.backward()
		self.actor_optim.step()
		
		return loss.detach().item()


	def _update_target(self, model: nn.Module, target: nn.Module):
		
		for param, target_param in zip(model.parameters(), target.parameters()):
			target_param.data.copy_(self.tau*param.data + (1.0 - self.tau)*target_param.data)
