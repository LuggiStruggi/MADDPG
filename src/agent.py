import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from replay_buffer import ReplayBuffer, HistoryBuffer
from networks import Actor, MADDPGCritic, MADDPGCritic2
import os
import copy

class Agents:

	def __init__(self, actor, critic, optim, noise_generator, n_agents: int, obs_dim: int, act_dim: int, lr_critic: float, lr_actor: float, gamma: float,
				 tau: float = 1.0, history: int = 0, batch_size: int = 32, continuous: bool = False, independent: bool = False):
		
		self.actor = actor
		self.actor_target = copy.deepcopy(actor)
		self.actor_target.eval()

		self.critic = critic	
		self.critic_target = copy.deepcopy(critic)
		self.critic_target.eval()
		self.critic_loss = nn.MSELoss()

		self.critic_optim = optim(self.critic.parameters(), lr=lr_critic)
		self.actor_optim = optim(self.actor.parameters(), lr=lr_actor)

		self.noise_generator = noise_generator

		self.buffer = ReplayBuffer(obs_shape=(n_agents, obs_dim), act_shape=(n_agents, act_dim) if continuous else (n_agents,), history=history)	
		self.tau = tau
		self.gamma=gamma
		self.batch_size = batch_size

		# the shapes of the actual tensors (without batchsize in dim 0)
		self.obs_shape = (history+1, n_agents, obs_dim)
		self.act_shape = (n_agents, act_dim)
		
		self.independent = independent
	
		self.eval()


	def act(self, obs: torch.Tensor, deterministic: bool = True, discretized: bool = False, target: bool = False) -> torch.Tensor:
			
		noise = self.noise_generator.generate()

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

		critic_loss, avg_Q = self._train_critic(s, a, r, ns, d)
		actor_loss = None
	
	
		if optim_actor:	
			actor_loss = self._train_actor(s)

		if update_targets:
			self._update_target(self.critic, self.critic_target)
			self._update_target(self.actor, self.actor_target)

		self.eval()

		return critic_loss, actor_loss, avg_Q

	
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
			na = self.actor_target(obs=ns)
			nQ = self.critic_target(ns, na)
			
			# s -> (bs, n_agents, his+1, obs_dim)
			# a -> (bs, n_agents, act_dim)
			# Q -> (bs, n_agents)
			# d -> (bs)
			# r -> (bs)
			y = (r + self.gamma*(1.0-d)*nQ.T).T

			# y -> (bs, n_agents)
			
		Q = self.critic(s, a)
		loss = self.critic_loss(Q, y)
		self.critic_optim.zero_grad()
		loss.backward()
		self.critic_optim.step()
		
		return loss.detach().item(), torch.mean(Q).detach().item()

	def _train_actor(self, s):
	
		pi = self.actor(s)
		Q = self.critic(s, pi)
		if self.independent:
			loss = -Q.mean(dim=0)
			for i in range(loss.shape[0] - 1):
				self.actor_optim.zero_grad()
				single_loss = loss[i]
				single_loss.backward(retain_graph=True)
			loss[-1].backward()
			loss = loss.mean()
		else:
			loss = -Q.mean()
			self.actor_optim.zero_grad()
			loss.backward()
		
		self.actor_optim.step()
		
		return loss.detach().item()


	def _update_target(self, model: nn.Module, target: nn.Module):
		
		for param, target_param in zip(model.parameters(), target.parameters()):
			target_param.data.copy_(self.tau*param.data + (1.0 - self.tau)*target_param.data)
