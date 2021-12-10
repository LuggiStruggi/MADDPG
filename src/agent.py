import numpy as np
import torch
from torch import nn
from replay_buffer import ReplayBuffer
from networks import Actor, MADDPGCritic



class Agents:

	def __init__(self, n_agents: int, obs_dim: int, act_dim: int, actor: nn.Module, critic: nn.Module, sigma: float, alpha: float, tau: float = 1.0, history: int = 0, batch_size: int = 256):
		
		self.actor = Actor(act_dim=act_dim, obs_dim=obs_dim, history=history)
		self.actor_target = Actor(act_dim=act_dim, obs_dim=obs_dim, history=history)
		self.actor_target.load_state_dict(self.actor.state_dict())

		self.critic = MADDPGCritic(n_agents=n_agents, act_dim=act_dim, obs_dim=obs_dim, history=history)
		self.critic_target = MADDPGCritic(n_agents=n_agents, act_dim=act_dim, obs_dim=obs_dim, history=history)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_loss = nn.MSELoss()

		self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.01)
		self.actor_optim = torch.optim.Adam(self.critic.parameters(), lr=0.01)

		self.buffer = ReplayBuffer(obs_shape=(n_agents, obs_dim), act_shape=(n_agents, act_dim), history=history)	
		self.sigma = sigma
		self.tau = tau
		self.alpha = alpha

		# the shapes of the actual tensors (without batchsize in dim 0)
		self.obs_shape = (history+1, n_agents, obs_dim)
		self.act_shape = (n_agents, act_dim)


	def act(self, obs: torch.Tensor, deterministic: bool = True, discretized: bool = False, target: bool = False) -> torch.Tensor:
		
		noise = torch.randn(self.act_shape)*self.sigma
		act = (self.actor_target(obs) if target else self.actor(obs)) + (noise if deterministic else 0)
		act = torch.clamp(act, min=0, max=1)
		
		if discretized:
			act = torch.argmax(act, dim=-1)

		return act


	def train_batch(self, optim_actor: bool = False):

		s, a, r, ns, d = self.buffer.sample_batch(self.batch_size)
		
		# (batch_size, history+1, n_agents, obs_dim) -> (batch_size, n_agents, history+1, obs_dim)
		s = torch.swapaxes(s, 1, 2)
		ns = torch.swapaxes(s, 1, 2)
			
		critic_loss = self._train_critic(s, a, r, ns, d)
	
		if optim_actor:
			
			actor_loss = self._train_actor(s, a)

		self._update_target(self.critic, self.critic_target)
		self._update_target(self.actor, self.actor_target)


	def _train_critic(s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, ns: torch.Tensor, d: torch.Tensor):
		
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
			
			Q = self.critic(s, a)
		
		self.critic_optim.zero_grad()	
		loss = self.critic_loss(Q, y)
		loss.backward()
		self.critic_optim.step()
		
		return loss.detach()

	def _train_actor(s, a):

		self.actor_optim.zero_grad()
		pi = self.actor(s)
		Q = self.critic(s, pi)
			
		pre_loss = self.alpha/Q.abs().mean().detach()
		loss = -pre_loss * Q.mean() + F.mse_loss(pi, a)
		loss.backward()
		self.actor_optim.step()
		
		return loss.detach()

	def _update_target(self, model: nn.Module, target: nn.Module):
		
		for param, target_param in zip(model.parameters(), target.parameters()):
			target_param.data.copy_(self.tau*param.data + (1.0 - self.tau)*target_param.data)
