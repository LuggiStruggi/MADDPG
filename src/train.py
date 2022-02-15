import gym
import marlenvs
from marlenvs.wrappers import NormalizeActWrapper, NormalizeObsWrapper, NormalizeRewWrapper
import os
import time
from datetime import datetime
import argparse
import torch
import numpy as np
import runnamegen
from agent import Agents
from networks import Actor, MADDPGCritic, MADDPGCritic2
from utils import AverageValueMeter, Parameters, CSVLogger


def train(params):

	# choose environment
	if params.env == "navigation":
		env = gym.make("Navigation-v0", n_agents=params.n_agents, world_size=5, max_steps=100, tau=1.0, hold_steps=100)
		continuous = True
	elif params.env == "switch":
		env = gym.make("Switch-v0", height=5, width=10, view=3, flatten_obs = True)
		continuous = False
	elif params.env == "two_step":
		env = gym.make("TwoStep-v0")
		continuous = False

	# normalizations
	if params.normalize_actions:
		env = NormalizeActWrapper(env)
	if params.normalize_observations:
		env = NormalizeObsWrapper(env)
	if params.normalize_rewards:
		env = NormalizeRewWrapper(env, high = 0.0, low = -1.0)
	
	# get dimensions
	act_dim = env.get_act_dim()
	obs_dim = env.get_obs_dim()

	# networks
	actor = Actor(act_dim=act_dim, obs_dim=obs_dim, history=params.history, hidden_dim=64)
	if params.critic_type == "n2n":
		critic = MADDPGCritic(n_agents=params.n_agents, act_dim=act_dim, obs_dim=obs_dim, history=params.history, hidden_dim=100)
	elif params.critic_type == "n21":
		critic = MADDPGCritic2(n_agents=params.n_agents, act_dim=act_dim, obs_dim=obs_dim, history=params.history, hidden_dim=100)
	if params.optim == "SGD":
		optim = torch.optim.SGD
	elif params.optim == "Adam":
		optim = torch.optim.Adam

	# make agents
	agents = Agents(actor=actor, critic=critic, optim=optim, n_agents=params.n_agents, obs_dim=obs_dim, act_dim=act_dim, sigma=params.exploration_noise,
					lr_critic=params.lr_critic, lr_actor=params.lr_actor, gamma=params.discount, tau=params.soft_update_tau,
					history=params.history, batch_size=params.batch_size, continuous=continuous)

	# make directory to log	
	log_dir = os.path.join("training_runs", params.run_name)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	params.save_to_file(os.path.join(log_dir, "parameters.json"))

	print(params)

	loss_logger = CSVLogger(os.path.join(log_dir, "losses.csv"), header=["actor loss", "actor loss std", "critic loss",
																		 "critic loss std", "average Q", "average Q std",
																		 "batches trained", "transitions trained",
																		 "episodes gathered", "transitions made"], log_time=True)

	test_logger = CSVLogger(os.path.join(log_dir, "tests.csv"), header=["average episode return", "avg ep ret std", "batches trained",
																		"transitions trained", "episodes gathered", "transitions made"], log_time=True)
	
	# main experiment part
	episode_counter = 0
	batch_counter = 0
	transitions_made_counter = 0
	transitions_trained_counter = 0

	while batch_counter < params.total_batches:
		
		# gather data -----------------------------------------------------
		for episode in range(params.gather_episodes):
			done = 0
			obs = env.reset()
			agents.buffer.history.store(torch.Tensor(obs))
			while(not done):
				with torch.no_grad():
					obs = agents.buffer.history.get_new_obs()
					obs = obs=torch.swapaxes(obs, 0, 1)
					act = agents.act(obs=obs, deterministic=False).squeeze()
					if params.n_agents == 1:
						act = act.unsqueeze(dim = 0) # 1 agent case
				n_obs, rew, done, _ = env.step(act.numpy())
				agents.buffer.store(act, rew, torch.Tensor(n_obs), done)
				transitions_made_counter += 1
			episode_counter += 1

		# train ------------------------------------------------------------
		
		# enough transitions for one batch required
		if len(agents.buffer) < params.batch_size:
			continue

		critic_loss = AverageValueMeter()
		actor_loss = AverageValueMeter()
		avg_Q = AverageValueMeter()
		for batch in range(params.train_batches):
			c_loss, a_loss, avq = agents.train_batch(optim_actor = True, update_targets = True)
			actor_loss + a_loss
			critic_loss + c_loss
			avg_Q + avq

			batch_counter += 1
			transitions_trained_counter += params.batch_size

			if batch_counter % params.save_weights_freq == 0:
				save_dir = os.path.join(log_dir, f"batch_{batch_counter}")
				if not os.path.exists(save_dir):
					os.makedirs(save_dir)
				agents.save(save_dir)

			if batch_counter % params.log_loss_freq == 0 or batch_counter % params.test_freq == 0:
				print(f"Episodes Gathered: {episode_counter} Batches Trained: {batch_counter}")

			if batch_counter % params.log_loss_freq == 0:
				print(f"Actor loss: {actor_loss}")
				print(f"Critic loss: {critic_loss}")
				loss_logger.log([actor_loss.mean(), actor_loss.std(), critic_loss.mean(), critic_loss.std(), avg_Q.mean(), avg_Q.std(),
								 batch_counter, transitions_trained_counter, episode_counter, transitions_trained_counter])
				actor_loss.reset()				
				critic_loss.reset()				
				avg_Q.reset()				

			# test ----------------------------------------------------------------
			if batch_counter % params.test_freq == 0:
				
				episode_return = AverageValueMeter()
				for episode in range(params.test_episodes):
					done = 0
					e_return = 0
					obs = agents.buffer.history.clear()
					obs = env.reset()
					agents.buffer.history.store(torch.Tensor(obs))
					while(not done):
						with torch.no_grad():
							obs = agents.buffer.history.get_new_obs()
							obs = obs=torch.swapaxes(obs, 0, 1)
							act = agents.act(obs=obs, deterministic=True).squeeze()
							if params.n_agents == 1:
								act = act.unsqueeze(dim = 0) # 1 agent case
						n_obs, rew, done, _ = env.step(act.numpy())
						agents.buffer.history.store(torch.Tensor(n_obs))
						e_return += rew
					episode_return + e_return
			
				print(f"Episode return: {episode_return}")
				test_logger.log([episode_return.mean(), episode_return.std(), batch_counter, transitions_trained_counter, episode_counter, transitions_made_counter])
				episode_return.reset()	

			if batch_counter % params.log_loss_freq == 0 or batch_counter % params.test_freq == 0:
				print('\n')

if __name__ == '__main__':

	parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
	parser.add_argument('--env', type=str, help='Name of the environment', choices=['navigation', 'two_step', 'switch'])
	parser.add_argument('--normalize_actions', type=bool, help='If to normalize actions.')
	parser.add_argument('--normalize_observations', type=bool, help='If to normalize observations.')
	parser.add_argument('--normalize_rewards', type=bool, help='If to normalize rewards.')
	parser.add_argument('--critic_type', type=str, help='Critic network type', choices=["n2n", "n21"])
	parser.add_argument('--total_batches', type=int, help='Number of batches to train in total.')
	parser.add_argument('--n_agents', type=int, help='Number of agents.')
	parser.add_argument('--exploration_noise', type=float, help='Exploraition noise of agent.')
	parser.add_argument('--lr_critic', type=float, help='Learning rate of critic.')
	parser.add_argument('--lr_actor', type=float, help='Learning rate of actor.')
	parser.add_argument('--optim', type=str, help='The optimizer used', choices = ["SGD", "Adam"])
	parser.add_argument('--discount', type=float, help='Discount factor for episode reward.')
	parser.add_argument('--soft_update_tau', type=float, help='Soft update parameter.')
	parser.add_argument('--history', type=int, help='History length.')
	parser.add_argument('--batch_size', type=int, help='Batch size.')
	parser.add_argument('--gather_episodes', type=int, help='Number of consecutive episodes to gather data.')
	parser.add_argument('--train_batches', type=int, help='Number of consecutive batches trained.')
	parser.add_argument('--save_weights_freq', type=int, help='Frequency (batches) of saving the weights during training.')
	parser.add_argument('--log_loss_freq', type=int, help='Frequency (batches) of logging the loss.')
	parser.add_argument('--test_freq', type=int, help='Frequency (batches) of testing and logging the agent performance.')
	parser.add_argument('--test_episodes', type=int, help='How many episodes to test to take average return.')

	parser.add_argument('--cfg', type=str, help='path to json config file',
						default='parameter_files/default.json')
	
	args = parser.parse_args()
	
	args.run_name = runnamegen.generate("_")
	
	now = datetime.now()
	args.date = now.strftime("%d/%m/%Y")
	args.time = now.strftime("%H:%M:%S")

	params = Parameters(args.cfg)

	params.overload(args, ignore=['cfg'])
	
	params.fix()
	
	train(params)		
