import gym
import os
import argparse
import torch
import marlenvs
from marlenvs.wrappers import NormalizeActWrapper, NormalizeObsWrapper, NormalizeRewWrapper
from agent import Agents
import numpy as np
import time
from utils import AverageValueMeter, Parameters
from networks import Actor, Actor2, MADDPGCritic, MADDPGCritic2, MADDPGCritic3, MADDPGCritic4

def test(params):

	if params.env == "navigation":
		env = gym.make("Navigation-v0", n_agents=params.n_agents, world_size=5, max_steps=100, tau=1.0, hold_steps=100)
		continuous = True
	elif params.env == "navigation_prepos":
		init_landmark_pos = np.array([[(i+1)*(params.n_agents*5-0.5)/params.n_agents, 0.5] for i in range(params.n_agents)])
		init_agent_pos = np.array([[(i+1)*(params.n_agents*5-0.5)/params.n_agents, params.n_agents*5 - 0.5] for i in range(params.n_agents)])	
		env = gym.make("Navigation-v0", n_agents=params.n_agents, world_size=5, max_steps=100, tau=1.0, hold_steps=100, init_agent_pos=init_agent_pos, init_landmark_pos=init_landmark_pos)
		continuous = True
	elif params.env == "switch":
		env = gym.make("Switch-v0", height=5, width=10, view=3, flatten_obs = True)
		continuous = False
	elif params.env == "two_step":
		env = gym.make("TwoStep-v0")
		continuous = False

	if params.normalize_actions:
		env = NormalizeActWrapper(env)
	if params.normalize_observations:
		env = NormalizeObsWrapper(env)

	if params.normalize_rewards == "-1to0":
		env = NormalizeRewWrapper(env, high = 0.0, low = -1.0, random_policy_zero=False)
	elif params.normalize_rewards == "random_policy_zero":
		env = NormalizeRewWrapper(env, high = 0.0, low = -1.0, random_policy_zero=True)
	

	act_dim = env.get_act_dim()
	obs_dim = env.get_obs_dim()

	# networks
	if params.actor_type == "shared":
		actor = Actor(act_dim=act_dim, obs_dim=obs_dim, history=params.history, hidden_dim=64)
	elif params.actor_type == "independent":
		actor = Actor2(act_dim=act_dim, obs_dim=obs_dim, history=params.history, hidden_dim=64, n_agents=params.n_agents)
	if params.critic_type == "n2n":
		critic = MADDPGCritic(n_agents=params.n_agents, act_dim=act_dim, obs_dim=obs_dim, history=params.history, hidden_dim=100)
	elif params.critic_type == "n21":
		critic = MADDPGCritic2(n_agents=params.n_agents, act_dim=act_dim, obs_dim=obs_dim, history=params.history, hidden_dim=100)
	elif params.critic_type == "single_q":
		critic = MADDPGCritic3(n_agents=params.n_agents, act_dim=act_dim, obs_dim=obs_dim, history=params.history, hidden_dim=100)
	elif params.critic_type == "independent":
		critic = MADDPGCritic4(n_agents=params.n_agents, act_dim=act_dim, obs_dim=obs_dim, history=params.history, hidden_dim=100)
	
	optim = torch.optim.Adam

	agents = Agents(actor=actor, critic=critic, optim=optim, n_agents=params.n_agents, obs_dim=obs_dim, act_dim=act_dim, sigma=params.exploration_noise,
					lr_critic=params.lr_critic, lr_actor=params.lr_actor, gamma=params.discount, tau=params.soft_update_tau,
					history=params.history, batch_size=params.batch_size, continuous=continuous)

	agents.load(params.weights)
	agents.eval()

	episode_return = AverageValueMeter()
	for episode in range(params.test_episodes):
		done = 0
		e_return = 0
		obs = agents.buffer.history.clear()
		obs = env.reset()
		agents.buffer.history.store(torch.Tensor(obs))
		env.render()
		time.sleep(0.05)
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
			env.render()
			time.sleep(0.05)
		episode_return + e_return
		env.close()

	print(f"Episode return: {episode_return}")

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
	parser.add_argument('--test_episodes', type=int, help='How many episodes to test to take average return.')
	parser.add_argument('--run_folder', type=str, help='Which saved weights to test (foldername).', required=True)
	parser.add_argument('--weights', type=int, help='Which excact weights to test (foldername).')
	parser.add_argument('--render', type=str, help='If to render.')

	args = parser.parse_args()

	args.cfg = os.path.join(args.run_folder, "parameters.json")
	
	if args.weights == None:
		try:
			args.weights = sorted([(int("".join(filter(str.isdigit, f))),f) for f in os.listdir(args.run_folder) if 'batch' in f])[-1][1]
		except IndexError:
			print("\nNo saved weights found. Make sure they are stored like batch_<digits>. Or instead use --weights command to link to them.\n")
		args.weights = os.path.join(args.run_folder, args.weights)

	params = Parameters(args.cfg)

	params.overload(args, ignore=['cfg'])

	params.fix()
	
	test(params)
