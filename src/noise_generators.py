import torch

class GaussianNoise:

	def __init__(self, sigma: float, shape: tuple = (1,)):
		
		self.mean = torch.zeros(shape) 
		self.sigma = sigma

	def generate(self):
		
		return torch.normal(mean = self.mean, std = self.sigma)


	def reset(self):

		pass

class UniformNoise:
	
	def __init__(self, sigma: float, shape: tuple = (1,)):
		
		self.sigma = sigma
		self.shape = shape

	def generate(self):
		
		return torch.rand(size = self.shape)*self.sigma

	def reset(self):
		
		pass


class WienerProcess:
		
	def __init__(self, stepsize: float = 0.001, shape: tuple = (1,)):

		self.shape = shape
		self.W = torch.zeros(shape)
		self.dt = stepsize

	def generate(self):

		ret = self.W.detach().clone()
		self.W = torch.normal(mean = self.W, std = self.dt**0.5)
		return ret

	def reset(self):
		
		self.x = torch.zeros(self.shape)


class OrnsteinUhlenbeckProcess:

	def __init__(self, stepsize: float = 1.00, shape: tuple = (1,), theta = 0.01, sigma = 0.2, mu = 0.0):

		self.shape = shape
		self.x = torch.zeros(shape)
		self.mean = torch.zeros(shape)
		self.theta = theta
		self.sigma = sigma
		self.mu = mu
		self.dt = stepsize

	def generate(self):
		
		ret = self.x.detach().clone()
		self.x += self.theta * (self.mu - self.x) * self.dt + torch.normal(mean = self.mean, std = self.dt**0.5*self.sigma)
		
		return ret

	def reset(self):

		self.x = torch.zeros(self.shape)



if __name__ == "__main__":
	
	import matplotlib.pyplot as plt

	process = OrnsteinUhlenbeckProcess(shape = (1,))

	x = [list(process.generate()) for i in range(100)]
	process.reset()
	y = [list(process.generate()) for i in range(100)]
	plt.plot(x)
	plt.plot(y)
		
	plt.show()
