import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
	'''The Actor network used by the agent.'''

	def __init__(self,state_size,action_size,seed=0):
		'''Initlise and defined the model.

		Parameters
		----------
		state_size : int
			The Dimension of each state.

		action_size : int
			The number of actions action.

		seed : int
			The random seed used.
		'''
		super(Actor,self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size,256)
		self.fc2 = nn.Linear(256,128)
		self.fc3 = nn.Linear(128,action_size)

	def forward(self,state):
		'''Build the network that estimates the action to be taken.
		
		Estimate the action to be taken. Each action has a range from 
		-1 to 1 so the tanh activation function is used.

		Parameters
		----------
		state : array_like
			The current state.


		Returns
		-------
		action : array_like
			The action to be taken by the agent.
		'''
		return torch.tanh(self.fc3(F.relu(self.fc2(F.relu(self.fc1(state))))))
