import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
	'''The cirtic network used by the agent.'''

	def __init__(self,state_size,action_size,seed=0):
		'''Initlise and defined the model.

		Parameters
		----------
		state_size : int
			The Dimension of each state.

		action_size : int
			The dimension of each action

		seed : int
			The random seed used.
		'''
		super(Critic,self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size,256)
		self.fc2 = nn.Linear(256+action_size,128)
		self.fc3 = nn.Linear(128,1)

	def forward(self,state,action):
		'''Build the network that estimates the value function for the state action pair.

		The model estimates the value function for the current state and action pair.
		Similar to the DDQN paper, the action is added to the last hidden layer.
		Added the action to the final hidden layer would speed up training if there where multiple hidden layers.
		This this case the majority of the hidden layers determinig high level feature representations of the state.
		The features are then added to the action in the finial state.
		This is the general idea behind transfer learning.
		Given that this model only has one hidden layer, it is done to more accurately recreate the architecture mentioned in the DDQN paper.

		Parameters
		----------
		state : array_like
			The current state.

		action :  array_like
			The current action that was taken

		Returns
		-------
		Value : float
			The estimate value of the value function for the current state action pair.
		'''
		x = F.relu(self.fc1(state))
		x1 = torch.cat((x,action.float()),dim=1)
		return self.fc3(F.relu(self.fc2(x1)))


