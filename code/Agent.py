import torch
import torch.nn.functional as F
import torch.optim as optim
import copy

import numpy as np
import random
from ReplayBuffer import ReplayBuffer
from Actor import Actor
from Critic import Critic 

ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-3
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 1024 
GAMMA = 0.99
TAU = 1e-3
WEIGHT_DECAY = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPDATE_EVERY = 1

class Agent():
	'''This agent Interacts with the environment to learn a policy that yields the highest commulative reward.
		The agent uses the Deep Deterministic Policy Gradient algorithm'''

	def __init__(self, state_size, action_size, seed=0):
		'''Initlize the Agent.
		
		Parameters
		----------
		state_size : int
			The dimension of each state
		
		action_size : int
			The dimension of each action
		
		seed : int
			The random seed used to generate random numbers.
		'''
		self.state_size = state_size
		self.action_size = action_size
		random.seed(seed)

		#actor gives the best action for given state
		self.actor_local = Actor(state_size, action_size, seed).to(device)
		self.actor_target = Actor(state_size, action_size, seed).to(device)

		#evaluates the action
		self.critic_local = Critic(state_size, action_size, seed).to(device)
		self.critic_target = Critic(state_size, action_size, seed).to(device)

		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LEARNING_RATE)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CRITIC_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

		#Replay Memory
		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

		#Noise
		self.noise = OUNoise(action_size,seed)
		self.t_step = 0


	def step(self, state, action, reward, next_state, done):
		'''Instructs the agent to take a step in the environment.

		Executes each time the agent takes a step in the environment.
		The observed (state, action, reward, next_state, done) tuple is saved in the replay buffer.
		Once enough experiences have been captured the model is trained.
		
		Parameters
		----------
		state : array_like
			The current state.
		
		action : int
			The action that was taken.

		reward : int
			The reward that was received.

		next_state : array_like
			The next state.

		done : boolean
			True if the episode is completed, else False
		'''
		self.memory.add(state, action, reward, next_state, done)
		self.t_step = (self.t_step+1)%UPDATE_EVERY
		if self.t_step == 0:
			if len(self.memory) > BATCH_SIZE:
				experiences = self.memory.sample()
				self.train_model_parameters(experiences)
	
	def get_action(self, state, epsilon=0, add_noise=True):
		'''Gets the action for the given state defined by the current policy.

		The method returns the action to take for the given state given the current policy.
		In order to explore in the continuous space noise is added to the action.
		

		Parameters
		----------
		state : array_like
			The current state.

		epsilon : float
			The epsilon value usedfor epsilon-greedy action selection.

		add_noise : boolean
			Add noise to the action to encourage exploration.

		Returns
		-------
		action : array-like
			The action to take. Each value is between -1 and 1.
		'''
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.actor_local.eval()
		with torch.no_grad():
			action = self.actor_local(state).cpu().data.numpy()
		self.actor_local.train()
		if add_noise:
			action+=self.noise.sample()
		return np.clip(action,-1,1)

	def train_model_parameters(self, experiences):
		'''Update the model parameters using the given batch of experience tuples.

		The models are train via the Actor Critic paradigm.
		The next action is optained fromt he target actor.
		This is then passed to the target critic to obtain the target next state.
		The target current state is calculated via the bellman equations.
		The local critic estimates the next state and is updated accordingly.	
		The local actions predictions the next actions given the current state.
		The loss for the actor is calculated as the ...

		Parameters
		----------
		experiences : Tuple[torch.Variable]
			A name tuple of state, action, reward, next_action and done.
		'''
		states, actions, rewards, next_states, dones = experiences
		
		#Update critic
		next_actions = self.actor_target(next_states)
		Q_next_states = self.critic_target(next_states,next_actions)
		Q_states = rewards + GAMMA*Q_next_states*(1-dones)
		Q_states_estimated = self.critic_local(states,actions)
		critic_loss = F.mse_loss(Q_states_estimated, Q_states)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		
		#Update actor
		actions_pred = self.actor_local(states)
		actor_loss = -self.critic_local(states,actions_pred).mean()
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()	

		self._update_model_parameters(self.critic_local, self.critic_target)     
		self._update_model_parameters(self.actor_local, self.actor_target)     

	def _update_model_parameters(self,local_network, target_network):
		'''Copy the learned local network parameters to the target network.

		This method updates the Target network with the learned network parameters.
		The target parameters are old movd TAU towards the learned local parameters.
		The is done to help redude the amount of harmful correlation by constating moving the target.
		'''
		for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
			target_param.data.copy_(TAU*local_param.data + (1-TAU) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process. This noise is added to the action to promote exploration."""

    def __init__(self, size, seed=0, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
