import numpy as np
import random
import torch
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
	'''Fixed-size bugger used to store experience tuples
	
	The experience tuples are stored in a double ended queue.
	As new experiences are added the old ones are removed once the buffer size is reached.
	Experiences are uniformly sampled from the queue.
	Experiences are saved as named tuples of (state,action,reward,next_state,done)
	'''	

	def __init__(self, action_size, buffer_size, batch_size, seed=0):
		'''Initlise the Replay Buffer.


		Parameters
		----------
		action_size : int
			The dimenstion of each action
		
		buffer_size : int
			The maximum size of the buffer.

		batch_size : int
			The size of the batch to return when the ReplayBuffer is sampled.

		seed : int
			The random seed used in random number generation.
		'''
		self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		random.seed(seed)
		self.experience = namedtuple("Experience", field_names=['state','action','reward','next_state','done'])

	def add(self, state, action, reward, next_state, done):
		'''Add an experience to the Replay Buffer.
		
		A named tuple is created and added the the deque.	

		Parameters
		----------
		state : array_like
			The current state.
		
		action : array_like 
			The action that was taken.

		reward : int
			The reward that was received.

		next_state : array_like
			The next state.

		done : boolean
			True if the episode is completed, else False
		'''
		e = self.experience(state,action,reward,next_state,done)
		self.memory.append(e)

	def sample(self):
		''' Uniformly samples the Replay Buffer to return BATCH_SIZE samples of experiences.

		Returns
		-------
		experiences : Tuple[torch.Variable]
			A name tuple of length BATCH_SIZE containing states, actions, rewards, next_actions and dones.
		'''
		sampled_experiences = random.sample(self.memory,k=self.batch_size)
		states = torch.from_numpy(np.vstack([e.state for e in sampled_experiences if e is not None])).float().to(device)
		actions = torch.from_numpy(np.vstack([e.action for e in sampled_experiences if e is not None])).float().to(device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in sampled_experiences if e is not None])).float().to(device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in sampled_experiences if e is not None])).float().to(device)
		dones = torch.from_numpy(np.vstack([e.done for e in sampled_experiences if e is not None]).astype(np.uint8)).float().to(device)
		return (states, actions, rewards, next_states, dones)

	def __len__(self):
		'''Gets the current size of the internal memory.

		Returns
		-------
		length : int
			The current length of the internal memory.		
		'''
		return len(self.memory)
