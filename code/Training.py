from collections import deque
from Agent import Agent
from unityagents import UnityEnvironment
import numpy as np
import torch
import pickle

NO_GRAPHICS = True 
GPU_SERVER = False 
TRAIN_MODE = True
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

env = UnityEnvironment(file_name='../Tennis_Linux_NoVis/Tennis.x86_64' if GPU_SERVER else '../Tennis.app', no_graphics=NO_GRAPHICS)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
print('The state for the second agent looks like:', states[1])

scores_window = deque(maxlen=100)
all_scores = []
i_episode = 1
print('\nTraining the Agents...\n')

#We can train one agent against itself by inverteing the x coordinate for the position and velocity of the paddle and ball.
#We train one agent but for the second observation we transform the x using [-1,1,-1,1,-1,1,-1,1]*3
agent = Agent(state_size, action_size, seed=0)
state_mask = [-1,1]*12
eps = EPS_START
while True:
		env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]     
		states = env_info.vector_observations                  
		states[1]*=state_mask
		scores = [0]*num_agents
		
		while True:
			
			#Call get action twice on the same agent, the second state is tranmsformed.
			#Actions are relative to the net so no need to flip returned actions
			actions = [agent.get_action(state,eps,add_noise=TRAIN_MODE) for state in states]
			env_info = env.step(np.concatenate(actions,axis=0))[brain_name]           
			next_states = env_info.vector_observations       
			next_states[1]*=state_mask
			rewards = env_info.rewards                         
			dones = env_info.local_done
			
			#Step the agent twice
			for i in range(num_agents):
				agent.step(states[i],actions[i],rewards[i],next_states[i],dones[i])
			
			for i,s in enumerate(rewards):
				scores[i] += s
			states = next_states

			if np.any(dones):
				break

		scores_window.append(scores)
		all_scores.append(scores)
		average_scores = [np.mean([x[i] for x in scores_window]) for i in range(num_agents)]
		max_score = np.max(average_scores)
		
		if i_episode % 100 == 0:
			print('Average score (max over agents) at episode {} : {:.4f}'.format(i_episode, max_score))
			print('Agent 0: {:.4f}\tAgent 1: {:.4f}'.format(*average_scores))

		if  max_score >= 0.5:
			print('The agent solved the environment after {} episodes, with a score of score {:.4f}'.format(i_episode, max_score))
			torch.save(agent.actor_local.state_dict(), 'checkpoint/{}_actor.pth'.format(i_episode))
			torch.save(agent.critic_local.state_dict(), 'checkpoint/{}_critic.pth'.format(i_episode))
			with open('scores.pkl','wb') as f:
				pickle.dump(all_scores,f)
			break

		if i_episode % 1000 == 0:
			print('Saving checkpoint... ')
			torch.save(agent.actor_local.state_dict(), 'actor_checkpoint.pth')
			torch.save(agent.critic_local.state_dict(), 'critic_checkpoint.pth')
			print('Checkpoint saved.')
			with open('scores.pkl','wb') as f:
				pickle.dump(all_scores,f)
		i_episode+=1
		eps = max(EPS_END,EPS_DECAY*eps)

env.close()
with open('scores.pkl','wb') as f:
	pickle.dump(all_scores,f)
