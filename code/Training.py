from collections import deque
from Agent import Agent
from unityagents import UnityEnvironment
import numpy as np
import torch
import pickle

NO_GRAPHICS = True 
GPU_SERVER = False 
TRAIN_MODE = True

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

agents = [Agent(state_size, action_size, seed=0),Agent(state_size, action_size, seed=0)]

scores_window = deque(maxlen=100)
all_scores = []
i_episode = 1
print('\nTraining the Agents...\n')
while True:
		env_info = env.reset(train_mode=TRAIN_MODE)[brain_name]     
		states = env_info.vector_observations                  
		scores = [0]*len(agents)
		while True:
			actions = [agent.get_action(state,add_noise=TRAIN_MODE) for agent,state in zip(agents,states)]
			env_info = env.step(np.concatenate(actions,axis=0))[brain_name]           
			next_states = env_info.vector_observations       
			rewards = env_info.rewards                         
			dones = env_info.local_done
			for i,agent in enumerate(agents):
				agent.step(states[i],actions[i],rewards[i],next_states[i],dones[i])		
			for i,s in enumerate(rewards):
				scores[i] += s
			states = next_states
			if np.any(dones):
				break
		scores_window.append(scores)
		all_scores.append(scores)
		max_score = np.max([np.mean([x[i] for x in scores_window]) for i in range(len(agents))])
		if i_episode % 100 == 0:
			print('Score (max over agents) from episode {}: {:.4f}'.format(i_episode, max_score))
		if  max_score >= 0.5:
			print('The agent solved the environment after {} episodes, with a score of score {:.4f}'.format(i_episode, max_score))
			for i,agent in enumerate(agents):
				torch.save(agent.actor_local.state_dict(), 'checkpoint/{}_actor_{}.pth'.format(i_episode,i))
				torch.save(agent.critic_local.state_dict(), 'checkpoint/{}_critic_{}.pth'.format(i_episode,i))
			with open('scores.pkl','wb') as f:
				pickle.dump(all_scores,f)
			break
		i_episode+=1
env.close()
with open('scores.pkl','wb') as f:
	pickle.dump(all_scores,f)
