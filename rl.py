import environment
import agent
import numpy as np
import random

# Environment ---------
gridH, gridW = 4, 8
start_pos = (3, 0)
end_positions = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]
end_rewards = [-100.0, -100.0, -100.0, -100.0, -100.0, -100.0, 100.0]
blocked_positions = []
default_reward= -1.0
scale=100

# gridH, gridW = 9, 7
# start_pos = None
# end_positions = [(0, 3), (2, 4), (6, 2)]
# end_rewards = [20.0, -50.0, -50.0]
# blocked_positions = [(2, i) for i in range(3)] + [(6, i) for i in range(4, 7)]
# default_reward = -0.1


env = environment.Environment(gridH, gridW, end_positions, end_rewards, blocked_positions, start_pos, default_reward, scale)

# Agent -------------
alpha = 0.2
epsilon = 0.25
discount = 0.99
action_space = env.action_space
state_space = env.state_space

# agent = agent.QLearningAgent(alpha, epsilon, discount, action_space, state_space)
# agent = agent.EVSarsaAgent (alpha, epsilon, discount, action_space, state_space)
agent = agent.SimpleSarsaAgent (alpha, epsilon, discount, action_space, state_space)

# Learning -----------

# Get initial state from environment

# Vanilla SARSA  

while(True):

	#input ("Episode ")
	env.reset_state()
	state = env.get_state()
	action = agent.get_action(state, range(action_space))

	while (True):
		#input ("Step ")
		next_state, reward, done = env.step(action)
		next_action = agent.get_action(next_state, range(action_space))
	
		#print (state, action, reward, next_state, next_action, done)
		#print ("Qt", agent.qvalues[state, action], "Qt+1", agent.qvalues[next_state, next_action], "Error", reward + discount * agent.qvalues[next_state, next_action] - agent.qvalues[state, action])
		agent.qvalues[state, action] = agent.qvalues[state, action] + alpha * ( reward + discount * agent.qvalues[next_state, next_action] - agent.qvalues[state, action])

		env.render(agent.qvalues, agent.get_policy_action)

		state = next_state
		action = next_action
		
		if done == True:	
			break

