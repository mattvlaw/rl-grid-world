
import environment

gridW, gridH = 4, 3
start_pos = (0, 0)
end_positions = [(3, 1), (3, 2)]
end_rewards = [-10.0, 10.0]
blocked_positions = [(1,1)]
default_reward= -1.0

scale=100
env = environment.Environment(gridW, gridH, end_positions, end_rewards, blocked_positions, start_pos, default_reward, scale)

'''
States are indexed (X, Y), starting from the bottom left corner, like in Russel & Norvig

	They are flattened bottom-left, going right, and up, i.e.:
	(0,1)=3, (1,0)=4, (2,0)=5
	(0,0)=0, (1,0)=1, (2,0)=2

	Actions are indexed from 0 (North) and then clockwise
'''
class PolicyIteration:
    '''
        States are [0,1,2,3,4,5]
        Actions are [0,1,2,3]

    '''

    def __init__(self,num_states,num_actions):
        self.states = range(num_states)
        self.actions = range(num_actions)
        self.values = np.random.rand(num_states)
        self.policy = np.random.rand(num_states,num_actions)
        self.threshold = 0.1
    def policy_evaluation(self):
        delta = self.threshold
        while delta >= self.threshold:
            delta = 0
            for state in self.states:
                old_val = self.values[state]
                # update self.values[state]
                # sum over s' and r given state and policy
                for action in self.actions:
                    next_state, reward = env.sa_eval(state,action)
                    
    def get_action_from_policy(self,state):
        return np.argmax(self.policy[state])

    def policy_improvement(self):
        policy_stable = True
        for state in range(len(values)):
            old_action = self.get_action_from_policy(state)
            # try each action from the given state and get a next state and a reward


# policy is a definition of probabilities of taking each action in each state
# that is, it should be state X action array
def iterative_policy_evaluation(policy, threshold):
	delta = threshold # just to enter the loop
	num_states,num_actions = policy.shape
	values = np.random.rand(num_states) # ignore terminal states for now
	while delta >= threshold:
		delta = 0
		for i in range(num_states):
			v = values[i]
			values[i] = 0
			for a in policy[i]:
				for 
				values[i] += a*
				