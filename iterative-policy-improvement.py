import numpy as np

def rent(): return np.array([np.random.poisson(3), np.random.poisson(4)])
def ret(): return np.array([np.random.poisson(3), np.random.poisson(2)])

discount = 0.9

# Able to move max of 5 cars per night, 
# loc_a to loc_b first one is free, everything else is 2
# More than 10 cars at 1 location means 4$ charge, max 20
MAX_CARS = 20
COST_MOVE = 2
COST_STORE = 4

DELTA_LIM = 20

# Negative means move from loc_a, positive means move to loc_a, 0 means do nothing
actions = range(-5, 6)

# Initalize
V_s = np.zeros([21,21])
policy = np.zeros([21,21])

def reward(state, action):
	state += np.array([action, -action])
	state += ret()
	state = np.clip(state, 0, 20)
	sold = np.minimum(state, rent())
	state -= sold
	return np.sum(sold) * 10, int(state[0]), int(state[1])

def policy_eval():
	delta = DELTA_LIM + 1
	while delta > DELTA_LIM:
		delta = 0
		for i in range(V_s.shape[0]):
			for j in range(V_s.shape[1]):
				v = V_s[i][j]
				r, s, t = reward([i,j], policy[i][j])
				V_s[i][j] = r + discount*V_s[s][t]
				delta = max(r, abs(V_s[i][j] - v))

def policy_improve():
	stable = True
	for i in range(V_s.shape[0]):
		for j in range(V_s.shape[1]):
			old_action = policy[i][j]
			rewards = []
			for a in actions:
				r, s, t = reward([i,j], a)
				rewards.append(r + discount*V_s[s][t])
			rewards = np.array(rewards)
			print(rewards)
			policy[i][j] = np.argmax(rewards) - 5
			print(old_action, policy[i][j])
			if rewards[int(old_action) + 5] != rewards[int(policy[i][j]) + 5]:
				stable = False
	return stable

stability = False
iteration = 0
while not stability:
	iteration += 1
	print("Iteration #:" + str(iteration))
	policy_eval()
	stability = policy_improve()
print(V_s)
print(policy)


