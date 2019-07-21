
# coding: utf-8

# In[ ]:


# This code is based on an exercise in the Reinforcement Learning textbook by Richard Sutton and Andrew Barto
# Link to the textbook: http://incompleteideas.net/book/RLbook2018.pdf
# The exercise is the car rental exercise found in section 4.3 on page 80

# This code uses the optimized matrix version solution, I also include the simpler iterative version for comparison.
# The dependencies for this code are numpy and python 3.6+


# In[ ]:


import numpy as np


# In[3]:


def poisson_prob(l):
  n = 0
  ret = []
  while True:
    ret.append((l**n * np.e ** -l) / np.math.factorial(n))
    if ret[-1] < 0.00005:
      return ret
    n += 1

POISSON_2 = poisson_prob(2)
POISSON_3 = poisson_prob(3)
POISSON_4 = poisson_prob(4)

print(len(POISSON_2), len(POISSON_3), len(POISSON_4))


# In[ ]:


MAX_CARS = 20
NUM_STATES = 21*21
COST_MOVE = 2
COST_STORE = 4
PROFIT = 10

# Positive means move to loc_a
# Negative means move from loc_a
ACTIONS = range(-5, 6)
DELTA_LIM = 0.001
DISCOUNT = 0.9


# In[ ]:


policy = np.zeros([21, 21])

V_s = np.zeros([21, 21])
V_index = np.dstack((
    np.repeat(np.expand_dims(np.array(range(21)), axis=0), 21, axis=0), 
    np.repeat(np.expand_dims(np.array(range(21)), axis=1), 21, axis=1)
)).astype(float)


# In[ ]:


p_row = []
for i, prob_ret_a in enumerate(POISSON_3):
  for j, prob_ret_b in enumerate(POISSON_2):
    for k, prob_rent_a in enumerate(POISSON_3):
      for l, prob_rent_b in enumerate(POISSON_4):
        prob = prob_ret_a * prob_ret_b * prob_rent_a * prob_rent_b
        if prob > 0.00005:
          p_row.append([
            prob,
            i, j, k, l
          ])
p_row = np.array(p_row)
p_table = np.array([[p_row] * 21] * 21)


# In[ ]:


def P(v, action=None):
#   The indices represent the state values (# of cars in loc_a, # of cars in loc_b)
    next_state = np.repeat(np.expand_dims(V_index, 2), p_row.shape[0], axis=2)  # (21, 21, 2628, 2)

#   Add in the returned cars
    next_state += p_table[:,:,:,1:3] 

    if action is None:
#     Apply the current policy
      next_state += np.stack((
          np.repeat(np.expand_dims(policy, axis=2), p_row.shape[0], axis=2), 
          np.repeat(np.expand_dims(-policy, axis=2), p_row.shape[0], axis=2)
      ), 3)
    else:
#     Apply the current action
      next_state += np.array([[[[action, -action]] * p_row.shape[0]] * 21] * 21)
  
#     Find all states with a negative # of cars (these are invalid ones)
      msk = np.prod(next_state >= 0, axis=3) # (21, 21, 2628)
    
#     Zero out all the invalid ones so the reward is zero
      next_state *= np.stack((msk, msk), axis=3)
      

#   Clip the values to between 0 and 20
    next_state = np.clip(next_state, 0, MAX_CARS) 

#   Get the number rented out (Can't rent out more than in stock)
    sold = np.minimum(next_state, p_table[:,:,:,3:5])
    next_state -= sold 
    
    reward = None
#   Calculate the immediate rewards
    if action is None:
      reward = np.sum(sold, axis=3) * PROFIT - np.repeat(np.expand_dims(np.absolute(policy) * COST_MOVE, axis=2), p_row.shape[0], axis=2) # (21, 21, 2628)
#     First car from loc_a to loc_b is free      
      reward += np.repeat(np.expand_dims(2 * (policy < 0), axis=2), p_row.shape[0], axis=2)

    else:
      reward = np.sum(sold, axis=3) * PROFIT - abs(action) * COST_MOVE # (21, 21, 2628)
#     First car from loc_a to loc_b is free      
      if action < 0:
        reward += 2

#   Cost of the extra parking lot if more than 10 cars at one location
    reward -= 4 * np.sum(next_state > 10, axis=3)
    
#   Calculate the discounted long term rewards
    v = p_table[:,:,:,0] * (reward + DISCOUNT * v[next_state[:,:,:,0].astype(np.intp), next_state[:,:,:,1].astype(np.intp)]) # (21, 21, 2628)
    v = np.sum(v, axis=2) # (21, 21) 
    
    return v

def policy_eval(V_s):
  delta = DELTA_LIM + 1
  
  while delta > DELTA_LIM:
    v = V_s.copy()
    
#   Get the new evaluated state values
    v = P(v)
    
#   Get the max difference between the old state values and the new one
    delta = np.amax(np.absolute(v - V_s))
    
    V_s = v
    print("DELTA", np.round(delta, 6))
    
  return V_s

def policy_improve(V_s, policy):
  stable = True
  
  old_policy = policy.copy()
  v_actions = []
  
  for a in ACTIONS:
    v = V_s.copy()
    
#   Get the new evaluated state values if we use the current action
    v = P(v, a)
    
#   Add it to the list of states rewards from each of the actions
    v_actions.append(np.expand_dims(v, axis = 2))
  
# Stack up all the state rewards from each action into an matrix
  v_actions = np.concatenate(v_actions, axis = 2) # (21, 21, 11)
  
# Get the actions with the best rewards
  policy = np.argmax(v_actions, axis = 2) - 5
  
# If any of the actions changes then the policy is not yet stable
  stable = False if np.sum(policy.astype(int) - old_policy.astype(int)) else True
  
  print("LOSS", np.round(np.sum(np.absolute(policy - old_policy)), 6))
  
  return stable, policy


# In[10]:


V_s = np.rint(np.zeros([21,21]))
policy = np.rint(np.zeros([21,21]))

stability = False
iteration = 1
while not stability:
  print("Iteration #" + str(iteration) + " State Values")
  print(V_s.astype(int))
  
  print("Iteration #" + str(iteration) + " Policy")
  print(policy.astype(int))
  
  V_s = policy_eval(V_s)
  stability, policy = policy_improve(V_s, policy)
  
  iteration += 1

print("Done!")
print("Iteration #" + str(iteration) + " State Values")
print(V_s.astype(int))

print("Iteration #" + str(iteration) + " Policy")
print(policy.astype(int))

