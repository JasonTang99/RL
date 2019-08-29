import numpy as np

# Set up sampling environment
state_values = np.random.normal(size=10)
states = range(10)

v_s = np.array([0.0] * len(states))
pi_s = np.array([0] * len(states))

def policy_evaluation():
  delta = 5
  while delta > 0.1:
    delta = 0
    for s in states:
      v = v_s[s]
      v_s[s] = 0
      for a in 
      


