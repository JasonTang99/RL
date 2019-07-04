import numpy as np

# Set up sampling environment
q = np.random.normal(4.0, size=10)
std = 0.1

n = 5000

# Set up inital values, set q_a to be way higher than expected values for optimistic method
q_a = np.array([5.0] * len(q))
n_a = np.array([0] * len(q))

# alpha sets a constant step size
# leaving it negative takes an average of historical data
# For convergence, we require:
#   1. sum of alphas to be infinite
#   2. sum of squared alphas to be finite
def alpha(action):
  # return 1/n_a[action]
  return 0.1

# Greedy Epsilon picks the greedy action usually
# Picks a random action epsilon percent of the time
def greedy_epsilon(ep):
  for _ in range(n):
    action = None
    if np.random.random() < 1 - ep:
      action = np.argmax(q_a)
    else:
      action = np.random.randint(10)
    reward = np.random.normal(q[action], std)
    n_a[action] += 1
    q_a[action] += alpha(action) * (reward - q_a[action])

# Upper Confidence Bound would be to choose actions based off of argmax_a [q_a(a) + c sqrt(ln(t)/n_a(a))]
# where t is the step number and c controls the degree of exploration
def ucb(c):
  t = 0
  for _ in range(n):
    t += 1
    action = np.argmax([q_a[i] + c * np.sqrt(np.log(t)/np.max([n_a[i], 1])) for i in range(len(q_a))])
    reward = np.random.normal(q[action], std)
    n_a[action] += 1
    q_a[action] += alpha(action) * (reward - q_a[action])


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# SGD follows a similar stepping algorithm like those in Neural Networks
def sgd(a):
  reward_avg = 0
  t = 0
  for _ in range(n):
    random_num = np.random.random()
    sm = softmax(q_a)
    action = None
    for i in range(len(sm)):
      random_num -= sm[i]
      if random_num <= 0:
        action = i
        break
    
    t += 1
    n_a[action] += 1
    reward = np.random.normal(q[action], std)
    reward_avg += 1/t * (reward - reward_avg)

    q_a[action] += a * (reward - reward_avg) * (1 - sm[action])
    for i in range(len(sm)):
      if i != action:
        q_a[i] -= a * (reward - reward_avg) * (sm[i])

# greedy_epsilon(ep = 0.1)
# ucb(c = 2)
sgd(0.1)

print(q)
print(q_a)
print(n_a)
print("Mean Squared Error:", np.sum(np.square(q - q_a)))
print("Best action was taken", n_a[np.argmax(q)] * 100 / n, "percent of the time.")