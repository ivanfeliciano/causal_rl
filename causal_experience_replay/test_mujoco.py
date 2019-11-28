import gym
env = gym.make('FetchPickAndPlace-v1')
env.reset()
print("Actions = {}".format(env.action_space))
flag = True
for _ in range(1000):
  env.render()
  action = env.action_space.sample()
  action[3] = -0.8 if flag else 0.9
  flag = not flag
  print(action)
  env.step(action) # take a random action