import gym

env = gym.make("FrozenLake-v0")
env.reset()
env.render()
done = False

while not done:
    # action = env.action_space.sample()
    action = int(input())
    new_state, reward, done, info = env.step(action)
    print()
    env.render()
    print("Reward: {:.2f}".format(reward))
    print(info) 
