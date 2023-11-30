import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
import os


env = gym.make("CartPole-v1", render_mode="human")

ACTIONNUM = env.action_space.n
STATESHAPE = env.observation_space.shape
learning = 0
isDone = False
env.reset()

while not isDone:
    env.render()
    nextState, reward, isDone, _, _ = env.step(env.action_space.sample())
    learning += 1

print(learning)
env.close()