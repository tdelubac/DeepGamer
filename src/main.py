import collections
import gym
import Q_learning
import random
import numpy as np
from NN_model import MLP

game = 'CartPole-v0'
training_steps = 1 
training_games = 200 # How many games to train on

def main():
    env = gym.make(game)
    
    random.seed(42)
    # while True:
    #     env.reset()
    #     done = False
    #     while not done:
    #         env.render()
    #         state, reward, done, info = env.step(np.random.choice(range(env.action_space.n)))

    model = MLP(env.observation_space.shape[0],env.action_space.n)
    memory = collections.deque()

    # for _ in range(training_games):
    #     model, memory = Q_learning.learningStep(model, env, memory, training_steps, render=True)

    for _ in range(training_steps):
        for __ in range(training_games):
            memory = Q_learning.observeGame(model, env, memory, render=False)
        model = Q_learning.learnFromMemory(model, env, memory)

    model.save("../models/MLP.tflearn")

    # Play!
    state = env.reset()
    done = False
    score = 0
    while not done: 
        env.render()
        action = np.argmax(model.predict( np.expand_dims(state, axis=0) ))
        print(model.predict( np.expand_dims(state, axis=0) ))
        new_state, reward, done, info = env.step(action)
        state = new_state
        score += reward
    print("play score:",score)

if __name__ == "__main__":
    main()

        