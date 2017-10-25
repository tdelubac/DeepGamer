import collections
import gym
import Q_learning
import random
import numpy as np
from NN_model import MLP

game = 'CartPole-v1'
training_steps = 10
training_games = 500 # How many games to train on

def main():
    env = gym.make(game)
    
    random.seed(42)

    model = MLP(env.observation_space.shape[0],env.action_space.n)
    memory = collections.deque()

    for _ in range(training_steps):
        scores = []
        for __ in range(training_games):
            game_memory, score = Q_learning.observeGame(model, env, epsilon=1./np.sqrt(_+1), render=False)#(__%100==0))
            if score > 50:
                memory.extend(game_memory)
            scores.append(score)
        print("mean score:", np.mean(score))
        model = Q_learning.learnFromMemory(model, env, memory)

    model.save("../models/MLP_CartPole-v0.tflearn")

    # Play!
    state = env.reset()
    done = False
    score = 0
    while not done: 
        env.render()
        action = np.argmax(model.predict( np.expand_dims(state, axis=0) ))
        new_state, reward, done, info = env.step(action)
        state = new_state
        score += reward
    print("play score:",score)

if __name__ == "__main__":
    main()

        