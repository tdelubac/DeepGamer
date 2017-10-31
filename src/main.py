import collections
import gym
import Q_learning
import random
import numpy as np
from NN_model import MLP

# Variables
game = 'CartPole-v0'
training_steps = 10
training_games = 1000 # How many games to train on
score_to_keep = 50
display = 100

# Increase number of maximum_steps
gym.envs.registry.spec(game).max_episode_steps = 1000

def attenuation(x):
    '''
    Attenuation function to reduce the fraction of random actions
    '''
    return 1./np.sqrt(x+1)

def main():
    env = gym.make(game)
    
    random.seed(42)
    state = env.reset()
    input_shape = len(state.reshape(-1))
    model = MLP(input_shape,env.action_space.n)
    memory = collections.deque()

    for _ in range(training_steps):
        scores = []
        print('######################')
        print('Training step',_+1,'/',training_steps)
        print('######################')
        for __ in range(training_games):
            if __%display == 0:
                print("Game",__+1,"/",training_games)
                game_memory, score = Q_learning.observeGame(model, env, epsilon=attenuation(_), render=True)
                print("Score",score)
            else:
                game_memory, score = Q_learning.observeGame(model, env, epsilon=attenuation(_), render=False)

            if score > score_to_keep:
                memory.extend(game_memory)
            scores.append(score)
        print("Score min:", np.min(scores), "| Score max:", np.max(scores), "| Mean:", np.mean(scores), "| Std:", np.std(scores))
        model = Q_learning.learnFromMemory(model, env, memory)
    print("Shape memory",np.shape(memory))
    model.save("models/MLP_CartPole-v0.tflearn")

    # Play!
    state = env.reset()
    state = state.reshape(-1)
    done = False
    score = 0
    while not done: 
        env.render()
        action = np.argmax(model.predict( np.expand_dims(state, axis=0) ))
        new_state, reward, done, info = env.step(action)
        new_state = new_state.reshape(-1)
        state = new_state
        score += reward
    print("play score:",score)

if __name__ == "__main__":
    main()

        