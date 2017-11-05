import gym
import random
import Q_learning
import numpy as np
from NN_model import MLP, Mnih15
from frame_processing import generateSequence
from keras.models import load_model

game = 'Breakout-v0'
model_name = 'Mnih_'+game+'.h5_step61000'

first_action = 1
number_of_action_repeat = 4
render = True
process_frame = True
epsilon = 0.1
def main():
    env = gym.make(game)
    env.reset()
    # Create first sequence
    sequence, reward, done = generateSequence(env, first_action, number_of_action_repeat, render=render, process_frame=process_frame)

    Q_network = load_model(model_name)  
  
    done = False
    score = 0
    while not done: 
        random_number = random.uniform(0,1)
        if random_number < epsilon:
            action = random.choice(range(env.action_space.n))
        else:
            action = np.argmax(Q_network.predict( np.expand_dims(sequence,axis=0) )[0])

        sequence, reward, done = generateSequence(env, action, number_of_action_repeat, render=render, process_frame=process_frame)
        score += reward
    print("play score:",score)
    return

if __name__ == "__main__":
    main()

