import collections
import gym
import Q_learning
import random
import numpy as np
import time
import copy
from frame_processing import generateSequence
from NN_model import MLP, Mnih15
from keras.models import load_model

# Variables
game = 'Breakout-v0'
model_name = 'Mnih_'+game+'.h5'
process_frame = True

load = True
load_name = '../run/Breakout/Mnih_Breakout-v0.h5_step100000'
load_step = 100000

number_of_action_repeat = 4                  # Number of time the action choosen by the Q_network will be repeated
number_of_Q_learning_steps = 10000           # Number of times to repeate the Q-learning loop. Final number of frames is = number_of_Q_learning_steps * number_of_action_repeat 
number_of_steps_to_update_network = 10000     # Number of Q-learning steps between each update of the Q_network with weights of the target_Q_network
initial_memory_size = 10000                   # Number of frames to save in memory before beginning Q-learning
max_memory_size = 100000                       # Maximum number of frames in memory (after we start to pop out old memory)
minibatch_size = 32                          # Size of minibatches on which the Q_network is trained
epsilon_ini = 1                              # Initial value of epsilon (to decide whether to pick an action at random or not)
epsilon_min = 0.1                            # Minimal value of epsilon
number_of_epsilon_steps = 100000              # Number of Q-learning steps to go from epsilon_ini to epsilon_min
gamma = 0.99                                 # Bellman's equation discount parameter
keras_verbose = False                       # Set Keras to verbose mode

play_game = False                            # Play a full game at each update of the Q_network
render_game = False                          # Render the game
full_render = False                           # Render every single step in Q_learning
save_steps = 1000                            # Number of steps between 2 saves of the model
first_action = 1                             # First action done when reseting the game

def epsilon(x):
    '''
    Attenuation function to reduce the fraction of random actions
    '''
    if x < number_of_epsilon_steps:
        return epsilon_ini + (epsilon_min - epsilon_ini) / number_of_epsilon_steps * x 
    else:
        return epsilon_min

def play(game,model,render):
    env = gym.make(game)
    env.reset()
    # Create first sequence
    sequence, reward, done = generateSequence(env, first_action, number_of_action_repeat, render, process_frame)
    while not done:
        action = np.argmax(model.predict( np.expand_dims(sequence,axis=0) )[0])
        # Generate new sequence
        new_sequence, reward, done = generateSequence(env, action, number_of_action_repeat, render, process_frame)
        sequence = new_sequence


def main():
    time_at_begining = time.time()
    # Initialize gym environment
    env = gym.make(game)
    env.reset()
    # Create first sequence
    sequence, reward, done = generateSequence(env, first_action, number_of_action_repeat, render=False, process_frame=process_frame)
    # Initialize Q_networks
    input_shape = sequence.shape
    Q_network = Mnih15(input_shape,env.action_space.n)
    target_Q_network = Mnih15(input_shape,env.action_space.n)
    #Q_network = MLP(input_shape,env.action_space.n)
    #target_Q_network = MLP(input_shape,env.action_space.n)
    if load:
        Q_network = load_model(load_name)  
        target_Q_network = load_model(load_name)
    # Initialize memmory
    memory = collections.deque()

    print("Build up initial memory:")
    while len(memory) < initial_memory_size:
        if len(memory)%100 == 0:
            print(len(memory),"/",initial_memory_size)

        if done:
            env.reset()
            sequence, reward, done = generateSequence(env, first_action, number_of_action_repeat, render=False, process_frame=process_frame)

        # Pick a random action
        action = env.action_space.sample()
        # Generate new sequence
        new_sequence, reward, done = generateSequence(env, action, number_of_action_repeat, render=False, process_frame=process_frame)
        # Save sequence, action, reward, new_sequence
        memory.append([sequence, action, reward, new_sequence, done])
        sequence = new_sequence

    score = 0
    scores = []
    number_of_games = 0
    current_game_done = done
    print("Begin Q-learning")
    for _ in range(load_step,number_of_Q_learning_steps+load_step+1): # +1 to save the model on the last step
        if _%100 == 0:
            print("Q-learning step",_,"/",number_of_Q_learning_steps+load_step)
        if current_game_done:
            number_of_games+=1
            scores.append(score)
            # print("Score:",score)
            score = 0
            env.reset()
            sequence, reward, done = generateSequence(env, first_action, number_of_action_repeat, full_render, process_frame)

        random_number = random.uniform(0,1)
        if random_number < epsilon(_):
            action = random.choice(range(env.action_space.n))
        else:
            action = np.argmax(Q_network.predict( np.expand_dims(sequence,axis=0) )[0])
        # Generate new sequence
        new_sequence, reward, done = generateSequence(env, action, number_of_action_repeat, full_render, process_frame)
        score+= reward
        current_game_done = done
        # Save sequence, action, reward, new_sequence
        memory.append([sequence, action, reward, new_sequence, done])
        sequence = new_sequence
        # Remove old memory as we go
        while len(memory) > max_memory_size:
            memory.popleft()
        # Sample random minibatch
        recollection = random.sample(memory,minibatch_size)
        minibatch = []
        targets = []
        for event in recollection:
            r_sequence     = event[0]
            r_action       = event[1]
            r_reward       = event[2]
            r_new_sequence = event[3]
            r_done         = event[4]
            # Define the targets for the action 
            # using the model prediction for all actions
            # but the one that was choosen which is updated
            # with Bellman's equation (if game is note done).
            target = target_Q_network.predict( np.expand_dims(r_sequence,axis=0) )[0]
            if r_done:
                target[r_action] = r_reward
            else:
                target[r_action] = r_reward + gamma * np.max(target_Q_network.predict( np.expand_dims(r_new_sequence,axis=0) )[0])        
            # Append lists and transform to arrays as arrays are required for Keras
            minibatch.append(r_sequence)
            targets.append(target)
        minibatch = np.asarray(minibatch)
        targets = np.asarray(targets)
        # Training the Q_network
        Q_network.fit(minibatch,targets,batch_size=minibatch_size,epochs=1,verbose=keras_verbose)

        if _%number_of_steps_to_update_network == 0:
            if number_of_games>0:
                print("---------------")
                print("Game statistics")
                print("---------------")
                print("# games:",number_of_games,"| Mean score:",np.mean(scores),"| Median score:",np.median(scores),"| Min score:",np.min(scores),"| Max score:",np.max(scores),"| Randomness:",epsilon(_))
                print("---------------")
            number_of_games = 0
            scores = []
            target_Q_network.set_weights(Q_network.get_weights())
            if play_game:
                play(game,Q_network,render_game)

        if _%save_steps == 0:
            Q_network.save(model_name+'_step'+str(_))
    print("Execution time:",time.time() - time_at_begining)       

if __name__ == "__main__":
    main()




        