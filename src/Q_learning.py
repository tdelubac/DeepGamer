import numpy as np
import random
import collections

def observeGame(model, env, epsilon=1, render=False):
     # Initialise environment and render
    state = env.reset()
    state = state.reshape(-1)
    score = 0
    memory = collections.deque()
    done = False
    while not done:
        if render:
            env.render()
        # Pick action allowing random action 
        # to allow the model to learn new moves
        random_number = random.uniform(0,1)
        if random_number < epsilon:
            action = random.choice(range(env.action_space.n))
        else:
            action = np.argmax(model.predict( np.expand_dims(state, axis=0) ))
        # Carry action and save <s,a,r,s'>
        new_state, reward, done, info = env.step(action)
        new_state = new_state.reshape(-1)
        memory.append([state,action,reward,new_state,done])
        score += reward
        state = new_state

    return memory, score


def learnFromMemory(model, env, memory, gamma=0.9, batch_size=5000): 
    # Take a random sample of the memory 
    # to train the model
    if len(memory) < batch_size:
        recollection = memory
    else:
        recollection = random.sample(memory,batch_size)
    batch = []
    targets = []
    for event in recollection:
        state     = event[0]
        action    = event[1]
        reward    = event[2]
        new_state = event[3]
        done      = event[4]
        # Define the targets for the action 
        # using the model prediction for all actions
        # but the one that was choosen which is updated
        # with Bellman's equation (if game is note done).
        target = model.predict( np.expand_dims(state, axis=0) )[0]
        if done:
            target[action] = reward
        else:
            target[action] = reward + gamma * np.max(model.predict( np.expand_dims(new_state, axis=0) )[0])
        batch.append(state)
        targets.append(target)
        
    trainModel(X=batch,y=targets,model=model)

    return model

def trainModel(X, y, model=False):
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')



        