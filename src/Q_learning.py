import numpy as np
import random
import collections

def learningStep(model, env, memory=False, training_steps=500, gamma=0.9, epsilon=0.7, batch_size=500, render=False): 
    # Initialise environment and render
    state = env.reset()
    if memory == False:
        memory = collections.deque()


    score = 0
    for _ in range(training_steps):
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
        memory.append([state,action,reward,new_state,done])
        score += reward
        # Take a random sample of the memory 
        # to train the model
        if len(memory) < batch_size:
            recollection = memory
        else:
            recollection = random.sample(memory,batch_size)
        batch = []
        targets = []
        for event in recollection:
            r_state     = event[0]
            r_action    = event[1]
            r_reward    = event[2]
            r_new_state = event[3]
            r_done      = event[4]
            # Define the targets for the action 
            # using the model prediction for all actions
            # but the one that was choosen which is updated
            # with Bellman's equation (if game is note done).
            target = model.predict( np.expand_dims(r_state, axis=0) )[0]
            if r_done:
                target[r_action] = r_reward
            else:
                target[r_action] = r_reward + gamma * np.max(model.predict( np.expand_dims(r_new_state, axis=0) )[0])
            batch.append(r_state)
            targets.append(target)
       
        if done:
            break
        else:
            state = new_state
    print("training step score:",score)
    
    trainModel(X=batch,y=targets,model=model)
            
    return model, memory

def observeGame(model, env, memory=False, observe_steps=50, epsilon=0.2, render=False):
     # Initialise environment and render
    state = env.reset()
    score = 0
    if memory == False:
        memory = collections.deque()

    for _ in range(observe_steps):
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
        memory.append([state,action,reward,new_state,done])
        score += reward
        if done:
            break
        state = new_state

    print("observed game score:",score)
    return memory


def learnFromMemory(model, env, memory, gamma=0.9, batch_size=500): 
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



        