import numpy as np
import skimage.measure

def processFrame(frame):
    '''
    Process the frame to be feed to the Q network
    '''

    # Remove edges that are uninformative
    frame = frame[32:194,8:152]

    # Extract luminance Y from RGB frame as color is uninformative
    conversion = [0.212, 0.715, 0.072]
    Y = np.dot(frame, conversion)

    # Resize to speedup NN training
    Y_reduced = skimage.measure.block_reduce(Y, (2,2), np.max)

    return Y_reduced


def generateSequence(env, action, number_of_frames=4, render=False, process_frame=False):
    '''
    Generate a sequence of states performing each times the same action
    '''
    sequence = []
    total_reward = 0
    for _ in range(number_of_frames):
        state, reward, done, info = env.step(action)
        if render:
            env.render()
        if process_frame:
            state = processFrame(state)
        total_reward+= reward
        sequence.append(state)
        if done:
            # Copy the last element of the sequence so that the total lenght is always number_of_frames
            while len(sequence) < number_of_frames:
                sequence.append(sequence[-1])
            break

    assert len(sequence) == number_of_frames
    
    sequence = np.asarray(sequence)
    if process_frame:
        sequence = sequence.reshape(sequence.shape[1],sequence.shape[2],sequence.shape[0])

    return sequence, total_reward, done
