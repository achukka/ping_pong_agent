import tensorflow as tf
import cv2

import ping_pong
import numpy as np
import random
from collections import deque # Store the history, queue data structure.
# Using queue data structure -  deque to append and pop, memory 

# Define hyper parameters for the rl algorithm
actions = 3 # stay, up, down

# learning rate
gamma = 0.99

# Used for gradient updates
init_epsilon = 1.0
final_epsilon = 0.05


# Frames to explore, observe
explore = 500000
observe = 50000

# History size
replay_memory = 500000

batch_size = 100


# Create the tensor flow graph
def create_graph():
    # Convolutional Neural Network with 3 convolutional layers, 1 FC and final FC Layer
    
    # First Convolutional Layer , bias vector
    w_conv_1 = tf.Variable(tf.zeros([8, 8, 4, 32])) # Empty weight tensor
    b_conv_1 = tf.Variable(tf.zeros([32])) # Empty bias tensor
    
    # Second Convolutional Layer , bias vector
    w_conv_2 = tf.Variable(tf.zeros([4, 4, 32, 64])) # Empty weight tensor
    b_conv_2 = tf.Variable(tf.zeros([64])) # Empty bias tensor
    
    # Third Convolutional Layer , bias vector
    w_conv_3 = tf.Variable(tf.zeros([3, 3, 64, 64])) # Empty weight tensor
    b_conv_3 = tf.Variable(tf.zeros([64])) # Empty bias tensor
    
    # Fourth Fully Connected Layer, bias vector
    w_fc_4 = tf.Variable(tf.zeros([3136, 784])) # Empty weight tensor
    b_fc_4 = tf.Variable(tf.zeros([784])) # Empty bias tensor
    
    # Last Layer
    w_fc_5 = tf.Variable(tf.zeros([784, actions])) # Empty weight tensor
    b_fc_5 = tf.Variable(tf.zeros([actions])) # Empty bias tensor
    
    # Input for pixel data
    s = tf.placeholder("float", [None, 84, 84, 4])
    
    # Compute Recitified Linear Unit (ReLU) activation function on 2d convolutional
    # Given 4D inputs and filter tensors
    
    conv_1 = tf.nn.relu(tf.nn.conv2d(s, w_conv_1, strides=[1, 4, 4, 1], padding="VALID") + b_conv_1)
    
    conv_2 = tf.nn.relu(tf.nn.conv2d(conv_1, w_conv_2, strides=[1, 2, 2, 1], padding="VALID") + b_conv_2)
    
    conv_3 = tf.nn.relu(tf.nn.conv2d(conv_2, w_conv_3, strides=[1, 1, 1, 1], padding="VALID") + b_conv_3)
    
    # Reshape the conv layer
    conv_3_flat = tf.reshape(conv_3, [-1, 3136])
    
    fc_4 = tf.nn.relu(tf.matmul(conv_3_flat, w_fc_4) + b_fc_4)
    
    fc_5 = tf.matmul(fc_4 , w_fc_5) + b_fc_5
    
    print 'Graph Created'
    
    return s, fc_5


# Deep Q Network. Feeding in pixel data to graph session
def train_graph(inpt, output, sess):
    
    # Calculate the argmax, 
    # multiply the predicted output with a vector with one value '1'
    # and rest as '0'
    argmax = tf.placeholder("float", [None, actions])
    
    # Ground truth
    ground_truth = tf.placeholder("float",[None])
    
    # Get the action, computes the sum of elements across the dimensions of a tensor
    action = tf.reduce_sum(tf.mul(output, argmax), reduction_indices = 1)
    
    # Cost function, squared error
    cost = tf.reduce_mean(tf.square(action - ground_truth))
    
    # Optimization function, reduces the cost function
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    
    # Initialize the game
    game = ping_pong.PongGame()
    
    # Create the deque to store the history to optimize policies
    dq = deque()
    
    # Initial frame
    frame = game.get_current_frame()
    
    # RGB to Gray Scale for preprocessing
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    
    # Binary Colors, white or black
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    
    # Stack Frames - this is the input tensor
    input_t = np.stack((frame, frame, frame, frame), axis=2)
    
    # Save the train instance
    saver = tf.train.Saver()
    
    # initialize the variables
    sess.run(tf.initialize_all_variables())
    
    # Initial time
    time = 0
    epsilon = init_epsilon
    
    # Training time
    while(True):
        # Output tensor
        output_t = output.eval(feed_dict = {inpt: [input_t]})[0]
        
        # Argmax function
        argmax_t = np.zeros([actions])
        ''' Select a random action - this is because of Q - Learning. You either choose
             1. the best action with '(1-epsilon)' probability (exploit) 
             2. a random action with 'epsilon' probability (explore) '''
        # Select a random index
        if (random.random() <= epsilon):
            maxIndex = random.randrange(actions)
        else:
            maxIndex = np.argmax(output_t)
            
        argmax_t[maxIndex] = 1
        
        if epsilon > final_epsilon: # Decrease the epislon by exploration constant
            epsilon -= (init_epsilon - final_epsilon) / explore
        
        # Reward if the score is positive
        reward_t, frame = game.get_next_frame(argmax_t)
        
        # Get pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))
        
        # new input tensor
        new_input_t = np.append(frame, input_t[:, :, 0:3], axis = 2)
        
        # Add our input tensor, argmax_tensor, reward and updated input tensor
        # to the history
        dq.append((input_t, argmax_t, reward_t, new_input_t))
        
        # Clear the old ones, if the memory is full
        if len(dq) > replay_memory:
            dq.popleft()
        
        # Training iteration(time)
        if time > observe: # Time for some observation
            
            # Get value from replay memory
            minibatch = random.sample(dq, batch_size)
            
            input_batch = [batch[0] for batch in minibatch]
            argmax_batch = [batch[1] for batch in minibatch]
            reward_batch = [batch[2] for batch in minibatch]
            new_input_batch = [batch[3] for batch in minibatch]
            
            ground_truth_batch = []
            output_batch = output.eval(feed_dict = {inpt : new_input_batch})
            
            # Now start adding reward to the batch
            for index in range(0, len(minibatch)):
                ground_truth_batch.append(reward_batch[index] + 
                                          gamma * np.max(output_batch[index]))
            
            # Train on the new reward values
            train_step.run(feed_dict = {ground_truth: ground_truth_batch,
                                        argmax : argmax_batch, 
                                        inpt : input_batch})
        # update the tensor for the next frame
        input_t = new_input_t
        time = time + 1
        
        # Print the stats and save the network
        if time % 10000 == 0:
            saver.save(sess, './' +'pong' +'-dqn', global_step = time)
        if time % 100 ==0:
            print "TIMESTEP",time,"EPSILON",epsilon, "ACTION", maxIndex,"Reward", reward_t, "Q_MAX %e" % np.max(output_t)

def main():
    # Create the tensor flow session
    sess = tf.InteractiveSession()
    
    # Input layer and the output layer
    inpt, output = create_graph()
    
    print 'Training the graph Created'
    train_graph(inpt, output, sess)


if __name__ == "__main__":
    print 'Calling Main'
    main()
