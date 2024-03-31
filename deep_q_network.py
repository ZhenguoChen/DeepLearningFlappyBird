#!/usr/bin/env python
import os
import sys

import cv2
import tensorflow as tf

sys.path.append("game/")
import random
from collections import deque

import numpy as np
import wrapped_flappy_bird as game

MODE = os.environ.get("MODE", "EVAL")

GAME = "bird"  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
REPLAY_MEMORY = 50000  # number of previous transitions to remember
MAX_EPOCHS = 1000000
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1

if MODE == "TRAIN":
    OBSERVE = 10000
    EXPLORE = 3000000
    FINAL_EPSILON = 0.0001
    INITIAL_EPSILON = 0.1
else:
    OBSERVE = 100000.0  # timesteps to observe before training
    EXPLORE = 2000000.0  # frames over which to anneal epsilon
    FINAL_EPSILON = 0.0001  # final value of epsilon
    INITIAL_EPSILON = 0.0001  # starting value of epsilon

# tf.compat.v1.disable_eager_execution()
variables = list()


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.01)
    weights = tf.Variable(initial)
    variables.append(weights)
    return weights


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    bias = tf.Variable(initial)
    variables.append(bias)
    return bias


# TODO check how saver save and load weights
# there must be a logic to tell which variable has which checkpoint weight
# network weights
W_conv1 = weight_variable([8, 8, 4, 32])
b_conv1 = bias_variable([32])

W_conv2 = weight_variable([4, 4, 32, 64])
b_conv2 = bias_variable([64])

W_conv3 = weight_variable([3, 3, 64, 64])
b_conv3 = bias_variable([64])

W_fc1 = weight_variable([1600, 512])
b_fc1 = bias_variable([512])

W_fc2 = weight_variable([512, ACTIONS])
b_fc2 = bias_variable([ACTIONS])


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, filters=W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool2d(
        input=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
    )


def model(x):
    # input layer
    # s = tf.compat.v1.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return readout


def loss(pred, target):
    action = tf.Variable([None, ACTIONS])
    readout_action = tf.reduce_sum(tf.multiply(pred, action), axis=1)
    cost = tf.reduce_mean(tf.square(target - readout_action))
    return cost


def trainNetwork():
    # define the cost function
    # a = tf.compat.v1.placeholder("float", [None, ACTIONS])
    # y = tf.compat.v1.placeholder("float", [None])
    # train_step = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost)

    def train_step(model, inputs, outputs):
        with tf.GradientTape() as tape:
            current_loss = loss(model(inputs), outputs)
        grads = tape.gradient(current_loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

    optimizer = tf.optimizers.Adam(1e-6)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # map tf1 checkpoint to tf2
    checkpoint = tf.train.Checkpoint(
        vars={
            "Variable": variables[0],
            "Variable_1": variables[1],
            "Variable_2": variables[2],
            "Variable_3": variables[3],
            "Variable_4": variables[4],
            "Variable_5": variables[5],
            "Variable_6": variables[6],
            "Variable_7": variables[7],
            "Variable_8": variables[8],
            "Variable_9": variables[9],
        }
    )
    checkpoint.restore("tf2_weights-1")
    # saving and loading networks
    # saver = tf.compat.v1.train.Saver(var_list=variables)
    # checkpoint = tf.train.get_checkpoint_state("saved_networks")
    # if checkpoint and checkpoint.model_checkpoint_path:
    #     saver.restore(sess, checkpoint.model_checkpoint_path)
    #     print("Successfully loaded:", checkpoint.model_checkpoint_path)
    # else:
    #     print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        readout_t = model(tf.convert_to_tensor([s_t], dtype=tf.float32))[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        # s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = model(s_j1_batch)
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # if terminal, only equals reward
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step(model, s_j_batch, y_batch)

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, "saved_networks/" + GAME + "-dqn", global_step=t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print(
            "TIMESTEP",
            t,
            "/ STATE",
            state,
            "/ EPSILON",
            epsilon,
            "/ ACTION",
            action_index,
            "/ REWARD",
            r_t,
            "/ Q_MAX %e" % np.max(readout_t),
        )
        # write info to files
        """
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        """
        if t > MAX_EPOCHS:
            break


def playGame():
    # sess = tf.compat.v1.InteractiveSession(
    #     config=tf.compat.v1.ConfigProto(log_device_placement=True)
    # )
    # s, readout = createNetwork()
    trainNetwork()


def main():
    playGame()


if __name__ == "__main__":
    main()
