import scipy
import tensorflow as tf
from tf_utils import random_mini_batches
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import numpy as np



def loadData(input_data_path, output_data_path):
    """
    load Training Data, include real and imag part
    :return: dict contains all data
    """
    train_input = scipy.fromfile(open(input_data_path), dtype = scipy.complex64)
    train_output = scipy.fromfile(open(output_data_path), dtype = scipy.complex64)

    data = {
        "X" : train_input,
        "Y" : train_output
    }
    return data

def complex_divide(X_train, Y_train, X_test, Y_test):
    """

    splite complex number to real and imag part
    for example:
    Before : X_train = np.array([1+2j, 2+3j, 3+4j])
    After  : X_train = np.array([[ 1.,  2.],
                            [ 2.,  3.],
                            [ 3.,  4.]])
    :param X_train: X_train data  shape(1,m) 1 is complex number
    :param Y_train: Y_train data
    :param X_test:  X_test data
    :param Y_test:  Y_test data
    :return: shape(2,m) 2 is real and imag part
    """

    def combine(real_part, imag_part):
        real_part = real_part.reshape(real_part.shape[0],1)
        imag_part = imag_part.reshape(imag_part.shape[0],1)
        combined = np.concatenate((real_part,imag_part),axis=1)
        return combined

    real_x_train, imag_x_train = X_train.real, X_train.imag
    real_y_train, imag_y_train = Y_train.real, Y_train.imag
    real_x_test, imag_x_test = X_test.real, X_test.imag
    real_y_test, imag_y_test = Y_test.real, Y_test.imag

    X_train = combine(real_x_train,imag_x_train).T
    Y_train = combine(real_y_train, imag_y_train).T
    X_test = combine(real_x_test, imag_x_test).T
    Y_test = combine(real_y_test, imag_y_test).T


    return X_train, Y_train, X_test, Y_test




def create_placeholders(n_x, n_y):
    """

    :param n_x: scalar, size of an complex number , real part + imag part = 2
    :param n_y: scalar, number of predict number, complex number
    :return: placeeholder of n_x and n_y
    """
    X = tf.placeholder(tf.float32, shape=(n_x,None))
    Y = tf.placeholder(tf.float32, shape=(n_y,None))

    return X,Y

def initial_Parameter():
    """

    :return:
    """
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1",[5,2],initializer= tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1",[5,1],initializer= tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 5], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [2, 12], initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [2, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

def forward_propogation(X,parameters):
    """
    Implement forward propogation for the model: LINEAR -> ReLu -> LINEAR -> TANH -> LINEAR
    :param X: input data placeholder
    :param parameters: initialed parameters
    :return: Z3, the output of the last LINEAR unit
    """
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]

    Z1 = tf.matmul(W1,X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2,A1) + b2
    A2 = tf.nn.tanh(Z2)
    Z3 = tf.matmul(W3,A2) + b3


    return Z3

def compute_cost(Z3, Y):
    """

    :param Z3: output of last LINEAR unit (2,1)
    :param Y: true value of output complex number
    :return: cost: Tensor of the cost function
    """
    cost = tf.reduce_mean((Z3 + Y)**2)
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 10, minibatch_size = 10000, print_cost = True):
    """

    :param X_train: training set, input size = 2, real and imag parts, number of training examples = unknown
    :param Y_train: test set, of shape( output size = 2, number of training examples = same as X_train)
    :param X_test: training set, of shape( input size = 2, number of test examples = 120)
    :param Y_test: test set, of shape( output size = 2, number of test examples = 120)
    :param learning_rate: learning rate
    :param num_epochs: epoch numbers
    :param minibatch_size: size of minibatch
    :param print_cost: True to print the cost every 100 expochs
    :return:
    :parameters -- parameters learnt by the model. They can then be used to predict
    """

    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x,m) = X_train.shape
    n_y = Y_train.shape[0]
    costs =[]

    X, Y = create_placeholders(n_x=n_x, n_y=n_y)
    parameters = initial_Parameter()
    Z3 = forward_propogation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            seed = seed +1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch


                _, minibatch_cost = sess.run([optimizer,cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 10 == 0:

                print ("Cost after epoch %i: %f"% (epoch, epoch_cost))
            if print_cost == True and epoch% 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("learning rate = " + str(learning_rate))
        plt.show()


        parameters = sess.run(parameters)
        print ("Parameters have been trained")

        # correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        #
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        accuracy = Y+Z3

        print ("Train Accuracy: ", accuracy.eval({X:X_train, Y:Y_train}))
        print ("Test Accuracy: ", accuracy.eval({X:X_test, Y:Y_test}))

        return parameters

data = loadData("X_train","Y_train")

Y_train = data["Y"]
length = len(Y_train)
X_train = data["X"][0:length]



test_data = loadData("X_test","Y_test")
Y_test = data["Y"]
length = len(Y_test)
X_test = data["X"][0:length]


X_train, Y_train, X_test, Y_test = complex_divide(X_train, Y_train, X_test, Y_test)




parameters = model(X_train, Y_train, X_test, Y_test)




# tf.reset_default_graph()
# with tf.Session() as sess:
#     X, Y =  create_placeholders(n_x=2, n_y=2)
#     parameters = initial_Parameter()
#     Z3 = forward_propogation(X,parameters)
#     cost = compute_cost(Z3,Y)
#     print ("cost = " +str(cost))

# f1 = scipy.fromfile(open("dbpsk_out"),dtype = scipy.byte)
# for i in range(len(f1)):
#
#     if (i%2==0):
#
#         print str(f1[i])+str(f1[i+1])
#         i=i+1
#
# with open("dbpsk_out.txt") as f:
#     print f.readline().decode("utf-8")
