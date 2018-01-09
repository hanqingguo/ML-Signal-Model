import scipy
import tensorflow as tf

def loadData():
    """

    :return:
    """
    pass

def create_placeholders(n_x, n_y):
    """

    :param n_x: scalar, size of an complex number , real part + imag part = 2
    :param n_y: scalar, number of predict number, complex number
    :return: placeeholder of n_x and n_y
    """
    X = tf.placeholder(tf.float32, shape=(n_x,None))
    Y = tf.placeholder(tf.complex64, shape=(n_y,None))

    return X,Y

def initial_Parameter():
    """

    :return:
    """
    pass

def

f1 = scipy.fromfile(open("dbpsk_out"),dtype = scipy.byte)
for i in range(len(f1)):

    if (i%2==0):

        print str(f1[i])+str(f1[i+1])
        i=i+1
#
# with open("dbpsk_out.txt") as f:
#     print f.readline().decode("utf-8")
