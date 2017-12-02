################################# Setup ####################################
# data is hosted on Yann LeCun's website.
# This will download and read in the data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#import tensorflow before it can be used
import tensorflow as tf

# x isn't a specific value. 
# It's a placeholder, a value that we'll input when we ask TensorFlow to run a computation.
# Here None means that a dimension can be of any length.
# I think it's basically float x[784]; unanitialized
x = tf.placeholder(tf.float32, [None, 784])

# In this case, we initialize both W and b as tensors full of zeros. 
# Since we are going to learn W and b, it doesn't matter very much what they initially are.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

################################# define model ####################################
# define the softmax model
# (x * W) + b sent to tf.nn.softmax func
y = tf.nn.softmax(tf.matmul(x, W) + b)

# placeholder for inputting the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# implement the cross-entropy function -- For determing loss or cost
# sum of ( y log(y_) )
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# minimize cross_entropy using the gradient descent algorithm
# learning rate of 0.5
# Gradient descent is a simple procedure, where TensorFlow simply shifts -->
# each variable a little bit in the direction that reduces the cost
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# launch the model in an InteractiveSession
sess = tf.InteractiveSession()

# create an operation to initialize the variables we created
tf.global_variables_initializer().run()


######################################### TRAIN #####################################
# run the training step 1000 times
# Each step of the loop, we get a "batch" of one hundred random data points from our -->
# training set. We run train_step feeding in the batches data to replace the placeholders.
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


################################### Evaluating the model #####################################
# figure out where we predicted the correct label
# tf.argmax function gives the index of the highest entry in a tensor along some axis
# tf.argmax(y,1) is the label our model thinks is most likely for each input
# tf.argmax(y_,1) is the correct label.
# tf.equal to check if our prediction matches the truth
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# That gives us a list of booleans.^^^
# To determine what fraction are correct, we cast to floating point numbers and then take the mean.
# For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#display the accuracy on the test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))