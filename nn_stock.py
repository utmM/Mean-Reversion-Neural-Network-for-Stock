# -*- coding: utf-8 -*-
'''
2. Emphasis meaning of inputs and Machine Learning
    
    ★ Before running this code, you need to adjuct the input to "Emphasize" the meaning of the time series.
    "UP_RATIO (Accelaration)" to be adjusted in the 3 steps bellow.
    
    Step 1.
    Adopt the accelaration which gives the maximum absolete during past 5 days.
    
    Step 2.
    On Step 1. accelaratons, adopt the one gives the maximum absolete during 5 days, from 2 days before to 2 days past, including the day.
    
    Step 3.
    Normalize the accelarations by its maximum and minimum during all the term.
    
    The NN bellow uses "the adjusted accelarations" together with 25 differencials as a input.
    
    Please renew the "input.csv" by adjudted UP_RATIO (accelaration)
    A line consists of 25 differencials and "adjusted" accelaration.
    (Named the input as "input.csv".)
    
    (Why need the adjustments?)
    The row accelaration data has large variation or deviation, so the NN can't read the meaning of the time series. Before trainng, emphasize the trend and make it easy to understand for the NN.

'''
from __future__ import print_function
from os import path
from collections import namedtuple
import numpy as np
import pandas as pd
import tensorflow as tf
# For releasing the limitation for calling the recursive function many times
import sys

# <<<<<<<<<<<<<<<<<< Define Constants for NN >>>>>>>>>>>>>>>
Dataset = namedtuple(
                     'Dataset',
                     'training_predictors training_class test_predictors test_class')
Environ = namedtuple('Environ', 'sess model actual_class training_step dataset feature_data')
ERROR_RANGE = 0.0 # 【IMPORTANT! Please adjust this for yourself.】
T = 25 # days of moving average

# <<<<<<<<<<<<<<<<<< Code >>>>>>>>>>>>>>>>
# 【Load the input data】
def load_input_data_df():
    if path.isfile('./input.csv'):
        print('input_data exists. Start loading the file: ' + 'input.csv:')
        return pd.read_csv('./' + 'input.csv', header=1, index_col=0)
    else:
        print('input.csv not found.')

# 【Split input data (The 80% is for training, the rest is for test】
def split_input_data(input_data_df):
    # 【Separate horizontally】
    predictors_tf = input_data_df[input_data_df.columns[:T]]
    class_tf = input_data_df[input_data_df.columns[T:]]
    print('predictors_tf: ',predictors_tf,'class_tf: ',class_tf)
    # 【Separete vertically】80 : 20 = Train : Test
    training_set_size = int(len(input_data_df) * 0.8)
    test_set_size = len(input_data_df) - training_set_size
    return Dataset(
        training_predictors = predictors_tf[:training_set_size],
        training_class = class_tf[:training_set_size],
        test_predictors = predictors_tf[training_set_size:],
        test_class = class_tf[training_set_size:],
    )

# <<<<<<<<<<<<<<<<< 【For The Neural Network】>>>>>>>>>>>>>
# 【Training/Test Generate the data (dictionary type)】
def feed_dict(env, test = False):
    prefix = 'test' if test else 'training'
    predictors = getattr(env.dataset, '{}_predictors'.format(prefix))
    Class = getattr(env.dataset, '{}_class'.format(prefix))
    return {
        env.feature_data: predictors.values,
        env.actual_class: Class.values
    }

# 【Construct the Simple Neural Network】
def neural_network(dataset):
    sess = tf.Session()
    feature_data = tf.placeholder("float", [None, T])
    actual_class = tf.placeholder("float", [None, 1])

    weights1 = tf.Variable(tf.truncated_normal([T,T], stddev=0.0001), name='Weights1')
    biases1 = tf.Variable(tf.ones([T]), name='Biases1')
    weights2 = tf.Variable(tf.truncated_normal([T,1], stddev=0.0001), name='Weights2')
    biases2 = tf.Variable(tf.ones([1]), name='Biases2')

    # 【Activation function is tanh】
    hidden_layer = tf.nn.sigmoid(tf.matmul(feature_data, weights1) + biases1)
    model = tf.nn.tanh(tf.matmul(hidden_layer, weights2) + biases2)

    #cost = -tf.reduce_sum(actual_class * tf.log(model))# Entropy ✗
    cost = tf.reduce_mean(tf.square(actual_class - model))# 【Squared Error, simply】
    
    training_step = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)

    init = tf.initialize_all_variables()
    sess.run(init)

    return Environ(sess = sess, model = model, actual_class = actual_class, training_step = training_step, dataset = dataset, feature_data = feature_data)

# 【Machine Learning】
def train(env, steps, checkin_interval):
    # Prepare the error range tensor
    error_range_tensor = ERROR_RANGE * tf.ones_like(env.actual_class)
    # Judge whether to go through the error range（BOOL)
    correct_prediction = tf.logical_and(
        tf.less_equal(env.actual_class - error_range_tensor, env.model),
        tf.less_equal(env.model, env.actual_class + error_range_tensor),
    )
    # Accuracy (Mean of the BOOL above)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))*100
    # Training...
    for i in range(1, 1 + steps):
        env.sess.run(
            env.training_step,
            feed_dict = feed_dict(env, test = False)
        )
        # Present the progress every interval
        if i % checkin_interval == 0:
            print('学習回数:',i,', 正解率 (%)', env.sess.run(
                    accuracy,
                    feed_dict(env, test=False)
                )
            )
    
    # Show Conclusion after the test is finished
    tf_confusion_metrics(env.model, env.actual_class, env.sess, feed_dict(env, True))

                         
# 【Judge the prediction is correct or not】
# After the learning is finished, calculate the tensor (session.run()) and compare the teacher data in the error range
def tf_confusion_metrics(model, actual_class, session, feed_dict):
    # a. Prepare the error range tensor
    error_range_tensor = ERROR_RANGE * tf.ones_like(actual_class)
    # b. Prepare the operator to enumerate correct answer
    accuracy_operator = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.less_equal(actual_class - error_range_tensor, model),
                tf.less_equal(model, actual_class + error_range_tensor),
            ),
        "float"
        )
    )
    # c. Tensor to get the parent set
    all_operator = tf.reduce_sum(
        tf.cast(tf.equal(model, model), "float")
    )
    # d. Calculate the accuracy
    num_accuracy, num_all = session.run(
        [accuracy_operator, all_operator], feed_dict
    )
    print('num_accuracy:',num_accuracy,'type:',type(num_accuracy), 'num_all', num_all, 'type:', type(num_all))

    accuracy = num_accuracy / num_all
    print('【The accuracy of the model (%)】: ', accuracy*100)


    #【Save the model】
    saver = tf.train.Saver()
    saver.save(session,'./model')



# <<<<<<<<<<<<<<<< Main Function >>>>>>>>>>>>>>>>
# 【Load the data】
input_data_df = load_input_data_df()
#pd.set_option('display.max_colwidth', 100)#for chech in terminal
#pd.set_option('display.max_columns', 200)
#pd.set_option('display.max_rows', 10000)
#print('input_data_df:', input_data_df)
print('【Successfully loaded the data】')
# 【Separete the data】
dataset = split_input_data(input_data_df)
print('dataset:',dataset)
print('【Successfully separeted the data】')
# 【Cunstruct the Neural Network】
env = neural_network(dataset)
# 【Start Learning, Test, Evaluate the prediction】
train(env, steps=50000, checkin_interval=1000)

# <<<<<<<<<<<<<<< End of The Code >>>>>>>>>>>>>>>>>>







