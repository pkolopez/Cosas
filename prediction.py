#!/usr/bin/env python3
#  coding: utf-8 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn  

rnn_unit=10       
input_size=7
lr=0.0006    
f=open('datos.csv') 
df=pd.read_csv(f)     
data=df.iloc[:,2:10].values  
pre_days=3
dijitian=1
time_step=20
output_size=pre_days

def data_preprocess(data,pre_days):
    train_data=data
    target=data[:,1]
    for i in range(pre_days):
        target=np.delete(target,0,axis=0)
        train_data=np.delete(train_data,-1,axis=0)
        train_data=np.column_stack((train_data,target))
    tmp=target.shape[0]
    np.delete(train_data,[range(tmp,train_data.shape[0])],axis=0)
    return train_data


def get_test_data(time_step,test_begin):
    test_data=data_preprocess(data,pre_days)
    data_test=test_data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  
    size=(len(normalized_test_data)+time_step-1)//time_step   
    test_x,test_y=[],[]  
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:7]
       y=normalized_test_data[i*time_step:(i+1)*time_step,7:]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:7]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,7:]).tolist())
    return mean,std,test_x,test_y


weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,output_size]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[output_size,]))
       }

def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  
    cell=rnn.BasicLSTMCell(rnn_unit,state_is_tuple=False)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


mean,std,test_x,test_y=get_test_data(time_step=20,test_begin=5800)
def prediction(time_step=20,dijitian=2):
    checkpoint_dir='tmp'
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step=20,test_begin=5800)
    pred,f=lstm(X)     
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        
        ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path) 
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]}) 
#          print(prob)
          predict_tmp=prob[:,dijitian].reshape((-1))
          test_predict.extend(predict_tmp)
        test_y=np.array(test_y)[:,dijitian]*std[7+dijitian-1]+mean[7+dijitian-1]
        test_predict=np.array(test_predict)*std[7+dijitian-1]+mean[7+dijitian-1]

        
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()
#    return test_predict
    return test_predict,test_y
#p=prediction(time_step,dijitian)
test_predict,test_y=prediction(time_step,dijitian) 