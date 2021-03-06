
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn  
import matplotlib.pyplot as plt

rnn_unit=10       
input_size=7
lr=0.0006
pre_days=3
output_size=pre_days
f=open('datos.csv') 
df=pd.read_csv(f)    
data=df.iloc[:,2:].values

def data_preprocess(data,pre_days):
    train_data=data
    target=data[:,3]
    for i in range(pre_days):
        target=np.delete(target,0,axis=0)
        train_data=np.delete(train_data,-1,axis=0)
        train_data=np.column_stack((train_data,target))
    tmp=target.shape[0]
    np.delete(train_data,[range(tmp,train_data.shape[0])],axis=0)
    return train_data

train_data=data_preprocess(data,pre_days)

def get_train_data(batch_size,time_step,train_begin,train_end):
    dely_day=0
    batch_index=[]
    data_train=train_data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  
    train_x,train_y=[],[]   
    for i in range(len(normalized_train_data)-time_step-dely_day):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:7]

       y=normalized_train_data[i+dely_day:i+dely_day+time_step,7:]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,output_size]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[output_size,]))
       }


def lstm(X,a):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])

    cell=rnn.BasicLSTMCell(rnn_unit,state_is_tuple=True)
      
    init_state=cell.zero_state(batch_size,dtype=tf.float32)

    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


def train_lstm(batch_size=80,time_step=20,train_begin=0,train_end=5800,a='rnn1'):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    pred,f=lstm(X,1)
    
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))

    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
 
    checkpoint_dir=''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(101):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            if i % 100==0:
                print("save model：",saver.save(sess,checkpoint_dir+'tmp/model.ckpt',global_step=i))
    del pred
    del f
train_lstm()