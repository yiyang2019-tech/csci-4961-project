# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 08:04:10 2021

@author: caiy3
"""
import tensorflow as tf
    
def batch_norm(in_tensor,is_training):
    return tf.contrib.layers.batch_normalization(in_tensor,center=True,decay=0.9997,scale=True,
                                                 epsilon=1.1e-5,training=is_training)

def make_divisible(v,divisor,min_value=None):

    k=v/divisor+1
    if min_value!=None and k*divisor>min_value:
        return (k-1)*divisor
    return k*divisor
def conv(in_tensor,layer_name,k,s,out_c,is_training=True, with_bias=False, use_bn=True, relu=tf.nn.relu6):
    with tf.variable_scope(layer_name):
        in_size=in_tensor.get_shape().as_list()
        ss=[1,s,s,1]
        kernel_shape=[k,k,in_size[3],out_c]
        #conv
        kernel = tf.get_variable('weights',kernel_shape,tf.float32,tf.contrib.layer.xavier_initializer_conv2d(),
                                 trainable=is_training,collections=['wd','variables','filters'])
        x=tf.nn.conv2d(in_tensor,kernel,ss,padding='SAME')
        if with_bias:
            biases=tf.get_variable('biases',[kernel_shape[3]],tf.float32,tf.constant_initializer(0.0001),
                                   trainable=is_training,
                                   collection3=['wd','variables','biases'])
            x=tf.nn.bias_add(x, biases)
        if use_bn:
            x=batch_norm(x,is_training)
        if not relu is None:
            x=relu(x,"Relu6")
        return x

def expanded_conv(in_tensor,layer_name,k,s,out_c,expand_rate,is_training=True,relu=tf.nn.relu6):
    in_size=in_tensor.get_shape().as_list()
    expansion_size=make_divisible(in_size[3]*expand_rate,8)
    x=in_tensor
    if expansion_size>in_size[3]:
        with tf.variable_scope(layer_name+'_expand'):
            pw_kernel_shape=[1,1,in_size[3],expansion_size]
            pw_kernel = tf.get_variable('weights',pw_kernel_shape,tf.float32,tf.contrib.layer.xavier_initializer_conv2d(),
                                 trainable=is_training,collections=['wd','variables','filters'])
            x= tf.nn.conv2d(in_tensor,pw_kernel,[1,1,1,1],padding='SAME')
            x=batch_norm(x,is_training=is_training)
            x=relu(x,"Relu6")
        with tf.variable_scope(layer_name+'_depthwise'):
            ss =[1,s,s,1]
            dw_kernel_shape=[k,k,x.get_shape().as_list()[3],1]
            dw_kernel=tf.get_variable('depthwise_weights',dw_kernel_shape,tf.float32,tf.contrib.layer.xavier_initializer_conv2d(),
                                 trainable=is_training,collections=['wd','variables','filters'])
            x=tf.nn.depthwise_conv2d(x,dw_kernel,ss,padding='SAME',name=layer_name+'_depthwise')
            x=batch_norm(x,is_training=is_training)
            x=relu(x,'Relu6')
        with tf.variable_scope(layer_name+'_project'):
            pw_kernel_shape=[1,1,x.get_shape().as_list()[3],out_c]
            pw_kernel=tf.get_variable('weights',pw_kernel_shape,tf.float32,tf.contrib.layer.xavier_initializer_conv2d(),
                                 trainable=is_training,collections=['wd','variables','filters'])
            x=tf.nn.conv2d(x,pw_kernel,[1,1,1,1],padding='SAME')
            x=batch_norm(x,is_training=is_training)
        if s==1 and out_c==in_size[3]:
            return in_tensor+x
        return x



            

def global_pool(input_tensor,pool_op=tf.nn.avg_pool):
    return tf.nn.avg_pool(input_tensor, [7,7,input_tensor.get_shape()[3]], padding='SAME')
###Bottleneck


    