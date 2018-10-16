import tensorflow as tf

w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x=tf.constant([[0.7,0.9]])

#                 ***                *
#       **  X           =>  ***  X   *  =>  *
#                 ***                *

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

with tf.Session() as sess:
    #初始化w1
    sess.run(w1.initializer)
    #初始化w2
    sess.run(w2.initializer)
    print(sess.run(a))
    print(sess.run(y))


with tf.Session() as sess:
    #初始化全部
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(y))