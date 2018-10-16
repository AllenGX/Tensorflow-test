import tensorflow as tf

a=tf.constant([1.0,2.0],name="a")
b=tf.constant([2.0,3.0],name="b")

result=a+b

#两者效果类似
sess=tf.Session()
print(sess.run(result))
sess.close()

#两者效果类似
with tf.Session() as sess:
    print(result.eval())



#判断计算图是否为默认的
#为Tensorflow调用GPU提供了可能
print(a.graph is tf.get_default_graph())
