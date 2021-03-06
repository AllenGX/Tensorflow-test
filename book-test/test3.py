import tensorflow as tf

w1=tf.Variable(tf.random_normal([2,3],stddev=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1))

# 规定 feed_dict 的取值类型
# 替换  x=tf.constant([0.7,0.9],[0.1,0.4],[0.5,0.8])
x=tf.placeholder(tf.float32,shape=(3,2),name="input")
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))    #确定取值范围(注意是范围而不是固定值)
                                                                        #这点和constant不同