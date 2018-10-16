import tensorflow as tf

from numpy.random import RandomState

#定义神经网络结构
#输入层 2行3列矩阵
#隐藏层 3行1列矩阵
#设置随机种子 保证每次结果一致  seed=1
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#数据类型 float32
#数据结构 2列行数不定（None）
#名称 name  x-input
x=tf.placeholder(tf.float32,shape=(None,2),name="x-input")
y_=tf.placeholder(tf.float32,shape=(None,1),name="y-input")

#定义向前传播  x->w1
a=tf.matmul(x,w1)
#定义向前传播  a->w2
y=tf.matmul(a,w2)

#定义损失函数和反向传播的算法
cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
learning_rate=0.001
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#随机生成模拟数据集
rdm=RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2)

#定义学习规则
#X1+X2<1的都是正确答案
Y=[[int(x1+x2<1)]for (x1,x2)in X]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("w1:",sess.run(w1))
    print("w2:",sess.run(w2))

    #训练次数 5000
    STEPS=5000
    #定义batch大小
    batch_size=8


    for i in range(STEPS):
        #每次从样本取出   dataset_size - ( i * batch_size ) % dataset_size 个样品进行训练
        start=(i*batch_size)%dataset_size
        end=min(start+batch_size,dataset_size)

        #进行训练优化
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

        #输出交叉墒和训练次数
        #交叉墒越小说明越精确
        if i%1000==0:
            total_cross_entropy=sess.run(
                cross_entropy,feed_dict={x:X,y_:Y})
            print("After %d training step(s),cross entropy on all data is %g" %(i,total_cross_entropy))

    print("w1:",sess.run(w1))
    print("w2:",sess.run(w2))





    