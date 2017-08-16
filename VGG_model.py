'''
VGG : conv-relu-conv-relu-maxpooling 구조


'''


import tensorflow as tf, numpy as np



class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.learning_rate = 0.01
        self.dropout_rate = 0.7
        self.VGG()

    def VGG(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool, name='training') #boolean형태
            self.X = tf.placeholder(tf.float32, [None, 3072], name='x_data') #데이터
            X_img = tf.reshape(self.X, shape=[-1, 32, 32, 3]) #32 * 32로 reshape
            self.Y = tf.placeholder(tf.float32, [None, 10], name='y_data') #라벨

            ############################################################################################################
            ## ▣ VGG계층 - 1 conv-relu-conv-relu-maxpooling
            ##  - 합성곱 계층 → filter: (3, 3), output: 32 개, 초기값: He / shape = [행,열,input feature, output feature]
            ##  - 배치 정규화
            ##  - 편향        → shape: 32, 초기값: 0.001
            ##  - 활성화 함수 → pRelu
            ##  - 풀링 계층   → Max Pooling ( 커널: 2*2, 스트라이드 2)
            ##  - conv계층 통과시엔 shape의 변화 없음. maxpooling 통과시에만 shape이 절반으로 줄어든다.
            ##  - 32 X 32 -> 16 X 16
            ############################################################################################################
            self.W1_1 = tf.get_variable(name='W1_1', shape=[3, 3, 3, 64], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b1_1 = tf.Variable(tf.constant(value=0.001, shape=[64]), name='b1_1')#편향
            self.W1_2 = tf.get_variable(name='W1_2', shape=[3, 3, 64, 64], dtype=tf.float32,initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b1_2 = tf.Variable(tf.constant(value=0.001, shape=[64]), name='b1_2')  # 편향

            self.L1 = tf.nn.conv2d(input=X_img, filter=self.W1_1, strides=[1, 1, 1, 1], padding='SAME')
            self.L1 = self.batch_norm(self.L1,self.training,scale=True,name='VGG1_BN1')
            self.L1 = self.parametric_relu(self.L1, 'R1_1')
            self.L1 = tf.nn.conv2d(input=self.L1, filter=self.W1_2,strides=[1,1,1,1],padding='SAME')
            self.L1 = self.batch_norm(self.L1,self.training,scale=True,name='VGG1_BN2')
            self.L1 = self.parametric_relu(self.L1, 'R1_2')
            self.L1 = tf.nn.max_pool(value=self.L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 32x32 -> 16x16
            ############################################################################################################
            ## ▣ VGG계층 - 2 conv-relu-conv-relu-maxpooling
            ##  - 합성곱 계층 → filter: (3, 3), output: 64 개, 초기값: He
            ##  - 배치 정규화
            ##  - 편향        → shape: 64, 초기값: 0.001
            ##  - 활성화 함수 → Leaky Relu
            ##  - 풀링 계층   → Max Pooling
            ##  - 16 X 16 -> 8 X 8
            ############################################################################################################
            self.W2_1 = tf.get_variable(name='W2_1', shape=[3, 3, 64, 128], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b2_1 = tf.Variable(tf.constant(value=0.001, shape=[128]), name='b2_1')
            self.W2_2 = tf.get_variable(name='W2_2', shape=[3, 3, 128, 128], dtype=tf.float32,initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b2_2 = tf.Variable(tf.constant(value=0.001, shape=[128]), name='b2_2')

            self.L2 = tf.nn.conv2d(input=self.L1, filter=self.W2_1, strides=[1, 1, 1, 1], padding='SAME')
            self.L2 = self.batch_norm(self.L2, self.training, scale=True, name='VGG2_BN1')
            self.L2 = self.parametric_relu(self.L2, 'R2_1')
            self.L2 = tf.nn.conv2d(input=self.L2,filter=self.W2_2,strides=[1,1,1,1],padding='SAME')
            self.L2 = self.batch_norm(self.L2,self.training,scale=True,name='VGG2_BN2')
            self.L2 = self.parametric_relu(self.L2,'R2_2')
            self.L2 = tf.nn.max_pool(value=self.L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 16x16 -> 8x8
            ############################################################################################################
            ## ▣ VGG 계층 - 3 conv-relu-conv-relu-conv-relu-maxpooling
            ##  - 합성곱 계층 → filter: (3, 3), output: 128 개, 초기값: He
            ##  - 배치 정규화
            ##  - 편향        → shape: 128, 초기값: 0.001
            ##  - 활성화 함수 → Leaky Relu
            ##  - 풀링 계층   → X
            ##  - 드롭 아웃 구현
            ############################################################################################################
            self.W3_1 = tf.get_variable(name='W3_1', shape=[3, 3, 128, 256], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b3_1 = tf.Variable(tf.constant(value=0.001, shape=[256]), name='b3_1')
            self.W3_2 = tf.get_variable(name='W3_2', shape=[3, 3, 256, 256], dtype=tf.float32,initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b3_2 = tf.Variable(tf.constant(value=0.001, shape=[256]), name='b3_2')
            self.W3_3 = tf.get_variable(name='W3_3', shape=[3, 3, 256, 256], dtype=tf.float32,initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b3_3 = tf.Variable(tf.constant(value=0.001, shape=[256]), name='b3_3')

            self.L3 = tf.nn.conv2d(input=self.L2, filter=self.W3_1, strides=[1, 1, 1, 1], padding='SAME')
            self.L3 = self.batch_norm(self.L3, self.training, scale=True, name='VGG3_BN1')
            self.L3 = self.parametric_relu(self.L3, 'R3_1')
            self.L3 = tf.nn.conv2d(input=self.L3,filter=self.W3_2,strides=[1,1,1,1],padding='SAME')
            self.L3 = self.batch_norm(self.L3,self.training,scale=True,name='VGG3_BN2')
            self.L3 = self.parametric_relu(self.L3,'R3_2')
            self.L3 = tf.nn.conv2d(input=self.L3,filter=self.W3_3,strides=[1,1,1,1],padding='SAME')
            self.L3 = self.batch_norm(self.L3,self.training,scale=True,name='VGG3_BN3')
            self.L3 = self.parametric_relu(self.L3,'R3_3')
            self.L3 = tf.nn.max_pool(value=self.L3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #8X8 -> 4X4

            self.L4 = tf.reshape(self.L3, shape=[-1, 4 * 4 * 256])

            ############################################################################################################
            ## ▣ fully connected 계층 - 1
            ##  ⊙ 가중치 → shape: (4 * 4 * 256, 625), output: 625 개, 초기값: He
            ##  ⊙ 편향   → shape: 625, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            self.W5 = tf.get_variable(name='W5', shape=[4 * 4 * 256, 625], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b5 = tf.Variable(tf.constant(value=0.001, shape=[625], name='b5'))
            self.L5 = tf.matmul(self.L4,self.W5)+self.b5
            self.L5 = self.batch_norm(self.L5,self.training,scale=True,name='FC_BN1')
            self.L5 = self.parametric_relu(self.L5, 'R5')
            #self.L5 = tf.layers.dropout(inputs=self.L5, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## fully connected 계층 - 2
            ##  ⊙ 가중치 → shape: (625, 625), output: 625 개, 초기값: He
            ##  ⊙ 편향   → shape: 625, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            self.W6 = tf.get_variable(name='W6', shape=[625, 625], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b6 = tf.Variable(tf.constant(value=0.001, shape=[625], name='b6'))
            self.L6 = tf.matmul(self.L5,self.W6)+self.b6
            self.L6 = self.batch_norm(self.L6,self.training,scale=True,name='FC_BN2')
            self.L6 = self.parametric_relu(self.L6, 'R6')
            #self.L6 = tf.layers.dropout(inputs=self.L6, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## fully connected 계층 - 3
            ##  ⊙ 가중치 → shape: (625, 625), output: 625 개, 초기값: He
            ##  ⊙ 편향   → shape: 625, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################

            self.W7 = tf.get_variable(name='W7', shape=[625, 625], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b7 = tf.Variable(tf.constant(value=0.001, shape=[625], name='b7'))
            self.L7 = tf.matmul(self.L6, self.W7) + self.b7
            self.L7 = self.batch_norm(self.L7,self.training,scale=True,name='FC_BN3')
            self.L7 = self.parametric_relu(self.L7, 'R7')
            #self.L7 = tf.layers.dropout(inputs=self.L7, rate=self.dropout_rate, training=self.training)


            ############################################################################################################
            ## 출력층
            ##  ⊙ 가중치 → shape: (625, 10), output: 10 개, 초기값: He
            ##  ⊙ 편향   → shape: 10, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Softmax
            ############################################################################################################
            self.W8 = tf.get_variable(name='W8', shape=[625, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b8 = tf.Variable(tf.constant(value=0.001, shape=[10], name='b8'))
            self.logits = tf.matmul(self.L7, self.W8) + self.b8

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)) + 0.01*tf.reduce_sum(tf.square(self.W7))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(self.logits, 1), tf.arg_max(self.Y, 1)), dtype=tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: False})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: False})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.accuracy, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: True})

    def parametric_relu(self, _x, name):
        alphas = tf.get_variable(name, _x.get_shape()[-1], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        return pos + neg

    def batch_norm(self, input, training, scale, name, decay=0.99):
        return tf.contrib.layers.batch_norm(input, decay=decay, scale=scale, is_training=training, updates_collections=None, scope=name)


