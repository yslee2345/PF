import tensorflow as tf, numpy as np



class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.learning_rate = 0.01
        self.dropout_rate = 0.7
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool, name='training') #boolean형태
            self.X = tf.placeholder(tf.float32, [None, 1024], name='x_data') #데이터
            X_img = tf.reshape(self.X, shape=[-1, 32, 32, 1]) #32 * 32로 reshape
            self.Y = tf.placeholder(tf.float32, [None, 10], name='y_data') #라벨

            ############################################################################################################
            ## ▣ Convolution 계층 - 1
            ##  ⊙ 합성곱 계층 → filter: (3, 3), output: 32 개, 초기값: He / shape = [행,열,input feature, output feature]
            ##  ⊙ 편향        → shape: 32, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 풀링 계층   → Max Pooling ( 커널: 2*2, 스트라이드 2
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################

            self.W1 = tf.get_variable(name='W1', shape=[3, 3, 1, 32], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b1 = tf.Variable(tf.constant(value=0.001, shape=[32]), name='b1')#편향
            self.L1 = tf.nn.conv2d(input=X_img, filter=self.W1, strides=[1, 1, 1, 1], padding='SAME')
            self.L1 = self.batch_norm(self.L1,self.training,scale=True,name='Conv1_BN')
            self.L1 = self.parametric_relu(self.L1, 'R1')
            self.L1 = tf.nn.max_pool(value=self.L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 32x32 -> 16x16
            #self.L1 = tf.layers.dropout(inputs=self.L1, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## ▣ Convolution 계층 - 2
            ##  ⊙ 합성곱 계층 → filter: (3, 3), output: 64 개, 초기값: He
            ##  ⊙ 편향        → shape: 64, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 풀링 계층   → Max Pooling
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            self.W2 = tf.get_variable(name='W2', shape=[3, 3, 32, 64], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b2 = tf.Variable(tf.constant(value=0.001, shape=[64]), name='b2')
            self.L2 = tf.nn.conv2d(input=self.L1, filter=self.W2, strides=[1, 1, 1, 1], padding='SAME')
            self.L2 = self.batch_norm(self.L2, self.training, scale=True, name='Conv2_BN')
            self.L2 = self.parametric_relu(self.L2, 'R2')
            self.L2 = tf.nn.max_pool(value=self.L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 16x16 -> 8x8
            #self.L2 = tf.layers.dropout(inputs=self.L2, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## ▣ Convolution 계층 - 3
            ##  ⊙ 합성곱 계층 → filter: (3, 3), output: 128 개, 초기값: He
            ##  ⊙ 편향        → shape: 128, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 풀링 계층   → X
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            self.W3 = tf.get_variable(name='W3', shape=[3, 3, 64, 128], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b3 = tf.Variable(tf.constant(value=0.001, shape=[128]), name='b3')
            self.L3 = tf.nn.conv2d(input=self.L2, filter=self.W3, strides=[1, 1, 1, 1], padding='SAME')
            self.L3 = self.batch_norm(self.L3, self.training, scale=True, name='Conv3_BN')
            self.L3 = self.parametric_relu(self.L3, 'R3')
            #self.L3 = tf.layers.dropout(inputs=self.L3, rate=self.dropout_rate, training=self.training)
            #self.L3 = tf.nn.max_pool(value=self.L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 8x8 -> 4x4

            ############################################################################################################
            ## ▣ Convolution 계층 - 4
            ##  ⊙ 합성곱 계층 → filter: (3, 3), output: 256 개, 초기값: He
            ##  ⊙ 편향        → shape: 256, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 풀링 계층   → X
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            self.W4 = tf.get_variable(name='W4', shape=[3, 3, 128, 256], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b4 = tf.Variable(tf.constant(value=0.001, shape=[256]), name='b4')
            self.L4 = tf.nn.conv2d(input=self.L3, filter=self.W4, strides=[1, 1, 1, 1], padding='SAME')
            self.L4 = self.batch_norm(self.L4, self.training, scale=True, name='Conv4_BN')
            self.L4 = self.parametric_relu(self.L4, 'R4')
            #self.L4 = tf.layers.dropout(inputs=self.L4, rate=self.dropout_rate, training=self.training)
            self.L4 = tf.reshape(self.L4, shape=[-1, 8 * 8 * 256])

            ############################################################################################################
            ## ▣ fully connected 계층 - 1
            ##  ⊙ 가중치 → shape: (7 * 7 * 256, 625), output: 625 개, 초기값: He
            ##  ⊙ 편향   → shape: 625, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            self.W5 = tf.get_variable(name='W5', shape=[8 * 8 * 256, 625], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b5 = tf.Variable(tf.constant(value=0.001, shape=[625], name='b5'))
            self.L5 = self.parametric_relu(tf.matmul(self.L4, self.W5) + self.b5, 'R5')
            self.L5 = tf.layers.dropout(inputs=self.L5, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## fully connected 계층 - 2
            ##  ⊙ 가중치 → shape: (625, 625), output: 625 개, 초기값: He
            ##  ⊙ 편향   → shape: 625, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Leaky Relu
            ##  ⊙ 드롭 아웃 구현
            ############################################################################################################
            self.W6 = tf.get_variable(name='W6', shape=[625, 625], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b6 = tf.Variable(tf.constant(value=0.001, shape=[625], name='b6'))
            self.L6 = self.parametric_relu(tf.matmul(self.L5, self.W6) + self.b6, 'R6')
            self.L6 = tf.layers.dropout(inputs=self.L6, rate=self.dropout_rate, training=self.training)

            ############################################################################################################
            ## 출력층
            ##  ⊙ 가중치 → shape: (625, 10), output: 10 개, 초기값: He
            ##  ⊙ 편향   → shape: 10, 초기값: 0.001
            ##  ⊙ 활성화 함수 → Softmax
            ############################################################################################################
            self.W7 = tf.get_variable(name='W7', shape=[625, 10], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            self.b7 = tf.Variable(tf.constant(value=0.001, shape=[10], name='b7'))
            self.logits = tf.matmul(self.L6, self.W7) + self.b7

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

