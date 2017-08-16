import numpy as np
import tensorflow as tf
#from CNN_model import Model
from VGG_model import Model
from Read_cifar10 import loaddata
import matplotlib.pyplot as plt


#load = loaddata('D:\\Python\\PF\\DATA\\rawdata\\grayscale\\',grayscale=True) #grayscale
load = loaddata('D:\\Python\\PF\\DATA\\rawdata\\non_grayscale\\',grayscale=False) #rawdata
x_data, x_label = load._loaddata(one_hot_encoding=True,normalization=True,single_data=True) #one-hot 형태로.

training_epochs = 10
batch_size = 100
sess = tf.Session()

models = []
num_models = 1
for m in range(num_models):
    models.append(Model(sess, 'model' + str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Started!')

import time
# 시작 시간 체크
stime = time.time()

for epoch in range(training_epochs): #training epoch = 20
    avg_cost_list = np.zeros(len(models))
    total_batch = int(x_data.shape[0] / batch_size)
    for i in range(0,10000,batch_size):
        batch_xs, batch_ys = x_data[i:i+batch_size],x_label[i:i+batch_size]
        # 각각의 모델 훈련
        for idx, m in enumerate(models):
            c, a, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[idx] += c / total_batch
            print('accuracy: ', a)
    print('Epoch: ', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
print('Learning Finished!')


##############################################################
# 테스트 모델에서 정확도(accuracy) 체크
test_x, test_y = load._loaddata('test_batch',one_hot_encoding=True) #one-hot 형태로.
test_size = len(test_y)
predictions = np.zeros(test_size * 10).reshape(test_size, 10)

for idx, m in enumerate(models):
    print(idx, 'Accuracy: ', m.get_accuracy(test_x, test_y))
    p = m.predict(test_x)
    predictions += p

ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_y, 1))
ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble Accuracy: ', sess.run(ensemble_accuracy))

# 종료 시간 체크
etime = time.time()
print('consumption time : ', round(etime-stime, 6))


