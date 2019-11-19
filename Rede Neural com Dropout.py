import numpy as np
import tflearn

from tensorflow import reset_default_graph
reset_default_graph()

from __future__ import print_function

import numpy as np
import tflearn
from __future__ import division, print_function, absolute_import



from tflearn.data_utils import load_csv
X, Y = load_csv('Image4.csv', target_column=0,
                        categorical_labels = True, n_classes=2)

testX, testY = load_csv('BaseOriginal.csv', target_column=0,
                       categorical_labels = True, n_classes=2)


# Building deep neural network
input_layer = tflearn.input_data(shape=[None, 128])
dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.8)

dense3 = tflearn.fully_connected(dropout1, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout3 = tflearn.dropout(dense3, 0.8)


dense4 = tflearn.fully_connected(dropout3, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout4 = tflearn.dropout(dense4, 0.8)

dense2 = tflearn.fully_connected(dropout4, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.8)

softmax = tflearn.fully_connected(dropout2, 2, activation='softmax')

# Regression using SGD with learning rate decay and Top-3 accuracy
sgd = tflearn.SGD(learning_rate=0.3, lr_decay=0.96, decay_step=1000)

top_k = tflearn.metrics.Top_k(3)

net = tflearn.regression(softmax, optimizer=sgd, #metric=top_k,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch = 40, #validation_set=(testX, testY),
          show_metric=True, run_id="dense_model")


pred = model.predict(testX)

for i in range(len(testX)):
	if(pred[i][0] >= 0.5):
		pred[i][0] = 1  #Entao eh cafe
	else:
		pred[i][0] = 0  #Entao nao eh cafe

correto = 0

for i in range(len(testY)):
	if(pred[i][0] == testY[i][0]):
		correto += 1

print(correto/(len(testX)))

