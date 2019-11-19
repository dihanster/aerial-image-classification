import numpy as np
import tflearn
import h5py


from tensorflow import reset_default_graph
reset_default_graph()

from __future__ import print_function

import numpy as np
import tflearn
import h5py

# Load CSV file, indicate that the first column represents labels
#from tflearn.data_utils import load_csv
#data, labels = load_csv('Image4.csv', target_column=0,
 #                       categorical_labels = True, n_classes=2)

#BaseValidation, labelsBase = load_csv('BaseOriginal.csv', target_column=0,
#                       categorical_labels = True, n_classes=2)

Resultado = []

for iter in range(30):
	reset_default_graph()
	import numpy as np
	import tflearn
	import h5py
	from tflearn.data_utils import load_csv
	data, labels = load_csv('BaseImagens.csv', target_column=0, categorical_labels = True, n_classes=2)
	BaseValidation, labelsBase = load_csv('BaseOriginal.csv', target_column=0, categorical_labels = True, n_classes=2)
	net = tflearn.input_data(shape=[None, 128])
	net = tflearn.fully_connected(net, 64)
	net = tflearn.fully_connected(net, 64)
	net = tflearn.fully_connected(net, 64)
	net = tflearn.fully_connected(net, 64)
	net = tflearn.fully_connected(net, 64)
	net = tflearn.fully_connected(net, 2, activation='softmax')
	sgd = tflearn.SGD(learning_rate = 0.3) 
	net = tflearn.regression(net , optimizer = sgd)
	model = tflearn.DNN(net)
	model.fit(data, labels, n_epoch = 500, batch_size= 100, show_metric=True) 
	pred = model.predict(BaseValidation)
	for i in range(len(BaseValidation)):
		if(pred[i][0] >= 0.5):
			pred[i][0] = 1  #Entao eh cafe
		else:
			pred[i][0] = 0  #Entao nao eh cafe
	correto = 0
	for i in range(len(labelsBase)):
		if(pred[i][0] == labelsBase[i][0]):
			correto += 1
	Resultado.append(correto/len(BaseValidation))
	print(correto/(len(BaseValidation)))



