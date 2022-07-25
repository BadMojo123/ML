import numpy
import B_Mind
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,GRU
import keras.models


def Boleslaw(
	testSet = 0.98,
	l1_size=512,
	l2_size=512,
	l3_size=200,
	l4_size=0,
	l5_size=0,
	l1_activation='relu',
	l2_activation='relu',
	l3_activation='relu',
	l4_activation='relu',
	l5_activation='relu',
	loss='mean_squared_error',
	optimizer='rmsprop',
	metrics=['accuracy'],
	epochs=80,
	validation_split=0.01,
	batch_size=200,
	FILE_NAME = 'histData/GER30Cash15.csv',
	dropout=0.2,
	histReq = 16
	):

	rand = time.time()
	Data = B_Mind.LoadFile(FILE_NAME)

	NormData = B_Mind.AddNormData(Data)

	Labels, Data = B_Mind.AddIndicatorsAndHist(NormData, histReq)
	train_x, train_y, test_x,test_y = B_Mind.SelectData(Data, Labels, testSet)
	i,j,k = train_x.shape
	# numpy.savetxt("data/x_ful.csv", train_x[100,:,:], delimiter=",")

	model = Sequential()
	model.add(GRU(l1_size,return_sequences=True,activation=l1_activation,input_shape=(j, k)))
	model.add(Dropout(dropout))
	if l2_size>0 :
		model.add(GRU(l2_size,return_sequences=True,activation=l2_activation))
		model.add(Dropout(dropout))
	# if l3_size>0:
	# 	model.add(GRU(l3_size,return_sequences=True,activation=l3_activation))
	# if l4_size>0:
	# 	model.add(Dense(l4_size))
	# 	model.add(Activation(l4_activation))
	# if l5_size>0:
	# 	model.add(Dense(l5_size))
	# 	model.add(Activation(l5_activation))
	model.add(Flatten())
	model.add(Dense(1))

	callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)
	callback_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)
	callback_tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True)
	callbacks = [callback_early_stopping,
				callback_tensorboard,
				callback_reduce_lr]

	model.compile(loss=loss,
	              optimizer=optimizer,
	              metrics=metrics)
	history = model.fit(train_x, train_y,
	                    batch_size=batch_size,
	                    epochs=epochs,
	                    verbose=0,
	                    validation_split=validation_split,
	                    callbacks=callbacks,
	                    shuffle=False)


	score = model.evaluate(test_x, test_y,
	                       batch_size=batch_size, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	model.summary()

	prediction = model.predict(test_x)

	i = len(test_y)
	t = numpy.arange(0,i,1)
	fig, ax = plt.subplots()
	ax.plot(t, test_y,'b-',t,prediction,'g-')
	ax.grid()
	plt.show()
	plt.close("all")
	

Boleslaw()