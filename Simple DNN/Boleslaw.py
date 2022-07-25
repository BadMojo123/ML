import numpy
import B_Mind
import time
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, GRU,Flatten
import keras.models


def Boleslaw(
	testSet = 0.95,
	l1_size=100,
	l2_size=100,
	l3_size=100,
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
	epochs=30,
	validation_split=0.001,
	batch_size=64,
	FILE_NAME = 'histData/GER30Cash1.csv',
	dropout=0.5,
	histReq = 100
	):

	modelName = 'klony/Boleslaw.h5'
	rand = time.time()
	Data = B_Mind.LoadFile(FILE_NAME)

	NormData = B_Mind.AddNormData(Data)

	Labels, Data = B_Mind.AddIndicatorsAndHist(NormData, histReq)
	train_x, train_y, test_x,test_y = B_Mind.SelectData(Data, Labels, testSet)
	i,j,k = train_x.shape
	print(train_x.shape)
	# numpy.savetxt("data/x_ful.csv", train_x[100,:,:], delimiter=",")
	
	while True:
		try:
			model = load_model(modelName)
			print("Model loaded")
			mExist = 1
			break
		except OSError:
			mExist = 0

	if mExist==0:
		print("Building Model")
		model = Sequential()
		model.add(GRU(l1_size,go_backwards=False,return_sequences=True,activation=l1_activation,input_shape=(j, k)))
		model.add(Dropout(dropout))
		if l2_size>0 :
			model.add(GRU(l2_size,go_backwards=False,return_sequences=True,activation=l2_activation))
			model.add(Dropout(dropout))
		if l3_size>0:
			model.add(GRU(l3_size,go_backwards=False,return_sequences=True,activation=l3_activation))
			model.add(Dropout(dropout))
		# if l4_size>0:
		# 	model.add(Dense(l4_size))
		# 	model.add(Activation(l4_activation))
		# if l5_size>0:
		# 	model.add(Dense(l5_size))
		# 	model.add(Activation(l5_activation))
		
		model.add(Flatten())
		model.add(Dense(1))

		tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)

		model.compile(loss=loss,
		              optimizer=optimizer,
		              metrics=metrics)
		history = model.fit(train_x, train_y,
		                    batch_size=batch_size,
		                    epochs=epochs,
		                    verbose=0,
		                    validation_split=validation_split,
		                    callbacks=[tbCallBack],
		                    shuffle=False)
		model.save(modelName)
		# score = model.evaluate(test_x, test_y,
		#                        batch_size=batch_size, verbose=0)
		# print('Test score:', score[0])
		# print('Test accuracy:', score[1])


	model.summary()
	i,j,k = test_x.shape

	t = numpy.arange(0,i,1)
	fig, ax = plt.subplots()
	ax.plot(t, test_y,'b-')

	predSize = 100
	PredStepsinF = 50
	prediction = numpy.zeros((i,PredStepsinF))
	for li in range(predSize,i-predSize,predSize):
		print(test_x[li-predSize:li-1,:,:].shape)
		print(test_y[li-predSize:li-1].shape)
		history = model.fit(test_x[li-predSize:li-1,:,:], test_y[li-predSize:li-1],
	                    batch_size=batch_size,
	                    epochs=3,
	                    verbose=0,
	                    validation_split=validation_split,
	                    shuffle=False)
		a = model.predict(test_x[li-PredStepsinF:li,:,:])
		# print(a)
		prediction[li] = a[:,0]
		t = numpy.arange(li,li+PredStepsinF,1)
		ax.plot(t,a,'g*')



	# s1 = test_y[:-1]
	# # s2 = test_y[:,1]
	# # s3 = test_y[:,2]
	# sp1 = prediction[:-1]
	# # sp2 = prediction[:-1,1]
	# # sp3 = prediction[:-1,2]
	# i = len(s1)

	# ax.plot(t, s1,'g-',t, s2,'b-',t, s3,'r-',t,sp1,'g*',t,sp2,'b*',t,sp3,'r*')
	# ax.plot(t, s1,'b-',t,sp1,'g-')
	ax.grid()
	plt.show()
	plt.close("all")

	# numpy.savetxt("data/prediction.csv",numpy.concatenate((prediction,test_y),axis=1), delimiter=",")
	# numpy.savetxt("data/prediction.csv",prediction, delimiter=",")
	# numpy.savetxt("data/test_y.csv",test_y, delimiter=",")
	

Boleslaw()