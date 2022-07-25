import numpy
import time

import keras.models
import matplotlib.pyplot as plt
import numpy
from keras.layers import Dense, Dropout, GRU, Flatten
from keras.models import Sequential, load_model

EURUSDMIN = 1.17631
EURUSDMAX = 1.24138


def sigmoid(x):
    X_std = (x - EURUSDMIN) / (EURUSDMAX - EURUSDMIN)


    return X_std


def LoadFile(FILE_NAME):
    RawData = []
    with open(FILE_NAME) as f:
        for n, line in enumerate(f):
            t_line = line.split(',')
            t_date = t_line[0].split('.')
            date = datetime.date(int(t_date[0]), int(t_date[1]), int(t_date[2]))
            time = t_line[1].split(':')
            volume = t_line[6].rstrip()
            # Day(0)        weekday(1)      Hour(2)  minute(3)   O(4)   H(5)    L(6)    C(7)    Volume(8)
            RawData.append([float(t_date[2]), float(date.weekday()), float(time[0]), float(time[1]), float(t_line[2]),
                            float(t_line[3]), float(t_line[4]), float(t_line[5]), float(volume)])

    return numpy.array(RawData)


def AddNormData(FullData):
    i, j = FullData.shape
    NormData = numpy.zeros((i, j))
    ## normalize Day
    NormData[:, 0] = FullData[:, 0]
    ## normalize Day
    NormData[:, 1] = FullData[:, 1]
    ## normalize Hour
    NormData[:, 2] = FullData[:, 2]
    ## normalize minute
    NormData[:, 3] = FullData[:, 3]
    ## normalize O - fuck Open - add bar sie
    NormData[:, 4] = sigmoid(FullData[:, 5] - FullData[:, 6])
    ## normalize H
    NormData[:, 5] = sigmoid(FullData[:, 5])  # - FullData[:,7])
    ## normalize L
    NormData[:, 6] = sigmoid(FullData[:, 6])  # - FullData[:,7])
    ## normalize C
    NormData[:, 7] = sigmoid(FullData[:, 7])  # - FullData[:i-1,7])
    ## normalize Volume
    NormData[:, 8] = sigmoid(FullData[:, 8])

    return NormData


def AddIndicatorsAndHist(NormData, histReq=100):
    # indicator paramiters
    i, j = NormData.shape
    TrainingData = numpy.zeros((i, j, histReq))
    Labels = numpy.zeros((i, 4))
    pick = 5
    in_feature = 1
    fee = 1

    for li in range(histReq, i - 1):
        # #Add historical data HistDataAded remining
        # Labels[li,:]=NormData[li+1,5:8]

        Labels[li] = NormData[li + 1, 7]

        if NormData[li + in_feature, 7] > NormData[li, 7] + pick:
            Labels[li] = [1, 0, 0, 0]
        elif NormData[li + in_feature, 7] >= NormData[li, 7] + fee:
            Labels[li] = [0, 1, 0, 0]
        elif NormData[li + in_feature, 7] < NormData[li, 7] - fee:
            Labels[li] = [0, 0, 1, 0]
        elif NormData[li + in_feature, 7] < NormData[li, 7] - pick:
            Labels[li] = [0, 0, 0, 1]

        for lj in range(j):
            TrainingData[li, lj, :] = NormData[li - histReq:li, lj]

    return Labels[:-1], TrainingData[:-1, :, :]


def SelectData(x_full, Labels, testsize=0.8):
    i, j, k = x_full.shape
    testSet = int(testsize * i)
    Train_x = x_full[0:testSet, :, :]
    Train_y = Labels[0:testSet]
    Test_x = x_full[testSet:, :, :]
    Test_y = Labels[testSet:]
    return Train_x, Train_y, Test_x, Test_y


def Boleslaw(
        testSet=0.95,
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
        FILE_NAME='histData/GER30Cash1.csv',
        dropout=0.5,
        histReq=100
):
    modelName = 'klony/BoleslawLabCat.h5'
    rand = time.time()
    Data = LoadFile(FILE_NAME)

    NormData = AddNormData(Data)

    Labels, Data = AddIndicatorsAndHist(NormData, histReq)
    train_x, train_y, test_x, test_y = SelectData(Data, Labels, testSet)
    i, j, k = train_x.shape
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

    if mExist == 0:
        print("Building Model")
        model = Sequential()
        model.add(GRU(l1_size, go_backwards=False, return_sequences=True, activation=l1_activation, input_shape=(j, k)))
        model.add(Dropout(dropout))
        if l2_size > 0:
            model.add(GRU(l2_size, go_backwards=False, return_sequences=True, activation=l2_activation))
            model.add(Dropout(dropout))
        if l3_size > 0:
            model.add(GRU(l3_size, go_backwards=False, return_sequences=True, activation=l3_activation))
            model.add(Dropout(dropout))
        # if l4_size>0:
        # 	model.add(Dense(l4_size))
        # 	model.add(Activation(l4_activation))
        # if l5_size>0:
        # 	model.add(Dense(l5_size))
        # 	model.add(Activation(l5_activation))

        model.add(Flatten())
        model.add(Dense(1))

        tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True,
                                                 write_images=True)

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
    i, j, k = test_x.shape

    # t = numpy.arange(0,i,1)
    # fig, ax = plt.subplots()
    # ax.plot(t, test_y,'b-')

    # predSize = 100
    # PredStepsinF = 50
    # prediction = numpy.zeros((i,PredStepsinF))
    # for li in range(predSize,i-predSize,predSize):
    # 	print(test_x[li-predSize:li-1,:,:].shape)
    # 	print(test_y[li-predSize:li-1].shape)
    # 	history = model.fit(test_x[li-predSize:li-1,:,:], test_y[li-predSize:li-1],
    #                     batch_size=batch_size,
    #                     epochs=3,
    #                     verbose=0,
    #                     validation_split=validation_split,
    #                     shuffle=False)
    # 	a = model.predict(test_x[li-PredStepsinF:li,:,:])
    # 	# print(a)
    # 	prediction[li] = a[:,0]
    # 	t = numpy.arange(li,li+PredStepsinF,1)
    # 	ax.plot(t,a,'g*')

    # s1 = test_y[:-1]
    # # s2 = test_y[:,1]
    # # s3 = test_y[:,2]
    # sp1 = prediction[:-1]
    # # sp2 = prediction[:-1,1]
    # # sp3 = prediction[:-1,2]
    # i = len(s1)

    ax.plot(t, s1, 'g-', t, s2, 'b-', t, s3, 'r-', t, sp1, 'g*', t, sp2, 'b*', t, sp3, 'r*')
    # ax.plot(t, s1,'b-',t,sp1,'g-')
    ax.grid()
    plt.show()
    plt.close("all")


# numpy.savetxt("data/prediction.csv",numpy.concatenate((prediction,test_y),axis=1), delimiter=",")
# numpy.savetxt("data/prediction.csv",prediction, delimiter=",")
# numpy.savetxt("data/test_y.csv",test_y, delimiter=",")


Boleslaw()
