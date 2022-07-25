import time

import joblib
import keras.models
import matplotlib.pyplot as plt
import numpy
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.neighbors import KNeighborsClassifier

import Z_Mind


def MieszkoI_learn(
        testSet=0.9,
        l1_size=500,
        l2_size=500,
        l3_size=500,
        l4_size=500,
        l5_size=500,
        l1_activation='relu',
        l2_activation='relu',
        l3_activation='relu',
        l4_activation='relu',
        l5_activation='relu',
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'],
        epochs=100,
        validation_split=0.001,
        batch_size=100,
        modelName="NA",
        instrument='EURUSD1M'
):
    DATA_FILE_NAME = 'data/' + instrument + '.csv'
    rand = time.time()
    Data = Z_Mind.LoadFile(DATA_FILE_NAME)
    Data, Labels, NormData = Z_Mind.AddLabels(Data)

    FileName = "data/NormData.csv"
    numpy.savetxt(FileName, NormData, delimiter=",")

    i, j = NormData.shape
    testSet = int(i * testSet)
    train_x, train_y, test_x, test_y = Z_Mind.SelectData(testSet, NormData, Labels)
    i, j = train_x.shape

    if (modelName == "NA"):
        model = Sequential()
        model.add(Dense(l1_size, input_dim=j))
        model.add(Activation(l1_activation))
        if l2_size > 0:
            model.add(Dense(l2_size))
            model.add(Activation(l2_activation))
        if l3_size > 0:
            model.add(Dense(l3_size))
            model.add(Activation(l3_activation))
        if l4_size > 0:
            model.add(Dense(l4_size))
            model.add(Activation(l4_activation))
        if l5_size > 0:
            model.add(Dense(l5_size))
            model.add(Activation(l5_activation))
        model.add(Dense(4))

        # tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                patience=6, verbose=1)
        callback_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                               factor=0.01,
                                                               min_lr=1e-4,
                                                               patience=5,
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
                            verbose=1,
                            validation_split=validation_split,
                            callbacks=callbacks)


    else:
        model = keras.models.load_model(modelName)

    score = model.evaluate(test_x, test_y,
                           batch_size=batch_size, verbose=0)
    prediction = model.predict(test_x)

    fig, ax = plt.subplots()
    b = test_y[:, 1] > 0
    ax.scatter(prediction[b, 1], prediction[b, 3])
    c = test_y[:, 3] > 0
    ax.scatter(prediction[c, 1], prediction[c, 3])
    ax.grid()
    fig.savefig("plots/hist.png")
    # plt.show()

    # classifier =  SVC(kernel="linear")
    classifier = KNeighborsClassifier(n_neighbors=5)
    UpDown = prediction[:, numpy.array([1, 3])]
    # print(UpDown)
    Labels = numpy.sum([test_y[:, 1], -test_y[:, 3]], axis=0)
    print(Labels)
    classifier.fit(UpDown, Labels)
    Result_final = classifier.predict(prediction[:, numpy.array([1, 3])])

    # save for testing
    FileName = "data/Prediction.csv"
    numpy.savetxt(FileName, numpy.concatenate((test_y, Result_final[:, None]), axis=1), delimiter=",")

    # print(classification_report(test_y, Result))

    # AccountBalance, trans =  Z_Mind.calculateProfit1(test_x, Result_final)
    # AccountBalance2, trans2 =  Z_Mind.calculateProfit2(test_x, Result_final)

    if (modelName == "NA"):
        modelName = 'klony/' + instrument + '/Mieszko' + str(rand) + '.h5'
        model.save(modelName)
        ClassifierName = 'klony/' + instrument + '/GraphMieszko' + str(rand) + '.h5'
        joblib.dump(classifier, ClassifierName)

    predictedPercent = numpy.sum([Result_final, Labels], axis=0)
    predictedPercent = predictedPercent / 2
    a = predictedPercent != 0

    result = (len(predictedPercent[a])) / (len(predictedPercent))
    print(result)

    summary = str(result) + ","
    # summary= str(AccountBalance)+","+str(AccountBalance/trans)+","+str(trans)+","
    # summary=summary+ str(AccountBalance2)+","+str(AccountBalance2/trans2)+","+str(trans2)+","
    summary = summary + str(l1_size) + "," + str(l1_activation) + "," + str(l2_size) + "," + str(
        l2_activation) + "," + str(l3_size) + "," + str(l3_activation) + "," + str(l4_size) + "," + str(
        l4_activation) + "," + str(l5_size) + "," + str(l5_activation)
    summary = summary + "," + str(score[0]) + "," + str(score[1]) + ","
    summary = summary + str(loss) + "," + str(optimizer) + "," + str(metrics) + "," + str(modelName) + "\n"

    print(summary)
    file = open('data/MieszkoRes.csv', 'a')
    file.write(summary)
    file.close()

    #  # ploting for data verification
    # Z_Mind.plotResult(Data, Result,testSet)
    return


def MieszkoI_think(DATA, modelName='klony/MieszkoI_1m.h5'):
    model.load(modelName)
    Data = prepareData(DATA)
    prediction = model.predict(Data)
    return
