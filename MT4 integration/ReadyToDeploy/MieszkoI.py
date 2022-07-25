import time

import keras.models
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

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
        batch_size=100
):
    rand = time.time()
    FILE_NAME = 'data/EURUSD1M.csv'
    Data = Z_Mind.LoadFile(FILE_NAME)
    Data, Labels = Z_Mind.AddLabels(Data)
    i, j = Data.shape
    testSet = int(i * testSet)
    train_x, train_y, test_x, test_y = Z_Mind.SelectData(testSet, Data, Labels)
    i, j = train_x.shape

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
                                                            patience=5, verbose=1)
    callback_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                           factor=0.01,
                                                           min_lr=1e-4,
                                                           patience=3,
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

    score = model.evaluate(test_x, test_y,
                           batch_size=batch_size, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    prediction = model.predict(test_x)

    MieszkoX = model.predict(train_x)
    classifier = DecisionTreeClassifier()
    classifier.fit(MieszkoX, train_y)
    Result = classifier.predict(prediction)

    # print(confusion_matrix(test_y, Result))
    print(classification_report(test_y, Result))
    # print(Result)

    AccountBalance, trans = Z_Mind.calculateProfit(test_x, Result, testSet)

    modelName = 'klony/Mieszko' + str(rand) + '.h5'
    model.save(modelName)
    ClassifierName = 'klony/MieszkoGraph' + str(rand) + '.h5'
    joblib.dump(classifier, ClassifierName)

    summary = "Mieszko," + str(rand) + "," + str(epochs) + "," + str(l1_size) + "," + str(l1_activation) + "," + str(
        l2_size) + "," + str(l2_activation) + "," + str(l3_size) + "," + str(l3_activation) + "," + str(
        l4_size) + "," + str(l4_activation) + "," + str(l5_size) + "," + str(l5_activation) + "," + str(
        score[0]) + "," + str(score[1]) + "," + str(AccountBalance) + "," + str(AccountBalance / trans) + "," + str(
        loss) + "," + str(optimizer) + "," + str(metrics) + "," + str(validation_split) + "," + str(trans) + "," + str(
        modelName) + "\n"
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
