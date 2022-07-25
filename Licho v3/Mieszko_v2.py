import time

import matplotlib.pyplot as plt
import numpy
import tensorflow
from sklearn.utils import shuffle

import MieszkoTransformer

NUM_ACTIONS = 3
DUMMY_ACTION, DUMMY_VALUE = numpy.zeros((NUM_ACTIONS)), numpy.zeros((1, 1))


class Mieszko:
    def __init__(self, Rycheza, ID=0):
        self.ID = ID
        self.Rycheza = Rycheza
        self.model = MieszkoTransformer.build_model()

        self.EPOCHS = 1  # how many passes through our data
        if (ID != 0):
            self.modelName = 'klony/Mieszko' + str(self.ID) + '.h5'
            self.model.load_weights(self.modelName)
            print('Mieszko model loaded')

    def TrainMieszko(self):
        # self.LichoID=5

        RNNLearningSet = numpy.zeros(
            (100000, self.Rycheza.TransformerHist, self.Rycheza.InstrumentCount, self.Rycheza.IndicatorCount))
        RNNLabels = numpy.zeros((100000, self.Rycheza.InstrumentCount, self.Rycheza.IndicatorCount))
        done = 0
        i = 0
        _, _, _ = self.Rycheza.FillRawData(DUMMY_ACTION)

        while done != 2:
            RNNLearningSet[i, :, :, :] = self.Rycheza.Indicators
            _, done, _ = self.Rycheza.FillRawData(DUMMY_ACTION)
            self.Rycheza.PrepTaransformerObservation(Predict=0)
            RNNLabels[i, :, :] = self.Rycheza.Indicators[0, :, :]
            # print(RNNLabels[i,:,:])
            i += 1

        i = i - 10

        FullData = numpy.asarray(RNNLearningSet)
        RNNLabels = numpy.asarray(RNNLabels)
        FullData = FullData[self.Rycheza.TransformerHist + 10:i, :, :, :]
        RNNLabels = RNNLabels[self.Rycheza.TransformerHist + 10:i, :, :]

        i, j, k, l = FullData.shape
        # print('Data shape: '+str(FullData.shape)+'Labels shape: '+str(RNNLabels.shape))
        FullData = numpy.reshape(FullData, (i, j, k * l))
        RNNLabels = numpy.reshape(RNNLabels, (i, k * l))
        print('Data shape: ' + str(FullData.shape) + 'Labels shape: ' + str(RNNLabels.shape))
        # numpy.savetxt("TempFullDatas.csv", FullData[0, :, :], delimiter=";")
        # numpy.savetxt("TempFullDatas2.csv", FullData[100, :, :], delimiter=";")
        # numpy.savetxt("TempFullDatasLab.csv", RNNLabels, delimiter=";")
        ValidSplit = int(i * 0.8)
        TestSplit = int(i * 0.9)

        X_train = FullData[:ValidSplit, :, :]
        y_train = RNNLabels[:ValidSplit, :]

        X_valid = FullData[ValidSplit:TestSplit, :, :]
        y_valid = RNNLabels[ValidSplit:TestSplit, :]

        X_test = FullData[TestSplit:, :, :]
        y_test = RNNLabels[TestSplit:, :]

        # print(X_train[:,:,0])
        X_train, y_train = shuffle(X_train, y_train)
        if (self.ID == 0):
            self.ID = int(time.time())
            # self.EPOCHS = 2  # how many passes through our data
            BATCH_SIZE = 2024  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
            callback_early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='loss',
                                                                               patience=3, verbose=1)
            callback_reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                              factor=0.1,
                                                                              min_lr=1e-5,
                                                                              patience=3,
                                                                              verbose=1)
            # callback_tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir=".\\AllRuns\\Licho", histogram_freq=1, write_graph=True)
            callbacks = [callback_early_stopping,
                         # callback_tensorboard,
                         callback_reduce_lr]
            self.model.fit(X_train, y_train,
                           batch_size=BATCH_SIZE,
                           epochs=self.EPOCHS,
                           validation_data=(X_valid, y_valid)
                           # callbacks = callbacks
                           )
            print('Mieszko Model saved. ID = ' + str(self.ID))
            self.modelName = 'klony/Mieszko' + str(self.ID) + '.h5'
            self.model.save_weights(self.modelName)

        predictedY = self.model.predict(X_test)

        print(y_test[:, 0])
        x = range(0, len(predictedY[:, 0]))
        plt.plot(x, predictedY[:, 0])
        plt.plot(x, y_test[:, 0])
        plt.show()
        plt.plot(x, predictedY[:, 1])
        plt.plot(x, y_test[:, 1])
        plt.show()
        plt.plot(x, predictedY[:, 2])
        plt.plot(x, y_test[:, 2])
        plt.show()
        plt.plot(x, predictedY[:, 3])
        plt.plot(x, y_test[:, 3])
        plt.show()

        return
