import numpy
from keras.models import Sequential
from keras.layers import Dense
import datetime
from sklearn.preprocessing import MinMaxScaler



def sigmoid(x):
    # return (1 / (1 + numpy.exp(-x)))
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x[:, numpy.newaxis]) 
    return x[:,0]
    # return x

def LoadFile(FILE_NAME):
    RawData = []
    with open(FILE_NAME) as f:
            for n, line in enumerate(f):
                    t_line = line.split(',')
                    t_date = t_line[0].split('.')
                    date = datetime.date(int(t_date[0]),int(t_date[1]),int(t_date[2]))
                    time = t_line[1].split(':')
                    volume = t_line[6].rstrip()
                    # Day(0)        weekday(1)      Hour(2)  minute(3)   O(4)   H(5)    L(6)    C(7)    Volume(8)
                    RawData.append([float(t_date[2]),float(date.weekday()),float(time[0]),float(time[1]),float(t_line[2]),float(t_line[3]),float(t_line[4]),float(t_line[5]),float(volume)])

    return numpy.array(RawData)


def AddNormData(FullData):
    i,j = FullData.shape
    NormData =numpy.zeros((i,j))
    ## normalize Day
    NormData[:,0] = FullData[:,0]/31
    ## normalize Day
    NormData[:,1] = FullData[:,1]/7
    ## normalize Hour
    NormData[:,2] = FullData[:,2]/24
    ## normalize minute
    NormData[:,3] = FullData[:,3]/60
    ## normalize O - fuck Open - add bar sie
    NormData[:,4] =sigmoid(FullData[:,5] - FullData[:,6])
    ## normalize H
    NormData[:,5] =sigmoid(FullData[:,5])# - FullData[:,7])
    ## normalize L
    NormData[:,6] =sigmoid(FullData[:,6])# - FullData[:,7])
    ## normalize C
    NormData[:,7] =sigmoid(FullData[:,7])# - FullData[:i-1,7])
    ## normalize Volume
    NormData[:,8] = sigmoid(FullData[:,8])
    
    return NormData

def AddIndicatorsAndHist(NormData,histReq = 100):
    # indicator paramiters
    i,j = NormData.shape
    TrainingData =numpy.zeros((i,j,histReq))
    # Labels =numpy.zeros((i,j-6))
    Labels =numpy.zeros((i))

    for li in range(histReq,i-1):
        # #Add historical data HistDataAded remining
        # Labels[li,:]=NormData[li+1,5:8]
        
        Labels[li]=NormData[li+1,7]
        for lj in range(j):
            TrainingData[li,lj,:] = NormData[li-histReq:li,lj]

    return Labels[:-1],TrainingData[:-1,:,:]


def SelectData(x_full,Labels,testsize=0.8):
    i,j,k = x_full.shape
    testSet = int(testsize *i)
    Train_x = x_full[0:testSet,:,:]
    Train_y =Labels[0:testSet]
    Test_x= x_full[testSet:,:,:]
    Test_y=Labels[testSet:]
    return Train_x,Train_y,Test_x,Test_y
