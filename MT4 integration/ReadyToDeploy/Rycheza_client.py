# Jadwiga - Python client

import datetime
import sys
import time

import keras.models
import numpy as np
# IMPORT zmq library
import zmq
from sklearn.externals import joblib

# Sample Commands for ZeroMQ MT4 EA
# eurusd_buy_order = "TRADE|OPEN|0|EURUSD|0|50|50|Python-to-MT4|price"
eurusd_buy_order = "TRADE|OPEN|0|EURUSD|0|0|0|Python-to-MT4"  # |price
eurusd_sell_order = "TRADE|OPEN|1|EURUSD|0|0|0|Python-to-MT4"
eurusd_close_orders = "TRADE|CLOSE|0|EURUSD|0|50|50|Python-to-MT4"
get_rates = "RATES|EURUSD"
BuyOrder = 'BUY'
SellOrder = 'SELL'
CloseOrder = 'CLOSE'
NewOrder = 0
NewOrderPrice = 0
MLS = 0.1
reqHist = 100
# DATA|SYMBOL|TIMEFRAME|start_pos|data count to copy
get_hist_1m = "DATA|EURUSD|5|0|" + str(reqHist)
EURUSDMIN = 1.00600
EURUSDMAX = 1.5
EURUSDVOLMIN = 0
EURUSDVOLMAX = 500
EURUSDMAXSIZE_1M = 0.0003
modelName = 'klony/Mieszko1527157255.7840197.h5'
ClassifierName = 'klony/MieszkoGraph1527157255.7840197.h5'
M1_Advisor = keras.models.load_model(modelName)
M1_Classifier = joblib.load(ClassifierName)


def Rycheza_Client():
    global TickData, reqSocket, pullSocket, NewOrder, NewOrderPrice
    CurentOrder = "0"

    reqSocket, pullSocket = InitComunication()
    EachMinDone = 0
    TickData = np.zeros((2, 2))
    hi = 0

    while True:
        # ExecuteOrders() #check pending, check price, make order
        # AskForOrderList()
        CurentOrder = CheckForNewData(TickData, CurentOrder)
        # GetRecomendation()
        # TrainAdvisors()
        time.sleep(MLS)

        # # Send RATES command to ZeroMQ MT4 EA
        # remote_send(reqSocket, get_rates)

        # print("Im alive2")
        # Send CLOSE EURUSD command to ZeroMQ MT4 EA. You'll need to append the
        # trade's ORDER ID to the end, as below for example:
        # remote_send(reqSocket, eurusd_closebuy_order + "|" + "12345678")

        now = datetime.datetime.now().second
        # Clear flags
        if (now > 30 and EachMinDone == 1):
            EachMinDone = 0

        # each 1M
        if (now < 30 and EachMinDone == 0):  # each minute save hist, check if close, learn advisors
            EachMinDone = 1
            print(datetime.datetime.now())
            print("[Rycheza] M1 advisor processing")
            remote_send(get_hist_1m)

        # early finish - for testing
        # hi=hi+1
        # if hi>10:
        # 	return


def InitComunication():
    # Create ZMQ Context
    context = zmq.Context()

    global reqSocket, pullSocket
    # Create REQ Socket
    reqSocket = context.socket(zmq.REQ)
    reqSocket.connect("tcp://127.0.0.1:5555")

    # Create PULL Socket
    pullSocket = context.socket(zmq.PULL)
    pullSocket.connect("tcp://127.0.0.1:5556")
    return reqSocket, pullSocket


def ExecuteOrders():
    global reqSocket, pullSocket, NewOrder, NewOrderPrice
    if (NewOrder == BuyOrder):
        # CheckPrice(NewOrderPrice)
        msg = remote_send(eurusd_buy_order)
        if (msg != '1'):
            # CheckPrice(NewOrderPrice)
            msg = remote_send(eurusd_buy_order)
        # log(msg)
    if (NewOrder == SellOrder):
        # CheckPrice(NewOrderPrice)
        msg = remote_send(eurusd_sell_order)
        if (msg != '1'):
            # CheckPrice(NewOrderPrice)
            msg = remote_send(eurusd_sell_order)
        # log(msg)
    if (NewOrder == CloseOrder):
        # CheckPrice(NewOrderPrice)
        msg = remote_send(eurusd_close_orders)
        if (msg != '1'):
            # CheckPrice(NewOrderPrice)
            msg = remote_send(eurusd_close_orders)
        # log(msg)

    NewOrder = None
    NewOrderPrice = None
    return True


def CheckForNewData(TickData, order):
    msg = remote_pull()

    if (msg == 'NA'):
        return order

    msgArray = msg.split('|')

    if msgArray[0] == 'TICK':
        print('TICK data recived')
        TickData = np.append(TickData, [float(msgArray[1]), float(msgArray[2])])

    # Historica data recived
    if msgArray[0] == 'DATA':
        print('[RYCHEZA] Requested data recived')
        # Day(0)        weekday(1)      Hour(2)  minute(3)   O(4)   H(5)    L(6)    C(7)    Volume(8)
        TimeArray = np.asarray(list(reversed(msgArray[3].split(';')[1:])))
        SizeArray = np.asarray(list(reversed(msgArray[4].split(';')[1:]))).astype(np.float)
        HighArray = np.asarray(list(reversed(msgArray[5].split(';')[1:]))).astype(np.float)
        LowArray = np.asarray(list(reversed(msgArray[6].split(';')[1:]))).astype(np.float)
        CloseArray = np.asarray(list(reversed(msgArray[7].split(';')[1:]))).astype(np.float)
        VolumeArray = np.asarray(list(reversed(msgArray[8].split(';')[1:]))).astype(np.float)
        DayArray = np.asarray(list(reversed(msgArray[9].split(';')[1:])))
        i = len(TimeArray)
        HourArray = np.zeros((i))
        MinArray = np.zeros((i))
        DayNpArray = np.zeros((i))
        WeekDayArray = np.zeros((i))
        for li in range(i):
            temp = TimeArray[li].split(':')
            HourArray[li] = temp[0]
            MinArray[li] = float(temp[1])
            temp = DayArray[li].split('.')[2]
            DayNpArray[li] = temp
            WeekDayArray[li] = float(temp) % 7

        # print(DayNpArray)
        FullData = np.zeros((i, 9))
        FullData[:, 0] = DayNpArray / 31
        FullData[:, 1] = WeekDayArray / 7
        FullData[:, 2] = HourArray / 24
        FullData[:, 3] = MinArray / 60
        FullData[:, 4] = SizeArray / EURUSDMAXSIZE_1M
        FullData[:, 5] = (HighArray - EURUSDMIN) / (EURUSDMAX - EURUSDMIN)
        FullData[:, 6] = (LowArray - EURUSDMIN) / (EURUSDMAX - EURUSDMIN)
        FullData[:, 7] = (CloseArray - EURUSDMIN) / (EURUSDMAX - EURUSDMIN)
        FullData[:, 8] = (VolumeArray - EURUSDVOLMIN) / (EURUSDVOLMAX - EURUSDVOLMIN)

        FullData = PrepareForPrediction(FullData)

        prediction = M1_Advisor.predict(FullData.reshape((1, 409)))
        print(prediction)
        prediction = M1_Classifier.predict(prediction)
        print(prediction)
        prediction = prediction[0]
        print("CurentOrder")
        print(order)
        if (prediction[0] == 1 and order != "BUY"):
            order = "BUY"
            print('[RYCHEZA] Sending CLOSE SELL order')
            remote_send(eurusd_close_orders)
            print('[RYCHEZA] Sending BUY order')
            remote_send(eurusd_buy_order)

        if (prediction[2] == 1 and order != "SELL"):
            order = "SELL"
            print('[RYCHEZA] Sending CLOSE BUY order')
            remote_send(eurusd_close_orders)
            print('[RYCHEZA] Sending SELL order')
            remote_send(eurusd_sell_order)

        # elif(prediction[1]==1 or prediction[3]==1):
        # 	print('[RYCHEZA] Sending CLOSE ALL order')
        # 	remote_send(eurusd_close_orders)

        return order

    # Unknown command
    else:
        print('[RYCHEZA] Uknown message:')
        print(msg)
        return order


# Function to send commands to ZeroMQ MT4 EA
def remote_send(data):
    socket = reqSocket
    try:
        socket.send_string(data)
        msg = socket.recv_string()
        print(msg)
    except zmq.Again as e:
        print("Waiting for PUSH from MetaTrader 4..")

    return msg


# Function to retrieve data from ZeroMQ MT4 EA
def remote_pull():
    try:
        msg = pullSocket.recv(flags=zmq.NOBLOCK).decode('UTF-8')
    # print(msg)
    except zmq.Again as e:
        # print(e)
        msg = 'NA'
    return msg


def PrepareForPrediction(DATA, Hist=80, samples=1):
    # one raw of data for prediction
    if (samples == 1):
        Hist = Hist + 1
        i, j = DATA.shape
        NormData = DATA[i - Hist:i - 1, 4:9]
        NormData = NormData.reshape((1, 5 * (Hist - 1)))
        NormData = np.concatenate((DATA[i - 1, :], NormData[0, :]), axis=0)
        return NormData
    else:
        # save file for futher training
        # Day(0)        weekday(1)      Hour(2)  minute(3)   O(4)   H(5)    L(6)    C(7)    Volume(8)	C[n+1](9)	 C[n+2](10)
        i, j = DATA.shape
        DATA = np.concatenate((DATA[0:i - 2], DATA[1:i - 1, 7].reshape((i - 2, 1)), DATA[2:i, 7].reshape((i - 2, 1))),
                              axis=1)

        i, j = DATA.shape
        FullData = np.zeros((i - Hist, 5 * Hist + j))
        for li in range(Hist, i):
            NormData = DATA[li - Hist:li, 4:9]
            NormData = NormData.reshape((1, 5 * (Hist)))
            NormData = np.concatenate((NormData[0, :], DATA[li, :]), axis=0)
            FullData[li - Hist, :] = NormData

        FileName = "data/EURUSD5M.csv"
        np.savetxt(FileName, FullData, delimiter=",")
        sys.exit("FORCED END - file saved")


# Run
Rycheza_Client()
