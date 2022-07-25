# Jadwiga - Python client

import sys
import time

import numpy as np
# IMPORT zmq library
import zmq

Test = 'YES'
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
MLS = 0.01

EURUSDMIN = 1.00600
EURUSDMAX = 1.5
EURUSDVOLMIN = 0
EURUSDVOLMAX = 500
EURUSDMAXSIZE_1M = 0.0003


def Rycheza_Client():
    global TickData, reqSocket, pullSocket, NewOrder, NewOrderPrice
    CurentOrder = "0"

    reqSocket, pullSocket = InitComunication()
    EachMinDone = 0
    TickData = np.zeros((2, 2))
    hi = 0

    while True:
        CurentOrder = CheckForNewData(TickData, CurentOrder)
        time.sleep(MLS)
        remote_send(get_hist_1m)


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
            WeekDayArray[li] = float(temp)

        # print(DayNpArray)
        FullData = np.zeros((i, 9))
        FullData[:, 0] = DayNpArray
        FullData[:, 1] = WeekDayArray
        FullData[:, 2] = HourArray
        FullData[:, 3] = MinArray
        FullData[:, 4] = SizeArray
        FullData[:, 5] = HighArray
        FullData[:, 6] = LowArray
        FullData[:, 7] = CloseArray
        FullData[:, 8] = VolumeArray

        FileName = "data/EURUSD120M.csv"
        np.savetxt(FileName, FullData, delimiter=",")
        sys.exit("FORCED END - file saved")

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


# Run
Rycheza_Client()
