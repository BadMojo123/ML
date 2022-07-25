import numpy as np
import zmq


class LichoConnector:
    def __init__(self):

        # self.InitComunication()
        self.reqSocket, self.pullSocket = self.InitComunication()

    def InitComunication(self):
        # Create ZMQ Context
        context = zmq.Context()
        # Create REQ Socket
        reqSocket = context.socket(zmq.REQ)
        reqSocket.connect("tcp://127.0.0.1:5555")

        # Create PULL Socket
        pullSocket = context.socket(zmq.PULL)
        pullSocket.connect("tcp://127.0.0.1:5556")
        return reqSocket, pullSocket

    def remote_send(self, data):
        try:
            self.reqSocket.send_string(data)
            msg = self.reqSocket.recv_string()
            print(msg)
        except zmq.Again as e:
            print("Waiting for PUSH from MetaTrader 4..")
        return msg

    # Function to retrieve data from ZeroMQ MT4 EA
    def remote_pull(self):

        try:
            msg = self.pullSocket.recv(flags=zmq.NOBLOCK).decode('UTF-8')
        # print(msg)
        except zmq.Again as e:
            # print(e)
            msg = 'Nothing in queue'
        return msg

    def InterpretMessage(self, msg):
        # return command,instrument,timeSpan,data, LastBarTime
        if (msg == 'NA'):
            return 'NA', 'NA', 'NA', 'NA', 'NA'

        msgArray = msg.split('|')

        # TICK|Ask|Bid
        if msgArray[0] == 'TICK':
            print('TICK data recived')
            return 'TICK', [float(msgArray[1]), float(msgArray[2])], 'NA', 'NA'

        # DATA|EURUSD|minutes|Time;Time|Size;Size|High;High|Low;Low|Close;Close|Vol;Vol|DayTime|TimeOfLastBar|END
        if msgArray[0] == 'DATA':
            print('[RYCHEZA] Data recived: ' + msgArray[1])
            TimeArray = np.asarray(list(reversed(msgArray[3].split(';')[1:]))).astype(np.float)
            SizeArray = np.asarray(list(reversed(msgArray[4].split(';')[1:]))).astype(np.float)
            HighArray = np.asarray(list(reversed(msgArray[5].split(';')[1:]))).astype(np.float)
            LowArray = np.asarray(list(reversed(msgArray[6].split(';')[1:]))).astype(np.float)
            CloseArray = np.asarray(list(reversed(msgArray[7].split(';')[1:]))).astype(np.float)
            VolumeArray = np.asarray(list(reversed(msgArray[8].split(';')[1:]))).astype(np.float)
            i = len(TimeArray)

            FullData = np.zeros((i, 5))
            FullData[:, 0] = TimeArray
            FullData[:, 1] = HighArray
            FullData[:, 2] = LowArray
            FullData[:, 3] = CloseArray
            FullData[:, 4] = VolumeArray

            Time = msgArray[10]
            return 'DATA', msgArray[1], msgArray[2], FullData, int(Time)

        # EQUITY|10000.00000|TimeOfLastBar|END
        if msgArray[0] == 'EQUITY':
            print('[RYCHEZA] Equity recived')
            Time = msgArray[2]
            return 'EQUITY', 'NA', 'NA', float(msgArray[1]), int(Time)

        # Unknown command
        print('[RYCHEZA] Uknown message: Is it error?')
        print(msg)
        return msg, 0, 'NA', 'NA', 'NA'
