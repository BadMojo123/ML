# Jadwiga - Python client

# IMPORT zmq library
import zmq
import time
import numpy as np
import keras.models
from sklearn.externals import joblib
from myconfig import *
import Z_Mind

#DATA|SYMBOL|TIMEFRAME|start_pos|data count to copy

def Rycheza_Client():
	OpenOrders = 0
	CurentOrder="0"	
	MLS = 0.00001
	AdvisorDNN_M1 = keras.models.load_model(AdvisorDNN_M1_Name)
	AdvisorDNN_M1Classifier = joblib.load(AdvisorDNN_M1_ClassifierName)
	AdvisorDNN_M5 = keras.models.load_model(AdvisorDNN_M5_Name)
	AdvisorDNN_M5Classifier = joblib.load(AdvisorDNN_M5_ClassifierName)
	Recomendations = np.zeros((8))
	# 0-8 next bar (date, ohlc) 9-12 M1,13-16 M5
	reqSocket,pullSocket = InitComunication()
	EachMinDone = 0
	TickData = np.zeros((2,2))
	normData = np.zeros((1,18))

	print("$$$$$$$$ INITIALIZATION COMPLETE  $$$$$$$$")
# Recomendations [1m pick, 1m pick donw, 1m trend(1 :up, -1 :down)],0,5m pick, 5m pick donw, 5m trend(1 :up, -1 :down)],0
	while True:
		time.sleep(MLS)
		msg = remote_pull()
		if(msg!='NA'):
			command,data, LastBarTime = InterpretMessage(msg)
			if(command=='DATAEURUSD5'):
				print('Advisor 5M')
				normData = Z_Mind.Norm(data)
				prediction = AdvisorDNN_M5.predict(np.array(normData).reshape((1,18)))
				Recomendations[4] = prediction[0,0]
				Recomendations[5] = prediction[0,2]
				Recomendations[6]=AdvisorDNN_M5Classifier.predict(prediction[0,(False,True,False,True)].reshape(1, -1))				


			if(command=='DATAEURUSD1'):
				print('Advisor 1M')
				normData = Z_Mind.Norm(data)
				prediction = AdvisorDNN_M1.predict(np.array(normData).reshape((1,18)))
				Recomendations[0] = prediction[0,0]
				Recomendations[1] = prediction[0,2]
				Recomendations[2]=AdvisorDNN_M1Classifier.predict(prediction[0,(False,True,False,True)].reshape(1, -1))	

				OpenOrders,CurentOrder = ExecuteOrders(OpenOrders,CurentOrder,Recomendations)

				# save data for future learnig
				summary=np.concatenate((Recomendations,data[63,:]),axis=0)
				print(','.join(map(str, summary.astype(str))))
				one_raw = ','.join(map(str, summary.astype(str)))
				print(LastBarTime)
				one_raw=one_raw+','+str(LastBarTime)+"\n"
				file = open('data/Recomendations.csv','a')
				file.write(one_raw)
				file.close()									

def InterpretMessage(msg):
	if(msg=='NA'):
		return 'NA',0

	msgArray = msg.split('|')

	if msgArray[0]=='TICK':
		print('TICK data recived')
		return 'TICK',[float(msgArray[1]),float(msgArray[2])],'NA'

	# Historica data recived
	if msgArray[0]=='DATA': 
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
		DayNpArray= np.zeros((i))
		WeekDayArray= np.zeros((i))
		for li in range(i):
			temp = TimeArray[li].split(':')
			HourArray[li] = temp[0]
			MinArray[li] = float(temp[1])
			temp = DayArray[li].split('.')[2]
			DayNpArray[li] = temp
			WeekDayArray[li] = float(temp)

		FullData = np.zeros((i,9))
		FullData[:,0] = DayNpArray
		FullData[:,1] = WeekDayArray
		FullData[:,2] = HourArray
		FullData[:,3] = MinArray
		FullData[:,4] = SizeArray
		FullData[:,5] = HighArray 
		FullData[:,6] = LowArray 
		FullData[:,7] = CloseArray 
		FullData[:,8] = VolumeArray
	
		return 'DATA'+msgArray[1]+msgArray[2],FullData[:i-1],str(TimeArray[i-2])+','+str(DayArray[i-2])
	# Unknown command
	print('[RYCHEZA] Uknown message:')
	return msg,0,'NA'

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
	return reqSocket,pullSocket

def ExecuteOrders(OpenOrders,CurentOrder,Recomendations):
	print(OpenOrders)
	print(CurentOrder)
	if(OpenOrders>0):
		if(CurentOrder==BuyOrder):
			if(Recomendations[2]!=1 and Recomendations[6]!=1):
				msg = remote_send(eurusd_close_orders)
				OpenOrders=OpenOrders-1
				CurentOrder='0'
				# log(msg)

		elif(CurentOrder==SellOrder):
			if(Recomendations[2]!=-1 and Recomendations[6]!=-1):
					msg = remote_send(eurusd_close_orders)
					OpenOrders=OpenOrders-1
					CurentOrder='0'
					# log(msg)

	if(OpenOrders==0):
		if(Recomendations[2]==1 and Recomendations[6]==1 and Recomendations[0]==1 and Recomendations[4]==1):
			msg = remote_send(eurusd_buy_order)
			OpenOrders=OpenOrders+1
			CurentOrder=BuyOrder
			# log(msg)
			return OpenOrders,CurentOrder
		elif(Recomendations[2]==-1 and Recomendations[6]==-1 and Recomendations[1]==1 and Recomendations[5]==1):
			msg = remote_send(eurusd_sell_order)
			OpenOrders=OpenOrders+1
			CurentOrder=SellOrder
			# log(msg)
			return OpenOrders,CurentOrder

	return OpenOrders,CurentOrder


# Function to send commands to ZeroMQ MT4 EA
def remote_send(data):
	socket=reqSocket
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