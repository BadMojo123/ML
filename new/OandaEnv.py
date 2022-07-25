import configparser  # 1
import datetime
import math
import time
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numba as nb
import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import oandapy


class OandaEnv:
    def __init__(self, DownloadMode=0):
        self.mode = 0
        self.DownloadMode = DownloadMode
        self.Instruments = ["SPX500_USD", "EUR_USD", "DE30_EUR"]
        # self.Instruments = ["SPX500_USD", "EUR_USD"]
        # DE30_EUR
        self.Granularity = ['H4', 'M30', 'M5']  # have to be sorted from largest to smallest
        # self.Granularity = ['M30']   # have to be sorted from largest to smallest
        self.lot = [1, 1000, 1]
        # self.Granularity = ['H1','M15']   # have to be sorted from largest to smallest
        # TimeFrames = [1,15,1440]
        self.TimeFramesLen = len(self.Granularity)
        self.InstrumentCount = len(self.Instruments)
        TXTconfig = configparser.ConfigParser()  # 3
        TXTconfig.read('oanda.cfg')  # 4
        self.oandaAPI = oandapy.API(environment='practice', access_token=TXTconfig['oanda']['access_token'])

        self.PastDataForStep = 5
        self.MeanMax = 21
        self.IndicatorsCount = 8
        self.AllIndicatorsLen = self.InstrumentCount * self.IndicatorsCount
        self.OandaFeatures = 10

        self.FullData = [object] * self.TimeFramesLen
        self.TimeData = [object] * self.TimeFramesLen
        self.priceData = [object] * self.TimeFramesLen
        self.TimeIndex = numpy.zeros((self.TimeFramesLen), int)

        # Tragind paramiters
        self.TransactionalInstruments = [1, 1, 1]
        self.action_space = self.InstrumentCount * 2 + 1
        self.CurrentPossitions = numpy.zeros((self.action_space))
        self.maxPositions = 10
        self.fee = 0.01
        self.equity = 0
        self.equityOld = 0

        self.LongPositions = numpy.zeros((self.InstrumentCount, self.maxPositions))
        self.ShortPositions = numpy.zeros((self.InstrumentCount, self.maxPositions))
        self.LastLongID = numpy.zeros((self.InstrumentCount), dtype=numpy.int8) - 1
        self.LastShortID = numpy.zeros((self.InstrumentCount), dtype=numpy.int8) - 1
        self.CurrentPrice = numpy.zeros((self.InstrumentCount,))

    def downloadHistory(self, GI):
        downloadHist = [object] * len(self.Instruments)
        a = date(2017, 1, 3)
        b = date(2017, 12, 20)
        delta = timedelta(days=6)
        days = int((b - a).days / 6) + 1

        for InstI in range(0, len(self.Instruments)):
            downloadHist[InstI] = [object] * days
            d = a
            for i in range(0, days):
                StartDate = d.strftime("%Y-%m-%d")
                d = d + delta
                EndDate = d.strftime("%Y-%m-%d")
                print(StartDate)
                print(EndDate)
                downloadHist[InstI][i] = self.InstrumentDownload(self.Granularity[GI], self.Instruments[InstI],
                                                                 StartDate, EndDate)

                time.sleep(2)

            print("po concacie")
            downloadHist[InstI] = pd.concat(downloadHist[InstI][:], axis=0)
            # downloadHist[InstI].reset_index(drop=True, inplace=True)
            FileName = "data/OandaPdData" + str(self.Granularity[GI]) + "_" + str(self.Instruments[InstI]) + ".csv"
            downloadHist[InstI].to_csv(FileName, sep=';', encoding='utf-8')

        return

    def InstrumentDownload(self, granularity, instrument, StartDate, EndDate):

        data = self.oandaAPI.get_history(instrument=instrument,  # our instrument
                                         start=StartDate,  # start data
                                         end=EndDate,  # end date
                                         granularity=granularity)  # minute bars  # 7

        df = pd.DataFrame(data['candles']).set_index('time')  # 8
        df.index = pd.DatetimeIndex(df.index)  # 9

        return df

    def resetEnv(self):
        print('Reeting Env')
        self.TimeIndex = numpy.zeros((self.TimeFramesLen), int)
        self.TimeIndex[0] = self.PastDataForStep + self.MeanMax
        # self.CurrentPrice = numpy.zeros((self.InstrumentCount,))
        self.ExecuteTrades(numpy.zeros((self.action_space)))  # close all trades

        # self.LongPositions = numpy.zeros((self.InstrumentCount,self.maxPositions))
        # self.ShortPositions = numpy.zeros((self.InstrumentCount,self.maxPositions))
        # self.LastLongID = numpy.zeros((self.InstrumentCount),dtype = numpy.int8) -1
        # self.LastShortID = numpy.zeros((self.InstrumentCount),dtype = numpy.int8) -1

        LastTime = self.TimeData[0][self.TimeIndex[0]]

        for i in range(1, self.TimeFramesLen):
            for j in range(0, len(self.TimeData[i])):
                if (self.TimeData[i][j] >= self.TimeData[0][self.TimeIndex[0]]):
                    self.TimeIndex[i] = j
                    break

        return

    @nb.jit
    def Next(self):
        done = 0
        # TODO max batch <= długości gry. W trakcie batcha nie można przerwać gry
        # TODO uczenie się każdy dzień po koleji

        lastTFI = self.TimeFramesLen - 1

        if (self.RawDataHigh == self.TimeIndex[lastTFI] + 2):  # bo kurwa zupa za słona
            self.resetEnv()

        self.TimeIndex[lastTFI] += 1
        CurrentTime = self.TimeData[lastTFI][self.TimeIndex[lastTFI]]

        for tfi in range(0, lastTFI):  # without last one (added as first and as reference)
            if (self.TimeIndex[tfi] + 1 != len(self.TimeData[tfi])):  # zabezpieczenie koncowe
                if (self.TimeData[tfi][self.TimeIndex[tfi] + 1] <= CurrentTime):
                    self.TimeIndex[tfi] += 1

        for InstI in range(0, self.InstrumentCount):
            self.CurrentPrice[InstI] = self.priceData[lastTFI][self.TimeIndex[lastTFI], InstI]

        State = numpy.zeros((self.TimeFramesLen, self.PastDataForStep, self.StanardFeatureLen))

        for tfi in range(0, self.TimeFramesLen):
            maxIndex = self.TimeIndex[tfi]
            minIndex = self.TimeIndex[tfi] - self.PastDataForStep
            State[tfi, :, :] = self.FullData[tfi][minIndex:maxIndex, :]

        CurrentTime = self.NormTime(CurrentTime)

        return State, CurrentTime, self.CurrentPrice

    @nb.jit
    def NormTime(self, Time):
        Time = datetime.datetime.utcfromtimestamp(Time.tolist() / 1e9)
        Day = numpy.zeros((32))
        WeekDay = numpy.zeros((7))
        Hour = numpy.zeros((24))
        Minute = numpy.zeros((60))
        Day[Time.day] = 1
        WeekDay[Time.weekday()] = 1
        Hour[Time.hour] = 1
        Minute[Time.minute] = 1
        # Add time to normalized data
        NormalizedTime = numpy.concatenate((Day, WeekDay, Hour, Minute), axis=0)
        return NormalizedTime

    @nb.jit
    def ExecuteTrades(self, action):
        # print('Old Portfolio: '+str(self.CurrentPossitions))
        # print('Newactions:'+str(action))

        action = action * 5

        for insti in range(1, self.InstrumentCount + 1):
            if (action[insti * 2 - 1] < 0):
                action[insti * 2 - 1] = 0
            if (action[insti * 2] < 0):
                action[insti * 2] = 0

            # print('Inst act:'+str(action))

            Long = self.CurrentPossitions[insti * 2 - 1]
            Short = self.CurrentPossitions[insti * 2]
            NLong = action[insti * 2 - 1]
            NShort = action[insti * 2]

            if (NLong > NShort and NLong > 0):
                NLong = math.ceil(NLong)
                NShort = 0
            if (NLong < NShort and NShort > 0):
                NShort = math.ceil(NShort)
                NLong = 0

            ShortChange = NShort - Short
            LongsChange = NLong - Long
            self.CurrentPossitions[insti * 2 - 1] = NLong
            self.CurrentPossitions[insti * 2] = NShort
            insti = insti - 1  # to get proper InstrumnetID

            if (LongsChange > 0):
                for lci in range(0, int(LongsChange)):
                    self.OpenLong(insti)
            if (LongsChange < 0):
                for lci in range(0, int(-LongsChange)):
                    self.CloseLong(insti)

            if (ShortChange > 0):
                for lci in range(0, int(ShortChange)):
                    self.OpenShort(insti)
            if (ShortChange < 0):
                for lci in range(0, int(-ShortChange)):
                    self.CloseShort(insti)

        Reward = self.equity - self.equityOld
        self.equityOld = self.equity
        return Reward

    @nb.jit
    def OpenLong(self, InstrumentID):
        if (self.TransactionalInstruments[InstrumentID] == 0):  # skip if its not tradable instrument
            return

        if (self.LastLongID[InstrumentID] + 1 == self.maxPositions):
            return

        if (self.mode == 1):
            message = "TRADE|OPEN|" + self.instruments[InstrumentID] + "|0|0.1|0|0|0|Python-to-MT4"
            self.Connector.remote_send(message)

        self.LastLongID[InstrumentID] = self.LastLongID[InstrumentID] + 1
        self.LongPositions[InstrumentID, self.LastLongID[InstrumentID]] = self.CurrentPrice[InstrumentID]
        return

    @nb.jit
    def OpenShort(self, InstrumentID):
        if (self.TransactionalInstruments[InstrumentID] == 0):  # skip if its not tradable instrument
            return

        if (self.LastShortID[InstrumentID] + 1 == self.maxPositions):
            return

        if (self.mode == 1):
            message = "TRADE|OPEN|" + self.instruments[InstrumentID] + "|1|0.1|0|0|0|Python-to-MT4"
            self.Connector.remote_send(message)

        self.LastShortID[InstrumentID] = self.LastShortID[InstrumentID] + 1
        self.ShortPositions[InstrumentID, self.LastShortID[InstrumentID]] = self.CurrentPrice[InstrumentID]

    @nb.jit
    def CloseLong(self, InstrumentID):

        if (self.TransactionalInstruments[InstrumentID] == 0):  # skip if its not tradable instrument
            return

        # eurusd_close_orders = "TRADE|CLOSE|1-Short,0-Long|Vloume|EURUSD|0|Python-to-MT4"
        if (self.LastLongID[InstrumentID] == -1):
            return

        if (self.mode == 1):
            message = "TRADE|CLOSE|" + self.instruments[InstrumentID] + "|0|0.1|0|0|0|Python-to-MT4"
            self.Connector.remote_send(message)

        profit = (self.CurrentPrice[InstrumentID] - self.LongPositions[InstrumentID, self.LastLongID[InstrumentID]]) * \
                 self.lot[InstrumentID]
        self.equity = self.equity + profit - self.fee
        self.LongPositions[InstrumentID, self.LastLongID[InstrumentID]] = 0
        self.LastLongID[InstrumentID] = self.LastLongID[InstrumentID] - 1
        return

    @nb.jit
    def CloseShort(self, InstrumentID):
        if (self.TransactionalInstruments[InstrumentID] == 0):  # skip if its not tradable instrument
            return

        if (self.LastShortID[InstrumentID] == -1):
            return

        if (self.mode == 1):
            message = "TRADE|CLOSE|" + self.instruments[InstrumentID] + "|1|0.1|0|0|0|Python-to-MT4"
            self.Connector.remote_send(message)

        profit = (self.CurrentPrice[InstrumentID] - self.ShortPositions[InstrumentID, self.LastShortID[InstrumentID]]) * \
                 self.lot[InstrumentID]
        self.equity = self.equity + profit - self.fee
        self.ShortPositions[InstrumentID, self.LastShortID[InstrumentID]] = 0
        self.LastShortID[InstrumentID] = self.LastShortID[InstrumentID] - 1
        return

    def getHistory(self):
        if (self.DownloadMode == 1):
            for GI in range(0, len(self.Granularity)):
                self.downloadHistory(GI)

        # self.FullDataPD = [object]*self.TimeFramesLen
        self.PriceScallarsD = [object] * self.TimeFramesLen
        self.VolumeScallarsD = [object] * self.TimeFramesLen

        for GI in range(0, len(self.Granularity)):
            downloadHist = [object] * len(self.Instruments)
            for InstI in range(0, len(self.Instruments)):
                FileName = "data/OandaPdData" + str(self.Granularity[GI]) + "_" + str(self.Instruments[InstI]) + ".csv"
                downloadHist[InstI] = pd.read_csv(FileName, sep=';', encoding='utf-8')
                downloadHist[InstI].index = pd.DatetimeIndex(downloadHist[InstI]['time'])
                downloadHist[InstI].drop_duplicates(inplace=True)

            ConcatData = pd.concat(downloadHist, axis=1, join='outer')
            ConcatData = ConcatData.drop(['complete', 'highBid', 'lowBid', 'openBid', 'time'], axis=1)
            ConcatData['volume'] = ConcatData['volume'].fillna(0)
            ConcatData['closeAsk'] = ConcatData['closeAsk'].fillna(method='ffill')
            ConcatData['closeBid'] = ConcatData['closeBid'].fillna(method='ffill')
            ConcatData['highAsk'] = ConcatData['highAsk'].fillna(method='ffill')
            ConcatData['lowAsk'] = ConcatData['lowAsk'].fillna(method='ffill')
            ConcatData['openAsk'] = ConcatData['openAsk'].fillna(method='ffill')
            ConcatData['closeAsk'] = ConcatData['closeAsk'].fillna(method='bfill')
            ConcatData['closeBid'] = ConcatData['closeBid'].fillna(method='bfill')
            ConcatData['highAsk'] = ConcatData['highAsk'].fillna(method='bfill')
            ConcatData['lowAsk'] = ConcatData['lowAsk'].fillna(method='bfill')
            ConcatData['openAsk'] = ConcatData['openAsk'].fillna(method='bfill')

            ConcatData[['closeAsk', 'closeBid', 'highAsk', 'lowAsk', 'openAsk']].info()
            priceData = ConcatData[['closeAsk', 'closeBid', 'highAsk', 'lowAsk', 'openAsk']].values
            volumeData = ConcatData['volume'].values

            # self.DataExploration(priceData,1)
            priceDELTA = priceData[1:, :] - priceData[:-1, :]
            # volumeDELA = volumeData[1:,:]-volumeData[:-1,:]   # chyba nie będzie potrzebne, delta chyba jest gorsza ni wartość bezwzględna
            TimeData = ConcatData.index.values[1:]
            priceData = priceData[1:]
            volumeData = volumeData[1:]

            i, j = priceDELTA.shape
            self.PriceScallarsD[GI] = [object] * j
            for fj in range(0, j):
                self.PriceScallarsD[GI][fj] = MinMaxScaler(feature_range=(-1, 1))
                self.PriceScallarsD[GI][fj].fit(priceDELTA[:, fj].reshape(-1, 1))
                # print(self.MinMaxScallarsD[GI][fj].data_max_)
                priceDELTA[:, fj] = self.PriceScallarsD[GI][fj].transform(priceDELTA[:, fj].reshape(-1, 1)).reshape(
                    (i,))

            i, j = volumeData.shape
            self.VolumeScallarsD[GI] = [object] * j
            for fj in range(0, j):
                self.VolumeScallarsD[GI][fj] = MinMaxScaler(feature_range=(0, 1))
                self.VolumeScallarsD[GI][fj].fit(volumeData[:, fj].reshape(-1, 1))
                # print(self.MinMaxScallarsD[GI][fj].data_max_)
                volumeData[:, fj] = self.VolumeScallarsD[GI][fj].transform(volumeData[:, fj].reshape(-1, 1)).reshape(
                    (i,))

            Indicators = self.CalcIndicatorsFull(priceDELTA, volumeData, GI)
            # self.FullData[GI] = numpy.concatenate((priceDELTA,volumeData,Indicators), axis = 1)

            # self.FullData[GI] = ConcatData.values
            self.FullData[GI] = numpy.concatenate((priceDELTA, volumeData, Indicators), axis=1)
            self.priceData[GI] = priceData
            self.TimeData[GI] = TimeData
            self.RawDataHigh, self.StanardFeatureLen = self.FullData[GI].shape  # potem uywam i tak tylko ostatniego

        return

    def CalcIndicatorsFull(self, priceDELTA, volumeData, GI):
        i, j = priceDELTA.shape
        print(priceDELTA.shape)
        self.IndicatorsCount = 12

        Indicators = numpy.zeros((i, self.IndicatorsCount))

        for li in range(self.MeanMax + self.PastDataForStep, i):
            if (max(priceDELTA[li - self.MeanMax:li, 6]) - min(priceDELTA[li - self.MeanMax:li, 9]) == 0):
                osc1 = 0
            else:
                osc1 = (priceDELTA[li, 0] - min(priceDELTA[li - self.MeanMax:li, 9])) / (
                            max(priceDELTA[li - self.MeanMax:li, 6]) - min(priceDELTA[li - self.MeanMax:li, 9]))
            if (max(priceDELTA[li - self.MeanMax:li, 7]) - min(priceDELTA[li - self.MeanMax:li, 10]) == 0):
                osc2 = 0
            else:
                osc2 = (priceDELTA[li, 1] - min(priceDELTA[li - self.MeanMax:li, 10])) / (
                            max(priceDELTA[li - self.MeanMax:li, 7]) - min(priceDELTA[li - self.MeanMax:li, 10]))
            if (max(priceDELTA[li - self.MeanMax:li, 8]) - min(priceDELTA[li - self.MeanMax:li, 11]) == 0):
                osc3 = 0
            else:
                osc3 = (priceDELTA[li, 3] - min(priceDELTA[li - self.MeanMax:li, 11])) / (
                            max(priceDELTA[li - self.MeanMax:li, 8]) - min(priceDELTA[li - self.MeanMax:li, 11]))
            Indicators[li, :] = [sum(priceDELTA[li - self.MeanMax:li, 0]) / self.MeanMax,
                                 sum(priceDELTA[li - self.MeanMax:li, 1]) / self.MeanMax,
                                 sum(priceDELTA[li - self.MeanMax:li, 2]) / self.MeanMax,
                                 sum(priceDELTA[li - int(self.MeanMax / 2):li, 0]) / self.MeanMax / 2,
                                 sum(priceDELTA[li - int(self.MeanMax / 2):li, 1]) / self.MeanMax / 2,
                                 sum(priceDELTA[li - int(self.MeanMax / 2):li, 2]) / self.MeanMax / 2,
                                 sum(priceDELTA[li - self.MeanMax:li, 0]) - sum(priceDELTA[li - self.MeanMax:li, 3]),
                                 sum(priceDELTA[li - self.MeanMax:li, 1]) - sum(priceDELTA[li - self.MeanMax:li, 4]),
                                 sum(priceDELTA[li - self.MeanMax:li, 2]) - sum(priceDELTA[li - self.MeanMax:li, 5]),
                                 osc1,
                                 osc2,
                                 osc3
                                 ]

        return Indicators

    def DataExploration(self, Data, Exit=0):
        FullData = Data
        i, j = FullData.shape
        print('Shape' + str(FullData.shape))
        # print(numpy.cov(Data))

        # print(numpy.corrcoef(Data))
        # numpy.savetxt("TempFullDatas.csv",numpy.corrcoef(Data), delimiter=";")
        x = range(0, i)
        y1 = FullData[:, 9]
        y2 = FullData[:, 10]
        y3 = FullData[:, 11]

        plt.subplot(2, 3, 1)
        plt.hist(y1)
        plt.subplot(2, 3, 2)
        plt.hist(y2)
        plt.subplot(2, 3, 3)
        plt.hist(y3)

        plt.subplot(2, 3, 4)
        plt.plot(x, y1)
        plt.subplot(2, 3, 5)
        plt.plot(y2)
        plt.subplot(2, 3, 6)
        plt.plot(y3)
        plt.show()

        print('Data exploration over')
        if (Exit):
            exit()
        return

# Oanda = OandaEnv(DownloadMode = 0)
# Oanda.getHistory()
# Oanda.resetEnv()

# for i in range(0,10):
# 	print(i)
# 	# print(type(Oanda))
# 	State, CurrentTime, Reward,_ = Oanda.Next()
