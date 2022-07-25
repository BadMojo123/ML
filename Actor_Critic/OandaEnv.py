import configparser  # 1
import time
from datetime import date, timedelta

import numba as nb
import numpy
import pandas as pd

from oandapy import oandapy


class OandaEnv:
    def __init__(self, Rycheza, action_space, DownloadMode=0):
        self.Rycheza = Rycheza
        self.mode = 0
        self.DownloadMode = DownloadMode
        # self.Instruments = ["SPX500_USD", "EUR_USD", "DE30_EUR"]
        self.Instruments = ["SPX500_USD", "EUR_USD"]
        self.Granularity = "M1"

        self.lot = [1, 1000, 1]
        self.InstrumentCount = len(self.Instruments)
        TXTconfig = configparser.ConfigParser()  # 3
        TXTconfig.read('oanda.cfg')  # 4
        self.oandaAPI = oandapy.API(environment='practice', access_token=TXTconfig['oanda']['access_token'])
        self.OandaFeatures = 6

        # Tragind paramiters
        self.TransactionalInstruments = [1, 1, 1]
        self.action_space = action_space
        self.CurrentPossitions = numpy.zeros((self.action_space))
        self.maxPositions = 1  # działało z 2
        self.fee = 0
        self.equity = 0
        self.equityOld = 0
        self.LongCount = 0
        self.ShortCount = 0
        self.RunEquity = 0
        self.LastEquity = 0
        self.BatchID = 0
        # self.BatchSize = 100000
        self.BatchSize = 5000
        self.LongPositions = numpy.zeros((self.InstrumentCount, self.maxPositions * 100))
        self.ShortPositions = numpy.zeros((self.InstrumentCount, self.maxPositions * 100))
        self.LastLongID = numpy.zeros((self.InstrumentCount), dtype=numpy.int8) - 1
        self.LastShortID = numpy.zeros((self.InstrumentCount), dtype=numpy.int8) - 1
        self.CurrentPrice = numpy.zeros((self.InstrumentCount,))
        self.getHistory()

    def downloadHistory(self, GI):
        downloadHist = [object] * len(self.Instruments)
        a = date(2015, 1, 3)
        b = date(2018, 12, 20)
        delta = timedelta(days=50)
        days = int((b - a).days / 50) + 1

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

        print('Env Log   Equity:' + str(self.equity) + '  BatchID: ' + str(self.BatchID))
        if (self.RawDataHigh <= (self.BatchID + 10)):
            self.LogRun()
            self.BatchID = 0
            self.RunEquity = 0
            self.LongCount = 0
            self.ShortCount = 0
            self.Rycheza.reset()
        # done

        State, reward, done, CurrentTime = self.Next(numpy.zeros((self.action_space)))  # close all trades
        return State, reward, done, CurrentTime

    @nb.jit
    def Next(self, action_matrix):
        done = 0

        reward = self.ExecuteTrades(action_matrix)  # najpierw trade, potem kolejny krok
        # reward = 0

        self.BatchID = self.BatchID + 1
        CurrentTime = self.TimeData[self.BatchID]  # nowy czas
        if (self.RawDataHigh <= (
                self.BatchID + 10) or self.BatchID % self.BatchSize == 0):  # bo czasem zaglądamy w przyszłość
            done = 2
            self.resetEnv()

        State = self.priceData[self.BatchID, :, :]
        self.CurrentPrice = self.priceData[self.BatchID, :, 0]

        # TODO send reward fucntion insted of reward
        # reward = reward + self.OpenPositionsValue()/2

        return State, reward, done, action_matrix

    @nb.jit
    def ExecuteTrades(self, action):
        action = action * (self.maxPositions + 1)
        # print('Old Positions: ' + str(self.CurrentPossitions))
        # print('Action: '+str(action))
        # action = action * -1
        # print('New Action: '+str(action))
        # print('LastShortID:' + str(self.LastShortID))
        # print('LastLongID:' + str(self.LastLongID))
        for i in range(0, len(action)):
            if action[i] > 0:
                action[i] = numpy.floor(action[i])
            else:
                action[i] = numpy.ceil(action[i])

        for InstI in range(0, self.InstrumentCount):
            while action[InstI] != self.CurrentPossitions[InstI]:
                if (action[InstI] < self.CurrentPossitions[InstI]):  # go short
                    if (self.CurrentPossitions[InstI] > 0):  # close long
                        self.CloseLong(InstI)
                        self.CurrentPossitions[InstI] = self.CurrentPossitions[InstI] - 1
                        continue

                    if (self.CurrentPossitions[InstI] == 0):  # close long
                        self.CurrentPossitions[InstI] = self.CurrentPossitions[InstI] - 1
                        continue

                    if (self.CurrentPossitions[InstI] < 0):  # Open short
                        self.OpenShort(InstI)
                        self.CurrentPossitions[InstI] = self.CurrentPossitions[InstI] - 1
                        continue
                if (action[InstI] > self.CurrentPossitions[InstI]):  # go long
                    if (self.CurrentPossitions[InstI] < 0):  # close short
                        self.CloseShort(InstI)
                        self.CurrentPossitions[InstI] = self.CurrentPossitions[InstI] + 1
                        continue

                    if (self.CurrentPossitions[InstI] == 0):  # close long
                        self.CurrentPossitions[InstI] = self.CurrentPossitions[InstI] + 1
                        continue

                    if (self.CurrentPossitions[InstI] > 0):  # Open short
                        self.OpenLong(InstI)
                        self.CurrentPossitions[InstI] = self.CurrentPossitions[InstI] + 1
                        continue

        Reward = self.equity - self.equityOld
        self.equityOld = self.equity
        self.RunEquity = self.RunEquity + Reward
        # print('New positions: ' + str(self.CurrentPossitions))
        return Reward

    @nb.jit
    def OpenLong(self, InstrumentID):
        if (self.TransactionalInstruments[InstrumentID] == 0):  # skip if its not tradable instrument
            return

        if (self.mode == 1):
            message = "TRADE|OPEN|" + self.instruments[InstrumentID] + "|0|0.1|0|0|0|Python-to-MT4"
            self.Connector.remote_send(message)

        self.LastLongID[InstrumentID] = self.LastLongID[InstrumentID] + 1
        self.LongPositions[InstrumentID, self.LastLongID[InstrumentID]] = self.priceData[self.BatchID, InstrumentID, 0]
        self.LongCount += 1
        return

    @nb.jit
    def OpenShort(self, InstrumentID):
        if (self.TransactionalInstruments[InstrumentID] == 0):  # skip if its not tradable instrument
            return

        if (self.mode == 1):
            message = "TRADE|OPEN|" + self.instruments[InstrumentID] + "|1|0.1|0|0|0|Python-to-MT4"
            self.Connector.remote_send(message)

        self.LastShortID[InstrumentID] = self.LastShortID[InstrumentID] + 1
        self.ShortPositions[InstrumentID, self.LastShortID[InstrumentID]] = self.priceData[
            self.BatchID, InstrumentID, 1]
        self.ShortCount += 1
        return

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

        profit = (self.priceData[self.BatchID, InstrumentID, 1] - self.LongPositions[
            InstrumentID, self.LastLongID[InstrumentID]]) * self.lot[InstrumentID]
        self.equity = self.equity + profit - self.fee
        # print('CLOSE LONG - Equity: ' + str(self.equity) + ' Profit: ' + str(profit - self.fee) +'   Close: '+str(self.priceData[self.BatchID,InstrumentID,1])+ '  Open: '+str(self.LongPositions[InstrumentID,self.LastLongID[InstrumentID]]))
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

        profit = (self.ShortPositions[InstrumentID, self.LastShortID[InstrumentID]] - self.priceData[
            self.BatchID, InstrumentID, 1]) * self.lot[InstrumentID]
        self.equity = self.equity + profit - self.fee
        # print('CLOSE SORT - Equity: ' + str(self.equity) + ' Profit: ' + str(profit - self.fee) +'   Close: '+str(self.priceData[self.BatchID,InstrumentID,1])+ '  Open: '+str(self.ShortPositions[InstrumentID,self.LastShortID[InstrumentID]]))
        self.ShortPositions[InstrumentID, self.LastShortID[InstrumentID]] = 0
        self.LastShortID[InstrumentID] = self.LastShortID[InstrumentID] - 1
        return

    def getHistory(self):
        if (self.DownloadMode == 1):
            self.downloadHistory(GI)

        downloadHist = [object] * len(self.Instruments)
        for InstI in range(0, len(self.Instruments)):
            FileName = "data/OandaPdData" + str(self.Granularity) + "_" + str(self.Instruments[InstI]) + ".csv"
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

        # ConcatData[['closeAsk','closeBid','highAsk','lowAsk','openAsk']].info()
        priceData = ConcatData[['closeAsk', 'closeBid', 'highAsk', 'lowAsk', 'openAsk', 'volume']]
        i, j = priceData.values.shape

        FullPriceData = numpy.zeros((i, self.InstrumentCount, int(j / 2)))
        columnNames = ['closeAsk', 'closeBid', 'highAsk', 'lowAsk', 'openAsk', 'volume']
        # for instI in range(self.InstrumentCount):
        i = 0
        for C in columnNames:
            FullPriceData[:, :, i] = priceData[C].values[:, :]
            i += 1

        # self.DataExploration(FullPriceData)

        TimeData = ConcatData.index

        self.priceData = FullPriceData
        self.TimeData = TimeData
        self.RawDataHigh, _, _ = FullPriceData.shape
        print(self.RawDataHigh)

        return

    def DataExploration(self, Data, Exit=0):
        import matplotlib.pyplot as plt
        FullData = Data
        i, j, k = FullData.shape
        print('Shape' + str(FullData.shape))
        # print(numpy.cov(Data))

        # print(numpy.corrcoef(Data))
        # numpy.savetxt("TempFullDatas.csv",numpy.corrcoef(Data), delimiter=";")
        x = range(0, i)
        y1 = FullData[:, 0, 3]
        y2 = FullData[:, 0, 4]
        y3 = FullData[:, 0, 5]

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

    @nb.jit
    def LogRun(self):
        if (self.LongCount + self.ShortCount == 0):
            MeanProfit = 0
        else:
            MeanProfit = self.RunEquity / (self.LongCount + self.ShortCount)

        summary = 'Env For Licho;' + str(self.Rycheza.Licho.ID) + ";"
        summary = summary + ' Equity: ;' + str(self.RunEquity) + ";"
        summary = summary + ' Total fee: ;' + str((self.LongCount + self.ShortCount) * self.fee) + ";"
        summary = summary + ' Fee: ;' + str(self.fee) + ";"
        summary = summary + ' Long Count: ;' + str(self.LongCount) + ";"
        summary = summary + ' Short Count: ;' + str(self.ShortCount) + ";"
        summary = summary + ' Mean profit: ;' + str(MeanProfit) + ";"
        summary = summary + str(self.Instruments) + "\n"
        file = open('results/EnvResults.csv', 'a')
        file.write(summary)
        file.close()
        print(summary)
        return

# Oanda = OandaEnv(2,DownloadMode = 0)


# for i in range(0,10):
# 	print(i)
# 	# print(type(Oanda))
# 	State, reward, done, CurrentTime = Oanda.Next([0,0])
