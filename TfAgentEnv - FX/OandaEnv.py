import time

import configparser  # 1
import numpy
import pandas as pd
from datetime import date, timedelta

import oandapy


class OandaEnv:
    def __init__(self, trade_runner, action_space, DownloadMode=0):
        self.TradeRunner = trade_runner
        self.mode = 0
        self.DownloadMode = DownloadMode
        # self.Instruments = ["SPX500_USD", "EUR_USD", "DE30_EUR"]
        self.Instruments = ["SPX500_USD", "EUR_USD"]
        self.Granularity = ["M15", "H1"]

        self.lot = [1, 1000, 1]
        self.InstrumentCount = len(self.Instruments)
        txt_config = configparser.ConfigParser()  # 3
        txt_config.read('oanda.cfg')  # 4
        self.oandaAPI = oandapy.API(environment='practice', access_token=txt_config['oanda']['access_token'])
        self.OandaFeatures = 6

        # Trading parameters
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
        # self.BatchSize = 200
        self.BatchSize = 1000
        self.LongPositions = numpy.zeros((self.InstrumentCount, self.maxPositions * 100))
        self.ShortPositions = numpy.zeros((self.InstrumentCount, self.maxPositions * 100))
        self.LastLongID = numpy.zeros((self.InstrumentCount), dtype=numpy.int8) - 1
        self.LastShortID = numpy.zeros((self.InstrumentCount), dtype=numpy.int8) - 1
        self.CurrentPrice = numpy.zeros((self.InstrumentCount,))
        self.getHistory()

    def getHistory(self):
        if self.DownloadMode == 1:
            for GI in range(0, len(self.Granularity)):
                # GI = 0  # its index of granularity so '0' indicate its 'M1'
                self.download_history(GI)

        download_hist = [object] * len(self.Instruments)
        for GI in range(0, len(self.Granularity)):
            for InstI in range(0, len(self.Instruments)):
                file_name = "data/OandaPdData" + str(self.Granularity[GI]) + "_" + str(self.Instruments[InstI]) + ".csv"
                download_hist[InstI] = pd.read_csv(file_name, sep=';', encoding='utf-8')
                download_hist[InstI].index = pd.DatetimeIndex(download_hist[InstI]['time'])
                download_hist[InstI].drop_duplicates(inplace=True)

            ConcatData = pd.concat(download_hist, axis=1, join='outer')
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

            full_price_data = numpy.zeros((i, self.InstrumentCount, int(j / 2)))
            column_names = ['closeAsk', 'closeBid', 'highAsk', 'lowAsk', 'openAsk', 'volume']
            # for instI in range(self.InstrumentCount):
            i = 0
            for C in column_names:
                full_price_data[:, :, i] = priceData[C].values[:, :]
                i += 1

        # self.DataExploration(full_price_data)

        time_data = ConcatData.index

        self.priceData = full_price_data
        self.TimeData = time_data
        self.RawDataHigh, _, _ = full_price_data.shape
        print(f'(i) Enviorement data prepared, row count{self.RawDataHigh}')

        return

    def download_history(self, gi):
        # gi = 0
        download_hist = [object] * len(self.Instruments)
        a = date(2015, 1, 3)
        b = date(2015, 1, 30)
        delta = timedelta(days=50)
        days = int((b - a).days / 50) + 1

        for InstI in range(0, len(self.Instruments)):
            download_hist[InstI] = [object] * days
            d = a
            for i in range(0, days):
                start_date = d.strftime("%Y-%m-%d")
                d = d + delta
                end_date = d.strftime("%Y-%m-%d")
                print(
                    f'(i) Downloading Data. Period {start_date}) - {end_date}, instrument {self.Instruments[InstI]}, time span {str(self.Granularity[gi])}')
                download_hist[InstI][i] = self.instrument_download(str(self.Granularity[gi]), self.Instruments[InstI],
                                                                   start_date, end_date)

                time.sleep(2)

            download_hist[InstI] = pd.concat(download_hist[InstI][:], axis=0)
            # download_hist[InstI].reset_index(drop=True, inplace=True)
            file_name = "data/OandaPdData" + str(self.Granularity[gi]) + "_" + str(self.Instruments[InstI]) + ".csv"
            download_hist[InstI].to_csv(file_name, sep=';', encoding='utf-8')
            print(f'(i) Data Download complete {file_name}')
        return

    def instrument_download(self, granularity, instrument, StartDate, EndDate):
        data = self.oandaAPI.get_history(instrument=instrument,  # our instrument
                                         start=StartDate,  # start data
                                         end=EndDate,  # end date
                                         granularity=granularity)  # minute bars  # 7

        df = pd.DataFrame(data['candles']).set_index('time')  # 8
        df.index = pd.DatetimeIndex(df.index)  # 9

        return df

    def reset_env(self):
        self.log_run()
        self.BatchID = 0
        self.RunEquity = 0
        self.LongCount = 0
        self.ShortCount = 0
        state, reward, done, action_matrix = self.TradeRunner.reset()

        # state, reward, done, action_matrix = self.next(numpy.zeros((self.action_space)))  # close all trades
        return state, reward, done, action_matrix

    # @nb.jit
    def next(self, action):
        done = 0
        reward = self.execute_trades(action)  # najpierw trade, potem kolejny krok

        self.BatchID = self.BatchID + 1
        # CurrentTime = self.TimeData[self.BatchID]  # nowy czas
        if self.RawDataHigh <= (self.BatchID + 10):  # bo czasem zaglądamy w przyszłość
            done = 1  # koniec danych
            self.reset_env()

        if self.BatchID % self.BatchSize == 0:  # just for showing progress
            done = 2
            print('Env Log   Equity:' + str(self.equity) + '  BatchID: ' + str(self.BatchID))

        state = self.priceData[self.BatchID, :, :]
        # self.CurrentPrice = self.priceData[self.BatchID,:,0]


        # reward = reward + self.OpenPositionsValue()/2


        # print('Current positions? old_action matrix?'+str(self.CurrentPossitions))
        action_matrix = self.CurrentPossitions

        return state, reward, done, action_matrix

    # @nb.jit
    def execute_trades(self, action):
        action = action * (self.maxPositions + 1)
        # print('Old Positions: ' + str(self.CurrentPositions))
        # print('Action: '+str(old_action))
        # old_action = old_action * -1
        # print('New Action: '+str(old_action))
        # print('LastShortID:' + str(self.LastShortID))
        # print('LastLongID:' + str(self.LastLongID))
        for i in range(0, len(action)):
            if action[i] > 0:
                action[i] = numpy.floor(action[i])
            else:
                action[i] = numpy.ceil(action[i])

        for InstI in range(0, self.InstrumentCount):
            while action[InstI] != self.CurrentPossitions[InstI]:
                if action[InstI] < self.CurrentPossitions[InstI]:  # go short
                    if self.CurrentPossitions[InstI] > 0:  # close long
                        self.close_long(InstI)
                        self.CurrentPossitions[InstI] = self.CurrentPossitions[InstI] - 1
                        continue

                    if self.CurrentPossitions[InstI] == 0:  # close long
                        self.CurrentPossitions[InstI] = self.CurrentPossitions[InstI] - 1
                        continue

                    if self.CurrentPossitions[InstI] < 0:  # Open short
                        self.open_short(InstI)
                        self.CurrentPossitions[InstI] = self.CurrentPossitions[InstI] - 1
                        continue
                if action[InstI] > self.CurrentPossitions[InstI]:  # go long
                    if self.CurrentPossitions[InstI] < 0:  # close short
                        self.close_short(InstI)
                        self.CurrentPossitions[InstI] = self.CurrentPossitions[InstI] + 1
                        continue

                    if self.CurrentPossitions[InstI] == 0:  # close long
                        self.CurrentPossitions[InstI] = self.CurrentPossitions[InstI] + 1
                        continue

                    if self.CurrentPossitions[InstI] > 0:  # Open short
                        self.open_long(InstI)
                        self.CurrentPossitions[InstI] = self.CurrentPossitions[InstI] + 1
                        continue

        reward = self.equity - self.equityOld
        self.equityOld = self.equity
        self.RunEquity = self.RunEquity + reward
        # print('New positions: ' + str(self.CurrentPossitions))
        return reward

    # @nb.jit
    def open_long(self, instrument_id):
        if self.TransactionalInstruments[instrument_id] == 0:  # skip if its not tradable instrument
            return

        if self.mode == 1:
            message = "TRADE|OPEN|" + self.instruments[instrument_id] + "|0|0.1|0|0|0|Python-to-MT4"
            self.Connector.remote_send(message)

        self.LastLongID[instrument_id] = self.LastLongID[instrument_id] + 1
        self.LongPositions[instrument_id, self.LastLongID[instrument_id]] = self.priceData[
            self.BatchID, instrument_id, 0]
        self.LongCount += 1
        return

    # @nb.jit
    def open_short(self, instrument_id):
        if self.TransactionalInstruments[instrument_id] == 0:  # skip if its not tradable instrument
            return

        if self.mode == 1:
            message = "TRADE|OPEN|" + self.instruments[instrument_id] + "|1|0.1|0|0|0|Python-to-MT4"
            self.Connector.remote_send(message)

        self.LastShortID[instrument_id] = self.LastShortID[instrument_id] + 1
        self.ShortPositions[instrument_id, self.LastShortID[instrument_id]] = self.priceData[
            self.BatchID, instrument_id, 1]
        self.ShortCount += 1
        return

    # @nb.jit
    def close_long(self, instrument_id):

        if self.TransactionalInstruments[instrument_id] == 0:  # skip if its not tradable instrument
            return

        # eurusd_close_orders = "TRADE|CLOSE|1-Short,0-Long|Vloume|EURUSD|0|Python-to-MT4"
        if self.LastLongID[instrument_id] == -1:
            return

        if self.mode == 1:
            message = "TRADE|CLOSE|" + self.instruments[instrument_id] + "|0|0.1|0|0|0|Python-to-MT4"
            self.Connector.remote_send(message)

        profit = (self.priceData[self.BatchID, instrument_id, 1] - self.LongPositions[
            instrument_id, self.LastLongID[instrument_id]]) * self.lot[instrument_id]
        self.equity = self.equity + profit - self.fee
        # print('CLOSE LONG - Equity: ' + str(self.equity) + ' Profit: ' + str(profit - self.fee) +'   Close: '+str(
        # self.priceData[self.BatchID,instrument_id,1])+ '  Open: '+str(self.LongPositions[instrument_id,
        # self.LastLongID[instrument_id]]))
        self.LongPositions[instrument_id, self.LastLongID[instrument_id]] = 0
        self.LastLongID[instrument_id] = self.LastLongID[instrument_id] - 1
        return

    # @nb.jit
    def close_short(self, instrument_id):
        if self.TransactionalInstruments[instrument_id] == 0:  # skip if its not tradable instrument
            return

        if self.LastShortID[instrument_id] == -1:
            return

        if self.mode == 1:
            message = "TRADE|CLOSE|" + self.instruments[instrument_id] + "|1|0.1|0|0|0|Python-to-MT4"
            self.Connector.remote_send(message)

        profit = (self.ShortPositions[instrument_id, self.LastShortID[instrument_id]] - self.priceData[
            self.BatchID, instrument_id, 1]) * self.lot[instrument_id]
        self.equity = self.equity + profit - self.fee
        # print('CLOSE SORT - Equity: ' + str(self.equity) + ' Profit: ' + str(profit - self.fee) +'   Close: '+str(self.priceData[self.BatchID,instrument_id,1])+ '  Open: '+str(self.ShortPositions[instrument_id,self.LastShortID[instrument_id]]))
        self.ShortPositions[instrument_id, self.LastShortID[instrument_id]] = 0
        self.LastShortID[instrument_id] = self.LastShortID[instrument_id] - 1
        return

    def DataExploration(self, data, close_program=0):
        import matplotlib.pyplot as plt
        full_data = data
        i, j, k = full_data.shape
        print('Shape' + str(full_data.shape))
        # print(numpy.cov(data))

        # print(numpy.corrcoef(data))
        # numpy.savetxt("TempFullDatas.csv",numpy.corrcoef(data), delimiter=";")
        x = range(0, i)
        y1 = full_data[:, 0, 3]
        y2 = full_data[:, 0, 4]
        y3 = full_data[:, 0, 5]

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

        print('data exploration over')
        if close_program:
            exit()
        return

    def log_run(self):
        if self.LongCount + self.ShortCount == 0:
            mean_profit = 0
        else:
            mean_profit = self.RunEquity / (self.LongCount + self.ShortCount)

        summary = 'Env For Licho;' + str(self.TradeRunner.Licho.ID) + ";"
        summary = summary + ' Equity: ;' + str(self.RunEquity) + ";"
        summary = summary + ' Total fee: ;' + str((self.LongCount + self.ShortCount) * self.fee) + ";"
        summary = summary + ' Fee: ;' + str(self.fee) + ";"
        summary = summary + ' Long Count: ;' + str(self.LongCount) + ";"
        summary = summary + ' Short Count: ;' + str(self.ShortCount) + ";"
        summary = summary + ' Mean profit: ;' + str(mean_profit) + ";"
        summary = summary + str(self.Instruments) + "\n"
        file = open('results/EnvResults.csv', 'a')
        file.write(summary)
        file.close()
        print(summary)
        return


Oanda = OandaEnv(2, 2, DownloadMode=1)

# for i in range(0,10):
# 	print(i)
# 	# print(type(Oanda))
# 	State, reward, done, CurrentTime = Oanda.next([0,0])
