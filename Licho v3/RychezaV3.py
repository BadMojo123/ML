import numpy
from sklearn.preprocessing import MinMaxScaler

from new import LichoRLPPO
from Mieszko_v2 import Mieszko
from OandaEnv import OandaEnv

NUM_ACTIONS = 3
DUMMY_ACTION, DUMMY_VALUE = numpy.zeros((NUM_ACTIONS)), numpy.zeros((1, 1))
BatchSize = 10000
EMA_Price_Gamma = 0.9
EMA_Change_Gamma = 0.75


# RychezaV3
class Rycheza:
    def __init__(self, Fit=1):
        self.action_space = NUM_ACTIONS
        self.env = OandaEnv(self, self.action_space)
        self.InstrumentCount = self.env.InstrumentCount
        self.OandaFeatures = self.env.OandaFeatures
        self.SavedHist = 200  # steps saved, mainly for scalars
        self.TransformerHist = 60  # steps for transformer

        self.IndicatorCount = 2
        self.PriceScalars = [object] * self.InstrumentCount
        self.SpreadScalar = [object] * self.InstrumentCount
        self.BarSizeScalar = [object] * self.InstrumentCount

        # ConcatData[['closeAsk','closeBid','highAsk','lowAsk','openAsk','volume']]
        if (Fit == 1):
            for InstI in range(self.InstrumentCount):
                self.PriceScalars[InstI] = MinMaxScaler(feature_range=(0, 1))
                self.PriceScalars[InstI].fit(self.env.priceData[:, InstI, 0].reshape(-1, 1))

                self.SpreadScalar[InstI] = MinMaxScaler(feature_range=(0, 1))
                self.SpreadScalar[InstI].fit(
                    (self.env.priceData[:, InstI, 0] - self.env.priceData[:, InstI, 1]).reshape(-1, 1))

                self.BarSizeScalar[InstI] = MinMaxScaler(feature_range=(0, 1))
                self.BarSizeScalar[InstI].fit(
                    (self.env.priceData[:, InstI, 2] - self.env.priceData[:, InstI, 3]).reshape(-1, 1))

        self.RawData = numpy.zeros((self.SavedHist, self.InstrumentCount, self.OandaFeatures))
        self.Indicators = numpy.zeros((self.TransformerHist, self.InstrumentCount, self.IndicatorCount))
        self.OldMieszkoPred = numpy.zeros((1, self.InstrumentCount * self.IndicatorCount))
        self.BatchStep = 0

    def reset(self):
        # State, reward, done, CurrentTime = self.env.resetEnv()
        li = 0
        self.BatchStep = 0
        while li < self.TransformerHist:
            self.step(DUMMY_ACTION)
            li += 1
        observation, _, _, _ = self.step(DUMMY_ACTION)
        return observation

    def step(self, action):
        reward, done, action_matrix = self.FillRawData(action)
        MieszkoPred = self.PrepTaransformerObservation()
        observation = self.PrepRLObservation(MieszkoPred, action_matrix)
        return observation, reward, done, action_matrix

    def FillRawData(self, action):
        State, reward, done, action_matrix = self.env.Next(action)
        # State = [InstrumentCount, Oanda Features]
        # ConcatData[['closeAsk','closeBid','highAsk','lowAsk','openAsk','volume']]
        self.RawData[1:, :, :] = self.RawData[:-1, :, :]
        self.RawData[0, :, :] = State
        return reward, done, action_matrix

    def PrepTaransformerObservation(self, Predict=1):
        self.BatchStep = self.BatchStep + 1
        if ((self.BatchStep % 2500) == 0):
            print('Step, Batch ID: ' + str(self.BatchStep))

        for li in range(self.InstrumentCount):
            HighP = self.PriceScalars[li].transform(self.RawData[0, li, 2].reshape(1, -1))
            LowP = self.PriceScalars[li].transform(self.RawData[0, li, 3].reshape(1, -1))
            CloseP = self.PriceScalars[li].transform(self.RawData[0, li, 4].reshape(1, -1))
            Spread = self.SpreadScalar[li].transform((self.RawData[0, li, 0] - self.RawData[0, li, 1]).reshape(1, -1))
            BarSize = self.BarSizeScalar[li].transform((self.RawData[0, li, 2] - self.RawData[0, li, 3]).reshape(1, -1))

            MidPrice = (HighP + LowP + CloseP) / 3

            MidChange = MidPrice[:] - self.Indicators[0, li, 0]
            # EMA_Price = MidPrice*EMA_Price_Gamma+self.Indicators[0,li,2]*(1-EMA_Price_Gamma)
            # EMA_Change = MidChange*EMA_Change_Gamma+self.Indicators[0,li,3]*(1-EMA_Change_Gamma)

            self.Indicators[1:, li, :] = self.Indicators[:-1, li, :]
            self.Indicators[0, li, 0] = MidPrice
            self.Indicators[0, li, 1] = MidPrice
        # self.Indicators[0,li,1] = MidChange
        # self.Indicators[0,li,2] = EMA_Price
        # self.Indicators[0,li,3] = EMA_Change
        # self.Indicators[0,li,4] = Spread
        # self.Indicators[0,li,5] = BarSize

        Fited = self.Indicators.reshape((self.TransformerHist, (self.InstrumentCount * self.IndicatorCount)))
        Fited = Fited.reshape((1, self.TransformerHist, self.InstrumentCount * self.IndicatorCount))
        if (Predict == 1):
            Step1 = self.Mieszko.model.predict(Fited)
            MieszkoPred = Step1
            for li in range(0, 3):
                Fited[1:, :, :] = Fited[:-1, :, :]
                Fited[0, :, :] = MieszkoPred
                MieszkoPred = self.Mieszko.model.predict(Fited)
            return [Step1, MieszkoPred]
        else:
            return

    def PrepRLObservation(self, MieszkoPred, action_matrix):
        # observation = (self.RawData[0,:,:]-self.RawData[1,:,:]).flatten()
        # observation = numpy.concatenate([observation,FitedIndicators,MieszkoPred.flatten()])
        FitedIndicators = self.Indicators[0, :, :].flatten()
        MieszkoPred1 = MieszkoPred[0].flatten()
        MieszkoPredDiff1 = MieszkoPred1 - FitedIndicators
        MieszkoPred2 = MieszkoPred[1].flatten()
        MieszkoPredDiff2 = MieszkoPred2 - MieszkoPred1
        # RloIndicators =
        observation = numpy.concatenate(
            [FitedIndicators, MieszkoPred1.flatten(), MieszkoPredDiff1.flatten(), MieszkoPred2.flatten(),
             MieszkoPredDiff2.flatten(), action_matrix])
        self.OldMieszkoPred = MieszkoPred

        # # # # Taking future data for testing RL   !!! ULTRA CHEAT
        # CurrentPrice = self.env.priceData[self.env.BatchID,:,0]
        # FuturePrice = self.env.priceData[self.env.BatchID+1,:,0]
        # # FuturePrice5 = self.env.priceData[self.env.BatchID+5,:,0]
        # NextTimeFrameDiff = CurrentPrice - FuturePrice
        # # NextTimeFrameDiff5 = CurrentPrice - FuturePrice5
        # observation = numpy.concatenate([observation,NextTimeFrameDiff.flatten()])

        return observation

    def TrainLicho(self, LichoID, MieszkoID):
        self.Licho = LichoRLPPO.Licho(self, ID=LichoID)
        self.Mieszko = Mieszko(self, ID=MieszkoID)
        self.Licho.observation = self.Licho.env.reset()
        self.Licho.run()
        return

    def TradeRun(self):
        self.env.BatchSize = 500
        self.Licho = LichoRLPPO.Licho(self, ID=1559653792)
        self.Mieszko = Mieszko(self, ID=1561625696)
        self.Licho.observation = self.Licho.env.reset()
        action = DUMMY_ACTION
        a = 0
        i = 0
        while a == 0:
            observation, reward, done, action_matrix = self.step(action)
            self.Licho.observation = observation
            self.Licho.val = True
            action, action_matrix, p = self.Licho.get_action_continuous()
            i += 1
            # print('step'+str(i))
            if (i % 5000 == 0):
                print('Step: ' + str(i))
        return

    def TrainMieszko(self, ID=0):
        self.Mieszko = Mieszko(self, ID=ID)
        self.env.BatchSize = 100
        self.reset()
        self.Mieszko.EPOCHS = 2
        self.Mieszko.TrainMieszko()


Runner = Rycheza()
Runner.TrainMieszko(ID=0)
# Runner.TrainLicho(LichoID=1569407080, MieszkoID=1569405247)
# Runner.TradeRun()
