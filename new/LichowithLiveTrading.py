import time

import keras.models
import numba as nb
import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Flatten, LSTM, Lambda
from keras.models import Model
from keras.optimizers import Adam

from OandaEnv import OandaEnv

NOISE = 0.2
LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
ENTROPY_LOSS = 1e-4


@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1 - b1) * new


def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num / denom
        old_prob = old_prob_num / denom
        r = prob / (old_prob + 1e-10)

        return -K.mean(
            K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))

    return loss


class Licho:
    def __init__(self, ID):
        if (ID == 0):
            self.LichoID = time.time()
        else:
            self.LichoID = ID

        self.modelName = 'klony/Licho' + str(self.LichoID) + '.h5'
        self.CriticModelName = 'klony/Critic' + str(self.LichoID) + '.h5'
        self.episode = 0
        self.EPISODES = 100000  # ile razy ma wszystko leciec
        self.EPOCHS = 5  # ile razy ma sie uczyćza każdym razem
        self.GAMMA = 0.99
        self.BATCH_SIZE = 200  # ile minut na batch
        self.HIDDEN_SIZE = int(64)
        self.LR = 1e-4  # Lower lr stabilises training greatly

        # self.actionsSpaceLen = self.InstrumentCount*2 + 1 # [cash, instr1 Long, instr1 Short, instr2 Long...]
        self.actionsSpaceLen = 3
        self.env = OandaEnv(self.BATCH_SIZE, self.actionsSpaceLen)

        self.TimeFramesLen = self.env.TimeFramesLen
        self.reward_over_time = []
        self.InstrumentCount = self.env.InstrumentCount
        self.NormedFeaturesLen = self.env.StanardFeatureLen
        self.PastDataForStep = self.env.PastDataForStep

        self.TotalReward = 0

        self.DUMMY_ACTION, self.DUMMY_VALUE = np.zeros((1, self.actionsSpaceLen)), np.zeros((1, 1))
        self.val = False

        # self.writer = SummaryWriter('AllRuns/continuous')
        self.critic = self.build_critic()
        # self.critic.summary()
        self.actor = self.build_actor_continuous()
        # self.actor.summary()

        if (ID != 0):
            self.critic.load_weights(self.CriticModelName)
            self.actor.load_weights(self.modelName)
            print('Model loaded')
            f = open('episode.licho', 'r')
            self.episode = int(f.readline())
            f.close()

    def build_actor_continuous(self):
        state_input = Input(shape=(self.TimeFramesLen, self.PastDataForStep, self.NormedFeaturesLen,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.actionsSpaceLen,))
        time_input = Input(shape=(123,))
        price_input = Input(shape=(self.InstrumentCount,))

        xTime = Dense(16, activation='relu')(time_input)
        y = state_input
        tf0t0State_0 = Lambda(lambda y: y[:, 0, :, :], output_shape=(self.PastDataForStep, self.NormedFeaturesLen,))(y)
        # xRNN = Reshape((self.PastDataForStep,self.InstrumentCount*self.NormedFeaturesLen))(tf0t0State)
        xRNN_0 = LSTM(int(24))(tf0t0State_0)

        # tf0t0State_1 =Lambda(lambda y: y[:,1,:,:], output_shape=(self.PastDataForStep,self.NormedFeaturesLen,))(y)
        # # xRNN = Reshape((self.PastDataForStep,self.InstrumentCount*self.NormedFeaturesLen))(tf0t0State)
        # xRNN_1 = LSTM(int(24))(tf0t0State_1)

        # tf0t0State_2 =Lambda(lambda y: y[:,2,:,:], output_shape=(self.PastDataForStep,self.NormedFeaturesLen,))(y)
        # # xRNN = Reshape((self.PastDataForStep,self.InstrumentCount*self.NormedFeaturesLen))(tf0t0State)
        # xRNN_2 = LSTM(int(48))(tf0t0State_2)

        tfAt0State = state_input
        t0State = Lambda(lambda tfAt0State: tfAt0State[:, :, -1, :],
                         output_shape=(self.TimeFramesLen, self.NormedFeaturesLen,))(tfAt0State)
        t0DNN = Flatten()(t0State)
        t0DNN = Dense(self.HIDDEN_SIZE, activation='relu')(t0DNN)

        xRNN = keras.layers.concatenate([t0DNN, xRNN_0])
        xRNN = Dense(self.HIDDEN_SIZE, activation='relu')(xRNN)
        xRNN = Dropout(0.25)(xRNN)

        xPriceDNN = Dense(16, activation='relu')(price_input)
        xPriceDNN = Dropout(0.5)(xPriceDNN)

        # x = keras.layers.concatenate([xConv, t0DNN, xRNN, xTime, old_prediction,xPriceDNN])
        x = keras.layers.concatenate([xRNN, xTime, xPriceDNN])
        x = Dense(self.HIDDEN_SIZE, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(self.HIDDEN_SIZE, activation='relu')(x)

        out_actions = Dense(self.actionsSpaceLen, activation='tanh', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction, time_input, price_input], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=self.LR),
                      loss=[proximal_policy_optimization_loss_continuous(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        return model

    def build_critic(self):
        state_input = Input(shape=(self.TimeFramesLen, self.PastDataForStep, self.NormedFeaturesLen,))
        time_input = Input(shape=(123,))
        # old_prediction = Input(shape=(self.actionsSpaceLen,))
        price_input = Input(shape=(self.InstrumentCount,))

        DNNPrice = Dense(self.HIDDEN_SIZE, activation='relu')(price_input)
        DNNPrice = Dropout(0.5)(DNNPrice)

        xTime = Dense(16, activation='relu')(time_input)

        x1 = state_input
        t0DNN = Lambda(lambda x1: x1[:, :, 0, :], output_shape=(self.TimeFramesLen, self.NormedFeaturesLen,))(x1)
        t0DNN = Flatten()(t0DNN)

        # y1 = state_input
        # tf0t0State =Lambda(lambda y1: y1[:,0,:,:], output_shape=(self.PastDataForStep,self.NormedFeaturesLen,))(y1)
        # ConvX = Conv2D(16, (3, self.InstrumentCount), activation='relu')(tf0t0State)
        # ConvX = MaxPooling2D(pool_size=(2, 1),strides=(2,2))(ConvX)
        # ConvX = Conv2D(8, (2, 1),  activation='relu')(ConvX)
        # ConvX = Dropout(0.5)(ConvX)
        # ConvX = Flatten()(ConvX)

        # x = keras.layers.concatenate([ConvX, xTime, t0DNN,old_prediction,DNNPrice])
        x = keras.layers.concatenate([xTime, t0DNN, DNNPrice])
        x = Dense(self.HIDDEN_SIZE, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.HIDDEN_SIZE, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.HIDDEN_SIZE, activation='relu')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input, time_input, price_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=self.LR), loss='mse')

        return model

    @nb.jit
    def LogEpisode(self, reward):
        self.TotalReward = self.TotalReward + np.array(reward).sum()
        summary = 'Licho;' + str(self.LichoID) + ";"
        summary = summary + str(self.episode) + ";"
        summary = summary + ' Sum in last period: ;' + str(np.array(reward).sum()) + ";"
        summary = summary + ' Min: ;' + str(np.array(reward).min()) + ";"
        summary = summary + ' Max: ;' + str(np.array(reward).max()) + ";"
        summary = summary + ' Long Count: ;' + str(self.env.LongCount) + ";"
        summary = summary + ' Short Count: ;' + str(self.env.ShortCount) + ";"
        summary = summary + ' Total reward: ;' + str(self.TotalReward) + ";"
        # summary=summary+' Total reward: '+str(np.array(self.reward_over_time).sum())+","
        summary = summary + str(self.env.Instruments) + "\n"
        print('Summary: ' + summary)

    # file = open('results/LichoResults.csv','a')
    # file.write(summary)
    # file.close()

    @nb.jit
    def transform_reward(self, breward):
        # print('Episode #', self.episode, '\tfinished with reward', breward.sum(),
        # 				'\tAverage reward of last 100 episode :', np.mean(self.reward_over_time[-100:]))
        self.reward_over_time.append(breward.sum())
        # self.writer.add_scalar('Episode reward', np.array(breward).sum(), self.episode)
        rewardLen = len(breward)
        nreward = np.zeros((rewardLen))

        for j in range(rewardLen):
            nreward[j] = breward[j]
            for k in range(j + 1, rewardLen):
                nreward[j] += breward[k] * self.GAMMA ** k

        return nreward

    @nb.jit
    def Act(self, Time, predicted_action, observation, CurrentPrice):
        # pred_values = self.critic.predict([observation,Time,predicted_action,CurrentPrice])   # To Explore later if it worth to add it
        # advantage = pred_values
        # p = self.actor.predict([observation, advantage, predicted_action,Time,CurrentPrice])
        p = self.actor.predict([observation, self.DUMMY_VALUE, self.DUMMY_ACTION, Time, CurrentPrice])

        action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
        return action_matrix, p

    @nb.jit
    def get_batch(self):
        # reset variables before each batch
        batchObservation = np.zeros((self.BATCH_SIZE, self.TimeFramesLen, self.PastDataForStep, self.NormedFeaturesLen))
        batchTime = np.zeros((self.BATCH_SIZE, 123))
        batchreward = np.zeros((self.BATCH_SIZE))
        batchrewardfunction = np.zeros((self.BATCH_SIZE))
        bachCurrentPrice = np.zeros((self.BATCH_SIZE, self.InstrumentCount))
        batchAction = np.zeros((self.BATCH_SIZE, self.actionsSpaceLen))
        batchPredicted_action = np.zeros((self.BATCH_SIZE, self.actionsSpaceLen))
        # initial values
        predicted_action = np.zeros((1, self.actionsSpaceLen))
        reward = 0

        observation, Time, CurrentPrice, RealReward, RewardFunction, done = self.env.Next(
            np.zeros((self.actionsSpaceLen)))

        for li in range(self.BATCH_SIZE):
            batchObservation[li] = observation  # observation and action from the same t?
            batchTime[li] = Time
            bachCurrentPrice[li] = CurrentPrice

            action_matrix, predicted_action = self.Act(Time.reshape((1, 123)),
                                                       predicted_action,
                                                       observation.reshape((1, self.TimeFramesLen, self.PastDataForStep,
                                                                            self.NormedFeaturesLen)),
                                                       CurrentPrice.reshape((1, self.InstrumentCount)))

            batchAction[li] = action_matrix
            batchPredicted_action[li] = predicted_action

            observation, Time, CurrentPrice, RealReward, RewardFunction, done = self.env.Next(action_matrix)
            # print(observation)
            # print(self.critic.predict([observation.reshape((1,self.TimeFramesLen, self.PastDataForStep,self.NormedFeaturesLen)),Time.reshape((1,123)),CurrentPrice.reshape((1,3))]))
            batchreward[li] = RealReward
            batchrewardfunction[li] = RewardFunction

        return batchObservation, batchAction, batchPredicted_action, batchreward, batchTime.reshape(
            (self.BATCH_SIZE, 123)), bachCurrentPrice, batchrewardfunction

    @nb.jit
    def train(self):
        actor_loss = []
        critic_loss = []

        while self.episode < self.EPISODES:
            obs, action, old_prediction, reward, Time, CurrentPrice, RewardFunction = self.get_batch()
            self.LogEpisode(reward)
            TransformedReward = self.transform_reward(RewardFunction)
            pred_reward = self.critic.predict([obs, Time, CurrentPrice])
            # print(pred_reward)
            advantage = TransformedReward - pred_reward.squeeze()  # Why not only pred_reward?

            print('Training...')

            for e in range(self.EPOCHS):
                critic_loss.append(self.critic.train_on_batch([obs, Time, CurrentPrice], [TransformedReward]))
                # Model(inputs=[state_input, advantage, old_prediction, time_input,price_input,Indicators_Input], outputs=[out_actions])
                actor_loss.append(
                    self.actor.train_on_batch([obs, advantage, old_prediction, Time, CurrentPrice], [action]))

            self.writer.add_scalar('Actor loss', np.mean(actor_loss), self.episode)
            self.writer.add_scalar('Critic loss', np.mean(critic_loss), self.episode)
            self.writer.add_scalar('Critic reward', np.mean(pred_reward), self.episode)
            self.writer.add_scalar('Episode reward', np.mean(reward), self.episode)

            print('Saving')
            self.episode = self.episode + 1
            f = open('episode.licho', 'w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
            f.write(str(self.episode))
            f.close()
            self.actor.save_weights(self.modelName)
            self.critic.save_weights(self.CriticModelName)
            print('Saved')


if __name__ == "__main__":
    ID = 0
    ID = '1555507047.4657013'
    licho = Licho(ID)
    licho.train()

# add ARIMA, ARMA, SMA to norm Data

# Reward function
# Action AND matrix to Critic
# critic estimation to Actor
# Explore loss functions
# change probability to Log
# a gdzie poprawny gradient do PD?
