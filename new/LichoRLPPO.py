# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

import time

import numba as nb
import numpy as np
import tensorflow
# from tensorboardX import SummaryWriter
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

EPISODES = 99999  # ile razy cała pentla

LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 1500
NOISE = 0.95  # Exploration noise
GAMMA = 0.99

BATCH_SIZE = 1000
NUM_ACTIONS = 3
# NUM_STATE = 42
NUM_STATE = 23
HIDDEN_SIZE = 200
NUM_LAYERS = 2
ENTROPY_LOSS = 1e-3
LR = 1e-3  # Lower lr stabilises training greatly

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))


@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1 - b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob / (old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING,
                                                       max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * (
                               prob * K.log(prob + 1e-10)))

    return loss


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
    def __init__(self, Rycheza, ID=0):
        self.observation = 0
        self.critic = self.build_critic()
        self.actor = self.build_actor_continuous()

        self.env = Rycheza
        self.episode = 0
        self.gradient_steps = 0
        self.TotalReward = 0
        self.val = False
        self.reward_over_time = []
        # self.writer = SummaryWriter('AllRuns/continuous')

        if (ID == 0):
            self.ID = int(time.time())
            self.ActorName = 'klony/Licho' + str(self.ID) + '.h5'
            self.CriticName = 'klony/L_Critic' + str(self.ID) + '.h5'
        else:
            self.ID = ID
            self.ActorName = 'klony/Licho' + str(self.ID) + '.h5'
            self.actor.load_weights(self.ActorName)
            self.CriticName = 'klony/L_Critic' + str(self.ID) + '.h5'
            self.critic.load_weights(self.CriticName)

            print('Licho model loaded')

    def build_actor_continuous(self):
        state_input = Input(shape=(NUM_STATE,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(HIDDEN_SIZE, activation='relu')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='relu')(x)

        out_actions = Dense(NUM_ACTIONS, name='output', activation='tanh')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss_continuous(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        # model.summary()

        return model

    def build_critic(self):

        state_input = Input(shape=(NUM_STATE,))
        x = Dense(HIDDEN_SIZE, activation='relu')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='relu')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')

        return model

    def reset_env(self):
        self.episode += 1

    @nb.jit
    def get_action_continuous(self):
        # print(self.observation)
        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])
        if self.val is False:
            action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
        else:
            action = action_matrix = p[0]

        return action, action_matrix, p

    @nb.jit
    def transform_reward(self, reward):
        for j in range(len(reward) - 2, -1, -1):
            reward[j] += reward[j + 1] * GAMMA

        return reward

    @nb.jit
    def get_batch(self):
        # batch = [[], [], [], []]

        tmp_batch = [[], [], [], []]
        done = 0
        while done == 0:
            # print('AAAAAAAAAAAAAAA')
            action, action_matrix, predicted_action = self.get_action_continuous()
            observation, reward, done, info = self.env.step(action)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            tmp_batch[3].append(reward)
            self.observation = observation

            if done:
                self.reset_env()

        # print(done)

        reward = self.transform_reward(np.array(tmp_batch[3]))
        obs, action, pred, reward = np.array(tmp_batch[0]), np.array(tmp_batch[1]), np.array(tmp_batch[2]), reward
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        while self.episode < EPISODES:
            obs, action, pred, reward = self.get_batch()
            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values.flatten()
            # advantage = (advantage - advantage.mean()) / advantage.std()

            callback_early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='loss',
                                                                               patience=1, verbose=2)
            callback_reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                                              factor=0.1,
                                                                              min_lr=1e-4,
                                                                              patience=1,
                                                                              verbose=2)
            callback_tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir='../Licho v3/AllRuns/Licho', histogram_freq=1,
                                                                          write_graph=True)
            callbacks = [callback_early_stopping,
                         callback_reduce_lr]

            actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True,
                                        epochs=EPOCHS, verbose=False, callbacks=callbacks)
            critic_loss = self.critic.fit([obs], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS,
                                          verbose=False)

            # self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
            # self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)
            # self.writer.add_scalar('Critic reward', np.mean(pred_values), self.gradient_steps)
            # self.writer.add_scalar('Episode reward', np.mean(reward), self.gradient_steps)
            self.LogEpisode(reward)
            self.actor.save_weights(self.ActorName)
            self.critic.save_weights(self.CriticName)
            print('Model saved: Licho ID: ' + str(self.ID))

            self.gradient_steps += 1

    @nb.jit
    def LogEpisode(self, reward):
        self.TotalReward = self.TotalReward + np.array(reward).sum()
        summary = 'Licho ID: ' + str(self.ID) + "; "
        summary = summary + str(self.gradient_steps) + ";"
        summary = summary + ' Sum in last period: ;' + str(np.array(reward).sum()) + ";"
        summary = summary + ' Min: ;' + str(np.array(reward).min()) + ";"
        summary = summary + ' Max: ;' + str(np.array(reward).max()) + ";"
        summary = summary + ' Total reward: ;' + str(self.TotalReward) + ";"
        print(summary)
        # file = open('results/LichoResults.csv','a')
        # file.write(summary)
        # file.close()


if __name__ == '__main__':
    ag = Licho()

    ag.run()
