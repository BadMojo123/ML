import numpy as np
import pandas as pd
from gym.utils import seeding
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

MAX_OPEN_POSITIONS = 3


def calculate_reward(old_price, new_price, action_matrix, old_action, spread):
    if any(np.isnan(old_price)):
        print('WARNING old_price  is NAN')
    if any(np.isnan(new_price)):
        print('WARNING new_price  is NAN')
    if any(np.isnan(action_matrix)):
        print('WARNING action_matrix  is NAN')
    if any(np.isnan(old_action)):
        print('WARNING old_action  is NAN')
    if any(np.isnan(spread)):
        print('WARNING spread  is NAN')
    # print(f'old_action matrix = {action_matrix}')
    action_matrix = action_matrix[:-1]
    old_action = old_action[:-1]
    # print(f'nowa cena : {new_price} - {type(new_price[0])} - {type(new_price[1])}, stara cena : {old_price} - {type(old_price[0])} - {type(old_price[1])}')
    difference = new_price - old_price
    reward = difference * action_matrix
    action_change = np.abs(action_matrix - old_action, )
    spread_cost = action_change * spread
    if any(np.isnan(spread_cost)):
        print('WARNING Spread is NAN')
    if any(np.isnan(reward)):
        print('WARNING Reward is NAN')
    # print(f'Difference: {difference}')
    # print(f'Reward = {reward}')
    # print(f'Action change : {action_change}')
    # print(f'spredchange: {spread_cost}')
    return np.sum(reward) - np.sum(spread_cost)


def normalize_action(action):
    action = np.around(action * MAX_OPEN_POSITIONS)
    return action


class FastRunner(py_environment.PyEnvironment):
    def __init__(self):
        self._state = 0
        self._episode_ended = False
        self.batch_id = 0
        self.price = np.zeros(2)
        self.action_count = 3
        self.dummy_action = np.zeros(self.action_count)
        self.batch_start = 0
        self.batch_end = 0
        self.total_reward = 0

        link = 'https://raw.githubusercontent.com/BadMojo123/LichoOpen/master/Observations.csv'
        self.observation = pd.read_csv(link, sep=',', encoding='utf-8').to_numpy()
        self.observation = self.observation.astype(np.float32)
        self.batch_max, self.observation_count = self.observation.shape
        print(f'data read. Dimentions: {self.observation.shape}')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,), dtype=np.float32, minimum=-1, maximum=1, name='old_action')
        self._reward_spec = np.float32
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(39,), dtype=np.float32, minimum=-2, maximum=2, name='observation')

        # self._time_step_spec = tf_agents.trajectories.time_step.time_step_spec(self._observation_spec,
        #                                                                        self._reward_spec)
        self.old_action = self.dummy_action
        self._current_time_step = None

        # def time_step_spec(self):

    #     return self._time_step_spec

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def step(self, action):
        if self._current_time_step is None:
            return self.reset()
        self._current_time_step = self._step(action)
        return self._current_time_step

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        old_bach_id = self.batch_id

        action = normalize_action(action)
        self.batch_id = self.batch_id + 1

        if self.batch_id >= self.batch_max:
            observation = self.observation[old_bach_id, :-4]
            new_price = self.observation[old_bach_id, -2:]
            spread = self.observation[old_bach_id, -4:-2]
            reward = calculate_reward(self.price, new_price, action, self.old_action, spread)
            # print(f'New Price: {new_price}, old price = {self.price}, spread = {spread}, reward = {reward}')
            self.price = new_price
            self.old_action = action
            self.total_reward = self.total_reward + reward
            self._episode_ended = True
            return ts.termination(observation, reward=0)

        observation = self.observation[self.batch_id, :-4]
        new_price = self.observation[self.batch_id, -2:]
        if (any(np.isinf(new_price))):
            observation = self.observation[self.batch_id - 1, :-4]
            new_price = self.observation[self.batch_id - 1, -2:]
        spread = self.observation[self.batch_id, -4:-2]
        reward = calculate_reward(self.price, new_price, action, self.old_action, spread)
        # print(self.batch_id)
        # print(f'New Price: {new_price}, old price = {self.price}, spread = {spread}, reward = {reward}')
        self.price = new_price
        self.old_action = action
        self.total_reward = self.total_reward + reward

        return ts.transition(observation, reward=reward, discount=1.0)

    def reset(self):
        self._current_time_step = self._reset()
        return self._current_time_step

    def _seed(self, seed=None):
        self.np_random, self.seed_value = seeding.np_random(seed)
        return [self.seed_value]

    def _reset(self):

        self._seed()
        self.batch_start = int(self.np_random.uniform(low=0, high=1) * (self.batch_max - 1001))
        self.batch_end = self.batch_start + 1000
        print(f'Env restart.Starting with id = {self.batch_start} total reward : {self.total_reward}')
        self.total_reward = 0
        self._episode_ended = False
        self.batch_id = self.batch_start

        return ts.restart(self.observation[self.batch_id, :-4])

    # Run trouhg envirement with random actions
    def selfTest(self):
        print("START self test")
        i = 0
        for i in range(0, 70000):
            action = np.random.rand(self.action_count)
            Traj = self.step(action)
            if np.isnan(Traj.reward):
                print(Traj.reward)

            if any(np.isnan(Traj.observation)):
                print(Traj.observation)


def main():
    env = FastRunner()
    env.selfTest()


if __name__ == "__main__":
    # execute only if run as a script
    main()
