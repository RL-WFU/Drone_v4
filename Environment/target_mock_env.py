from Environment.base_env import *


class SelectTargetTrain(Env):
    def __init__(self):
        # Set simulation
        self.set_simulation_map()

        # Define initial targets
        self.__class__.current_target_index = 0
        self.current_target = self.targets[self.__class__.current_target_index]

        # Set task-specific parameters
        self.num_actions = 9
        self.region_values = np.zeros([9, 3])

        # Set reward values
        self.MINING_REWARD = 100
        self.DISTANCE_PENALTY = -2
        self.COVERED_PENALTY = -1200
        self.HOVER_PENALTY = -100

    def reset_env(self):
        # Initialize tracking
        self.__class__.map = np.zeros([self.totalRows, self.totalCols])
        self.__class__.visited = np.ones([self.totalRows, self.totalCols])

        # Reset env parameters
        self.start_row = random.randint(12, 47)
        self.start_col = random.randint(12, 47)
        self.__class__.current_target_index = 0
        self.__class__.row_position = self.start_row
        self.__class__.col_position = self.start_col

        for i in range(len(self.region_values)):
            self.region_values[i, 0] = 0
            self.region_values[i, 1] = 0


    def set_target(self, next_target):
        self.current_target = self.targets[next_target]

        reward = self.get_reward(next_target)

        self.__class__.current_target_index = next_target



        self.__class__.row_position = np.random.randint(low=self.current_target[0] - self.totalRows / 6,
                                                        high=self.current_target[0] + self.totalRows / 6)
        self.__class__.col_position = np.random.randint(low=self.current_target[1] - self.totalRows / 6,
                                                        high=self.current_target[1] + self.totalRows / 6)

        self.update_regions()

        state = self.region_values.reshape(1, 27)
        state = np.asarray(state)
        append = np.zeros(shape=[1, 1])
        append[0, 0] = self.current_target_index
        state = np.append(state, append, axis=1)

        return next_target, state, reward

    def get_reward(self, next_target):
        hover = False
        if next_target == self.__class__.current_target_index:
            hover = True

        reward = self.region_values[next_target, 0]*self.MINING_REWARD + self.region_values[next_target, 1]*self.COVERED_PENALTY + \
            self.region_values[next_target, 2]*self.DISTANCE_PENALTY

        return reward

    def update_regions(self):

        self.region_values[:, 1] = self.get_covered()
        self.region_values[:, 0] = self.get_mining()
        self.region_values[:, 2] = self.get_distance()

    def get_mining(self):
        mining = self.region_values[:, 0]

        covered = self.region_values[self.current_target_index, 1]

        random = np.random.uniform(low=(covered - .2 if covered > .2 else 0), high=(covered + .2 if covered < .8 else 1.0))

        mining[self.current_target_index] = random

        return mining

    def get_covered(self):
        covered = self.region_values[:, 1]
        random = np.random.uniform(low=0, high=1.0 - covered[self.current_target_index])
        covered[self.current_target_index] += random

        return covered

    def get_distance(self):
        distance = np.zeros(9)
        for i in range(9):
            distance[i] = math.sqrt((self.__class__.row_position - self.targets[i][0])**2 +
                                    (self.__class__.col_position - self.targets[i][1])**2)

        return distance

    def select_next_target(self, row, col):
        self.__class__.row_position = row
        self.__class__.col_position = col
        self.update_regions()
        next_targets = np.zeros(9)
        for i in range(9):
            next_targets[i] = self.region_values[i, 0]*self.MINING_REWARD + self.region_values[i, 1]*self.COVERED_PENALTY + \
                self.region_values[i, 2]*self.DISTANCE_PENALTY
            if i == self.current_target_index:
                next_targets[i] += self.HOVER_PENALTY

        # print(self.region_values)
        return np.argmax(next_targets)



    def simple_select(self):
        next_target = self.current_target_index + 1
        return next_target


