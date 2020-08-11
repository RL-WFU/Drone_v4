import numpy as np

class Map:
    def __init__(self, numRows, numCols):
        self.rows = numRows
        self.cols = numCols
        self.map = np.zeros([self.rows, self.cols])
        self.search_map = np.zeros([self.rows, self.cols])
        self.sight_distance = 2


    def reset_map(self):
        self.map = np.zeros([self.rows, self.cols])

    def reset_search_map(self):
        self.search_map = np.zeros([self.rows, self.cols])

    def update_map(self, image, row, col, search=False):
        for i in range(self.sight_distance*2):
            for j in range(self.sight_distance*2):
                self.map[row + i - self.sight_distance, col + j - self.sight_distance] = image[i, j]
                if search:
                    self.search_map[row + i - self.sight_distance, col + j - self.sight_distance] = image[i, j]