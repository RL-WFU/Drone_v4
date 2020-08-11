import numpy as np


class Visited:
    def __init__(self, numRows, numCols):
        self.rows = numRows
        self.cols = numCols
        self.visited = np.ones([numRows, numCols])
        self.search_visited = np.ones([numRows, numCols])

    def reset_visited(self):
        self.visited = np.ones([self.rows, self.cols])

    def reset_search_visited(self):
        self.search_visited = np.ones([self.rows, self.cols])

    def visited_position(self, row, col, search=False):
        self.visited[row, col] = 0
        if search:
            self.search_visited[row, col] = 0
