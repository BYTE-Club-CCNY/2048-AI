import tkinter as tk
import random


class Board:
    def __init__(self, dim):
        self.dim = dim
        self.cells = self.generate_empty_board()
        self.slided = False
        self.moved = False
        self.combined = False
        self.current_score = 0

    def generate_random_cell(self):
        cell = random.choice(self.get_empty_cells())
        i = cell[0]
        j = cell[1]
        self.cells[i][j] = 2 if random.random() < 0.9 else 4

    def get_empty_cells(self):
        ret = []
        for i in range(self.dim):
            for j in range(self.dim):
                if self.cells[i][j] == 0:
                    ret.append((i, j))
        return ret

    def generate_empty_board(self):
        return [[0] * self.dim for i in range(self.dim)]

    def transpose(self):
        """
        This function is used to swap the rows and columns.
        With this in addition to the reverse function, we will
        simplify the code. If we wish to move up we will only
        need to transpose the grid and apply the merges, but if
        we want to move down, we need to apply the transpose and
        reverse the grid.
        """
        self.cells = [list(t) for t in zip(*self.cells)]

    def reverse(self):
        """
        This is used to reverse the order of the cells in each row.
        With this in addition to the transpose function, we will be able to implement the movement functions.
        """
        for i in range(self.dim):
            start = 0
            end = self.dim - 1
            while start < end:
                self.cells[i][start], self.cells[i][end] = \
                    self.cells[i][end], self.cells[i][start]
                start += 1
                end -= 1

    def remove_checks(self):
        self.slided = False
        self.combined = False
        self.moved = False

    def slide_cells(self):
        self.slided = False
        new_board = self.generate_empty_board()
        for i in range(self.dim):
            count = 0
            for j in range(self.dim):
                if self.cells[i][j] != 0:
                    new_board[i][count] = self.cells[i][j]
                    if count != j:
                        self.slided = True
                    count += 1
        self.cells = new_board

    def combine_cells(self):
        self.combined = False
        for i in range(self.dim):
            for j in range(self.dim - 1):
                if self.cells[i][j] == self.cells[i][j + 1] and self.cells[i][j] != 0:
                    self.cells[i][j] *= 2
                    self.cells[i][j + 1] = 0
                    self.current_score += self.cells[i][j]
                    self.combined = True

    def found_2048(self):
        for i in range(self.dim):
            for j in range(self.dim):
                if self.cells[i][j] >= 2048:
                    return True
        return False

    def has_empty_cells(self):
        for i in range(self.dim):
            for j in range(self.dim):
                if self.cells[i][j] == 0:
                    return True
        return False

    def mergeable(self):
        for i in range(self.dim):
            for j in range(self.dim - 1):
                if self.cells[i][j] == self.cells[i][j + 1]:
                    return True
        for j in range(self.dim):
            for i in range(self.dim - 1):
                if self.cells[i][j] == self.cells[i + 1][j]:
                    return True
        return False
