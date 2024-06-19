from config.Board import Board
from config.GameConfig import Game
from enum import Enum
import numpy as np


class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    NONE = 5


class _2048GameAI:
    def __init__(self, board: Board, game: Game, agent=None):
        self.board = board
        self.game = game
        self.agent = agent
        self.start_cells_num = 2
        self.game_over = False
        self.won = False
        self.keep_playing = False
        self.frame_iteration = 0
        self.max_block = 1
        self.direction = Direction.NONE

    def is_game_over(self):
        return self.game_over or (self.won and not self.keep_playing)

    def play(self):
        self.add_start_cells()
        self.game.draw()
        self.game.root.after(100, self.play_step_loop)
        self.game.root.mainloop()
        # self.game.root.after(100, self.play_step_loop)

    def play_step_loop(self):
        if not self.is_game_over():
            action = self.get_next_action()
            reward, done, frame_iteration = self.play_step(action)
            self.game.root.after(100, self.play_step_loop)
        else:
            pass

    def play_step(self, action):
        self._move(action)
        self.frame_iteration += 1
        reward = 0
        if self.is_game_over():
            reward = -15
            return reward, self.is_game_over(), self.frame_iteration

        self.game.draw()
        if self.board.found_2048():
            self.game_win_message()

        if self.board.moved:
            self.board.generate_random_cell()
            reward = 5

        self.game.draw()
        if not self.can_move():
            self.game_over = True
            self.reset()

        return reward, self.is_game_over(), self.frame_iteration

    def get_next_action(self):
        state_old = self.agent.get_state(self)
        final_move = self.agent.get_action(state_old)
        return final_move

    def add_start_cells(self):
        for i in range(self.start_cells_num):
            self.board.generate_random_cell()

    def can_move(self):
        return self.board.has_empty_cells() or self.board.mergeable()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP, Direction.NONE]
        if np.array_equal(action, [1, 0, 0, 0]):
            new_dir = clock_wise[0]
        elif np.array_equal(action, [0, 1, 0, 0]):
            new_dir = clock_wise[1]
        elif np.array_equal(action, [0, 0, 1, 0]):
            new_dir = clock_wise[2]
        elif np.array_equal(action, [0, 0, 0, 1]):
            new_dir = clock_wise[3]
        else:
            return

        if new_dir == Direction.UP:
            self.up()
        elif new_dir == Direction.DOWN:
            self.down()
        elif new_dir == Direction.LEFT:
            self.left()
        elif new_dir == Direction.RIGHT:
            self.right()

    def game_win_message(self):
        if not self.won:
            self.won = True
            self.keep_playing = True

    def reset(self):
        self.board.reset()
        self.game.reset()
        self.start_cells_num = 2
        self.game_over = False
        self.won = False
        self.keep_playing = False

    def left(self):
        self.board.slide_cells()
        self.board.combine_cells()
        self.board.moved = self.board.slided or self.board.combined
        self.board.slide_cells()

    def right(self):
        self.board.reverse()
        self.left()
        self.board.reverse()

    def up(self):
        self.board.transpose()
        self.left()
        self.board.transpose()

    def down(self):
        self.board.transpose()
        self.board.reverse()
        self.left()
        self.board.reverse()
        self.board.transpose()
