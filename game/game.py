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
    def __init__(self, board: Board, game: Game):
        self.board = board
        self.game = game
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
        self.game.root.bind('<Key>', self.handle_key_press)
        # TODO: Edit above bind. no more key pressing, only AI control
        self.game.root.mainloop()

    def add_start_cells(self):
        for i in range(self.start_cells_num):
            self.board.generate_random_cell()

    def can_move(self):
        return self.board.has_empty_cells() or self.board.mergeable()

    # Determine action after key press
    def handle_key_press(self, action):
        # actions: [up, down, left, right]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP, Direction.NONE]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0, 0, 0]):  # Action is slide right
            new_dir = clock_wise[0]
        elif np.array_equal(action, [0, 1, 0, 0, 0]):  # Action is slide down
            new_dir = clock_wise[1]
        elif np.array_equal(action, [0, 0, 1, 0, 0]): # Action is slide left
            new_dir = clock_wise[2]
        else: # Action is slide up
            new_dir = clock_wise[3]
        

        self.frame_iteration += 1
        reward = 0
        if self.is_game_over():
            reward = -15
            return reward
        # perhaps we can also return a value that represents highest block that we were able to reach?
        self.board.remove_checks()

        if self.direction == Direction.UP:
            self.up()
        elif self.direction == Direction.DOWN:
            self.down()
        elif self.direction == Direction.LEFT:
            self.left()
        elif self.direction == Direction.RIGHT:
            self.right()

        self.game.draw()
        if self.board.found_2048():
            pass
            # we want to note down that we have reached 2048 for stats.

        if self.board.moved:
            self.board.generate_random_cell()
            reward = 5

        self.game.draw()
        if not self.can_move():
            self.game_over = True
            # self.game_over_message() Not needed anymore
            self.reset()

    def game_win_message(self):
        if not self.won:
            self.won = True
            self.keep_playing = True

# def game_over_message(self):
    # messagebox.showinfo('2048', 'Game Over!')
    # Above function is no longer needed, the game will auto-reset and AI plays again

    def reset(self):
        # Reset the board state
        self.board.reset()
        # Reset the game state and any necessary UI components
        self.game.reset()
        # Reset the player's game state variables
        self.start_cells_num = 2
        self.game_over = False
        self.won = False
        self.keep_playing = False

    # Player Actions left right up down
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
