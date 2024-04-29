from game.config.Board import Board
from game.config.Game import Game
from game.config import colors as c
import tkinter.messagebox as messagebox


class Player:
    def __init__(self, board: Board, game: Game):
        self.board = board
        self.game = game
        self.start_cells_num = 2
        self.game_over = False
        self.won = False
        self.keep_playing = False
        self.frame_iteration = 0
        self.max_block = 1

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
    def handle_key_press(self, event, action):
        self.frame_iteration += 1
        reward = 0
        if self.is_game_over():
            reward = -15
            return reward
        # perhaps we can also return a value that represents highest block that we were able to reach?
        self.board.remove_checks()
        key_value = event.keysym

        if key_value in c.UP_KEYS:
            self.up()
        elif key_value in c.DOWN_KEYS:
            self.down()
        elif key_value in c.LEFT_KEYS:
            self.left()
        elif key_value in c.RIGHT_KEYS:
            self.right()
        else:
            pass

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
