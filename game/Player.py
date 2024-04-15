from Board import Board
from Game import Game
import colors as c
import tkinter.messagebox as messagebox


class Player:
    def __init__(self, board: Board, game: Game):
        self.board = board
        self.game = game
        self.start_cells_num = 2
        self.game_over = False
        self.won = False
        self.keep_playing = False

    def is_game_over(self):
        return self.game_over or (self.won and not self.keep_playing)

    def play(self):
        self.add_start_cells()
        self.game.draw()
        self.game.root.bind('<Key>', self.handle_key_press)
        self.game.root.mainloop()

    def add_start_cells(self):
        for i in range(self.start_cells_num):
            self.board.generate_random_cell()

    def can_move(self):
        return self.board.has_empty_cells() or self.board.mergeable()

    def handle_key_press(self, event):
        if self.is_game_over():
            return
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
            self.game_win_message()
            if not self.keep_playing:
                return

        if self.board.moved:
            self.board.generate_random_cell()

        self.game.draw()
        if not self.can_move():
            self.game_over = True
            self.game_over_message()

    def game_win_message(self):
        if not self.won:
            self.won = True
            if messagebox.askyesno('2048', 'You Win!\n'
                                           'Are you going to continue the 2048 game?'):
                self.keep_playing = True

    def game_over_message(self):
        messagebox.showinfo('2048', 'Game Over!')

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
