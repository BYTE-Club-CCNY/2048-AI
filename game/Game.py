import tkinter as tk
import colors as c


class Game:
    def __init__(self, board):
        self.board = board
        self.root = tk.Tk()
        self.root.title('2048')
        self.root.resizable(width=False, height=False)
        self.background = tk.Frame(self.root, bg=c.BACKGROUND_COLOR)
        self.cell_labels = []
        self.apply_labels()
        self.background.pack(side=tk.TOP)

    def apply_labels(self):
        for i in range(self.board.dim):
            row_labels = []
            for j in range(self.board.dim):
                label = tk.Label(self.background, text='',
                                 bg=c.EMPTY_CELL_COLOR,
                                 justify=tk.CENTER, font=c.FONT,
                                 width=4, height=2)
                label.grid(row=i, column=j, padx=10, pady=10)
                row_labels.append(label)
            self.cell_labels.append(row_labels)

    def draw(self):
        for i in range(self.board.dim):
            for j in range(self.board.dim):
                if self.board.cells[i][j] == 0:
                    self.cell_labels[i][j].configure(
                        text='',
                        bg=c.EMPTY_CELL_COLOR)
                else:
                    cell_text = str(self.board.cells[i][j])
                    if self.board.cells[i][j] > 2048:
                        bg_color = c.CELL_BACKGROUND_COLOR_DICT.get('beyond')
                        fg_color = c.CELL_COLOR_DICT.get('beyond')
                    else:
                        bg_color = c.CELL_BACKGROUND_COLOR_DICT.get(cell_text)
                        fg_color = c.CELL_COLOR_DICT.get(cell_text)
                    self.cell_labels[i][j].configure(
                        text=cell_text,
                        bg=bg_color, fg=fg_color)
