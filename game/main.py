from config.Board import Board
from game import _2048GameAI, Direction
from config.GameConfig import Game


def main():
    dim = 4
    board = Board(dim)
    game = Game(board)
    player = _2048GameAI(board, game)
    player.play()


if __name__ == '__main__':
    main()
