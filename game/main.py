from game.config.Board import Board
from Player import Player
from game.config.Game import Game


def main():
    dim = 4
    board = Board(dim)
    game = Game(board)
    player = Player(board, game)
    player.play()


if __name__ == '__main__':
    main()
