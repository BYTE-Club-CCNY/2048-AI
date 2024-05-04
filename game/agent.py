import torch
import random
from config.Board import Board
from config.GameConfig import Game
import numpy as np
from game import _2048GameAI, Direction
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
# We can edit these values later but these are common


class Agent:
    def __init__(self, board: Board, game: _2048GameAI):
        self.board = board
        self.game = game
        self.num_games = 0
        self.epsilon = 0  # control the randomness. represents tradeoff between exploration VS exploitation
        self.gamma = 0.9  # discount rate, adjust short-term and long-term rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # This should be self-explanatory...
        self.model = Linear_QNet(3,256,4)  # input size, hidden layer size, output size
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)


    # TODO: helper functions that identify availability of left, up, down, right movements

    def get_state(self, game):
        # retrieve board state that has all current cells (1D array)
        board_state = np.array(self.board.cells).flatten()
        # report available moves (the more the better)
        available_moves = np.array([
            self.can_move_right(),
            self.can_move_down(),
            self.can_move_left(),
            self.can_move_up()
        ], dtype=int)
        # identify the max tile (higher the better)
        max_tile = np.max(self.board.cells)
        # identify # of empty cells (more the better)
        empty_cells = len(self.board.get_empty_cells())
        # finally concatenate the state onto np.array
        state = np.concatenate([
            board_state,
            available_moves,
            [max_tile, empty_cells]
        ])
        return state

    def can_move_left(self):
        # Temporarily simulate a left move
        original_cells = self.board.cells.copy()
        self.board.slide_cells()
        self.board.combine_cells()
        can_move = self.board.moved
        self.board.cells = original_cells  # Restore original state
        return can_move

    def can_move_right(self):
        self.board.reverse()
        can_move = self.can_move_left()
        self.board.reverse()
        return can_move

    def can_move_up(self):
        self.board.transpose()
        can_move = self.can_move_left()
        self.board.transpose()
        return can_move

    def can_move_down(self):
        self.board.transpose()
        self.board.reverse()
        can_move = self.can_move_left()
        self.board.reverse()
        self.board.transpose()
        return can_move

    def remember(self, state, action, reward, next_state, done):
        # state-current state before taking action
        # action-action taken by agent in the given state (4 directions)
        # reward-reward received after taking action
        # next_state-state of env after action is taken
        # done-indication of game-over, or maybe 2048 is reached.
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # we have to unzip the mini sample since it is a long list of tuples. we need to extract each individual value

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state):
        # actions are either exploration or exploitation (which comes after exploration usually)
        self.epsilon = 80 - self.num_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)  # array goes from 0 to 3, so need to make move based on this
            final_move[move] = 1
        # the more games we have, the less randomness is involved, meaning lower epsilon each game
        # this eventually brings us to exploitation rather than exploration
        else:
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1  # prediction with exploitation once we are not exploring anymore
        return final_move


def train(epochs, board, _game, load_model=False ):  # NOTE merges are treated like "scores" in snakeAI game
    plot_merges = []
    plot_max_tile = []
    total_merges = 0
    record = 0
    game = _2048GameAI(board, _game)
    agent = Agent(board=board, game=_game)  # agent = Agent(load_model=load_model) later

    while True:
        state_old = agent.get_state(game)  # get old state
        final_move = agent.get_action(state_old)  # calculate move based on old state
        reward, done, merges = game.handle_key_press(final_move)  # Perform the move
        state_new = agent.get_state(game)  # retrieve the new state, use for memory

        # training the short memory:
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember:
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:  # basically if the game is over. we reset game here
            # training the long memory:
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()
            if merges > record:
                record = merges

            # TODO: plotting merges (score) on plot
            plot_merges.append(merges)
            total_score += merges
            # mean_score = total_score / agent.num_games
            plot_max_tile.append(record)
            plot(plot_merges, plot_max_tile)

if __name__ == '__main__':
    dim = 4
    board = Board(dim)
    game = Game(board)
    player = _2048GameAI(board, game)
    train(epochs=100, load_model=True, board=board, _game=game)
