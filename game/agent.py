import torch
import random
import numpy as np
from game import _2048GameAI, Direction
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
# We can edit these values later but these are common


class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0  # control the randomness. represents tradeoff between exploration VS exploitation
        self.gamma = 0  # discount rate, adjust short-term and long-term rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # This should be self-explanatory...
        pass

    # TODO: helper functions that identify availability of left, up, down, right movements

    def get_state(self, game):
        # retrieve board state that has all current cells
        # report available moves (the more the better)
        # identify the max tile (higher the better)
        # identify # of empty cells (more the better)
        # compress the board into a 1D array for computations
        # finally concatenate the state onto np.array

    def remember(self, state, action, reward, next_state, done):
        # state-current state before taking action
        # action-action taken by agent in the given state (4 directions)
        # reward-reward received after taking action
        # next_state-state of env after action is taken
        # done-indication of game-over, or maybe 2048 is reached.
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def train_long_memory():
        pass

    def get_action(self, state):
        pass

def train(epochs, load_model=False):  # NOTE merges are treated like "scores" in snakeAI game
    plot_merges = []
    plot_max_tile = []
    total_merges = 0
    record = 0
    agent = Agent()  # agent = Agent(load_model=load_model) later

    dim = 4
    board = Board(dim)
    game = Game(board)
    player = _2048GameAI(board, game)
    player.play()

    while True:
        state_old = agent.get_state(game) # get old state
        final_move = agent.get_action(state_old) # calculate move based on old state
        reward, done, merges = game.handle_key_press(final_move) # Perform the move
        state_new = agent.get_state(game) # retrieve the new state, use for memory

        # training the short memory:
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember:
        agent.remember(state_old, final_move, reward, state_new, done)

        if done: # basically if the game is over. we reset game here
            # training the long memory:
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()
            if merges > record:
                record = merges

            # TODO: plotting merges (score) on plot
    pass

if __name__ == '__main__':
    train(epoch=100, load_model=True)
