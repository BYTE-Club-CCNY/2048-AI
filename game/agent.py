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


class Agent:
    def __init__(self, board: Board, game: _2048GameAI):
        self.board = board
        self.game = game
        self.num_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(22, 256, 4)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game):
        board_state = np.array(self.board.cells).flatten()
        available_moves = np.array([
            self.can_move_right(),
            self.can_move_down(),
            self.can_move_left(),
            self.can_move_up()
        ], dtype=int)
        max_tile = np.max(self.board.cells)
        empty_cells = len(self.board.get_empty_cells())
        state = np.concatenate([
            board_state,
            available_moves,
            [max_tile, empty_cells]
        ])
        return state

    def can_move_left(self):
        original_cells = self.board.cells.copy()
        self.board.slide_cells()
        self.board.combine_cells()
        can_move = self.board.moved
        self.board.cells = original_cells
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
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state):
        self.epsilon = 80 - self.num_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state_0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train(epochs, load_model=False):
    plot_merges = []
    plot_max_tile = []
    total_merges = 0
    record = 0
    board = Board(4)
    game_ai = _2048GameAI(board, Game(board))
    agent = Agent(board=board, game=game_ai)
    game_ai.agent = agent
    print(999)

    for epoch in range(epochs):
        print(0)
        score = 0
        game_ai.play()
        print(1)
        while not game_ai.is_game_over():
            print(2)
            state_old = agent.get_state(game_ai)
            final_move = agent.get_action(state_old)
            reward, done, frame_iteration = game_ai.play_step(final_move)
            state_new = agent.get_state(game_ai)
            print(3)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                print(4)
                mean_score = total_merges / agent.num_games
                agent.num_games += 1
                agent.train_long_memory()
                if frame_iteration > record:
                    record = frame_iteration
                score = frame_iteration
                plot_merges.append(score)
                plot_max_tile.append(mean_score)
                print(plot_merges)
                print(plot_max_tile)
                total_merges += score
                plot(plot_merges, plot_max_tile)
                game_ai.reset()
                print(5)

        if epoch % 10 == 0:
            agent.model.save(epoch)
            print(6)


if __name__ == '__main__':
    train(epochs=100, load_model=True)

