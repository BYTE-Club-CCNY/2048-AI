# 2048 Reinforcement Deep Q-Learning Project (BYTE x ACM)


## Todo:
### Player.py:
- ~~Add reward variable~~
- ~~Have the AI control the game through actions, instead of key-user input~~

### agent.py:
- Implementing Bellman Equation: # of games, epsilon, gamma
- Implementing both short and long term memory
- Epochs
- Save and load model function based on epochs
- Get state function to determine the next state
- Remember function, which appends new state to a deque of states
- Train on short and long term memory
- Give way to actions based on exploration and later exploitation
- Basic train function

### model.py: (*This part is pretty trivial once the agent is fully designed*)
- Complete initialization of a Qnet module subclass
- Complete forward computation function 
- Save & loading model
- Training function:
  - Determine which variables need to be kept track of (e.g. state, action, reward, etc.)
  - Making sure all variables are converted to tensors for training
  - Making sure all torch tensors are un-squeezed to avoid dimension issues
  - Generate predicted Q-value for the current state, to create a target value for the next states
  - Use the Bellman Equation to compute the target itself BASED on current state
  - Calculate loss & implement optimizer function (should be easy)
  - Implement epoch function that saves all epochs and calculates highest-scoring epoch at the end
  
### plot.py:
- Scores are determined by max_block achieved, and (maybe) # of successful merges
- X-label: Number of games, Y-label: Score
- Calculate mean scores

## Miscellaneous Todo:
- Optimize all functions to decrease lines of code (maybe merging Board & Game to be 2 classes, but same python doc)
- implement animations for the game such as when tiles are moved.
- add sounds? (maybe)
- add documentation to all the functions and classes.
## Make sure to comment above each function on what it does and why we need it

