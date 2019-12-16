# autocheck
AI for Checkers
# Reference and model
https://github.com/plkmo/AlphaZero_Connect4
# Game Size
Will be varied through 6, 8, 10, 12, 14
# Contents:
```bash
checkers_Nnet.py: File for neural network
mcts.py: Implementing the MCTS and self play included in this file
checkers.py: All the game functions are in the file.
```
# Game Rules
The game rules of our modified Checkers follow the standard rules of Checkers but with two additional rules. 1. When a piece is jumped, it has a chance to remain on the board, unaffected. 2. Sequences of double-jumps are not allowed. 
# How to run
Run mcts.py.

IN the main function the evaluator() will call self_play to generate training data and use the data to train the network.
Uncomment checkers.print_board(board) to see the board after each action.

For personal play with random choice opponent run checkers.py

The MCTS_play_withRandom function: The tree(player) will play against a random choice opponent 
which always plays its turn by choosing an action uniformly at random
The evaluate function: Everything will combine together, starting with the search tree, train by the neural network, than update the data.
```
# Game Functions
```bash
def make_king(board,space): Switch a pawn(soldier) to a king
def isKing(board,space): Check if a piece is king
def expand_board(board):
def initial_board(nrows,ncols): Creates the initial state of the board, white is 1, black is 2, 
0 stands for the position a piece can be move to, 5 stands for the place where no pieces can be move to
def compress_board(board)
def get_state(board, player)
def get_state(board)
def get_board(state)
def switch_player(player): To switch player between white and black
def print_board(board): Represent the board in characters, W: white, B:black, 
x: position where no piece can be move to
def isBlacksquare(coords): 
def bounds_check(board,space)
def get_moves(board,space,player)
def get_all_moves(board,player)
def get_jumps(board,space,player)
def isBlackpiece(a), def isWhitepiece(a): Takes integer that represents the type of piece
Returns boolean - whether or not the piece is black (or white)
def isFriendly(a,player), def isEnemy(a,player): Check if a pirece is an enemy or not
def test_move(board,space,move,player)
def apply_move(board, space, move, player):Apply a move to the board
def isTerminal(board): The game is considered ended if one team is wiped out
def gameloop(nrows,ncols): Play function
```
![](https://github.com/jbot2000/autocheck/blob/master/initial_state1.png)
With the function initial_board, it generates a board with 3\*boardlength white pawns(soldiers) 
and black pawns(soldiers).

![](https://github.com/jbot2000/autocheck/blob/master/initial_state2.png)
The print_board function will give a visulaized 

# MCTS Functions
Tese functions will define all the node classes, and the functions to run the tree search.
```bash
class ParentRootNode(object): Define the root node
class Node(object): Includes all the functions needed to traverse the tree
def MCTS_Search(board, player, num_reads, n_net): Do MCST with neural network
def policy(node, temp=1): To calculate the policy
def MCTS_self_play(): The self play function
def MCTS_Play_WithRandom(nnet, num_games): To play with a random coice oponente returns win and draw 
def training(n_net, batch_size, n_epochs, learning_rate, dataset): To train the neural network
def evaluate(n_net, n_games): It calls self_play to generate traing data and calls training function returns the neural net
```
# CNN class
```bash
class Net(nn.Module):The CNN class with one concolution layer
class ErrorFnc(nn.Module): Use mean square error to calculate the loos function
```

