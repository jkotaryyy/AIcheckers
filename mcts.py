import numpy as np
import math
import collections
import checkers
import checkers_Nnet as Nnet
import pickle
from tqdm import tqdm
import torch
import time
from torch.utils.data import Dataset
import matplotlib.pylab as plt

# Exploration constant c is defined as C_input
C_input = 1
# The Larges number of possible moves form at any point in the game
Large_move = 20
board_size = 6
roll_out = 100


class ParentRootNode(object):
    def __init__(self):
        self.parent = None
        self.child_number_visits = collections.defaultdict(float)
        self.child_simulation_reward = collections.defaultdict(float)


class Node(object):
    def __init__(self, board, possible_moves, player, move=None, parent=None):
        self.board = board
        self.player = player
        self.is_expanded = False
        self.move = move   # index of the move that resulted in current Node/State
        self.possible_moves = possible_moves
        self.illegal_moves = Large_move - len(possible_moves)
        self.parent = parent
        self.child_prior_probability = np.zeros([Large_move], dtype=np.float32)
        self.child_number_visits = np.zeros([Large_move], dtype=np.float32)
        self.child_simulation_reward = np.zeros([Large_move], dtype=np.float32)
        self.children = {}

    @property
    def N(self):
        return self.parent.child_number_visits[self.move]

    @N.setter
    def N(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def R(self):
        return self.parent.child_simulation_reward[self.move]

    @R.setter
    def R(self, value):
        self.parent.child_simulation_reward[self.move] = value

    @property
    def Q(self):
        return self.R() / (1 + self.N())

    def child_Q(self):
        return self.child_simulation_reward/(1 + self.child_number_visits)

    def child_U(self):
        return C_input * math.sqrt(self.N)*abs(self.child_prior_probability)/(1 + self.child_number_visits)

    def child_score(self):
        return self.child_Q() + self.child_U()

    def best_child(self):
        return np.argmax(self.child_number_visits + self.child_score()/100)

    def select_leaf(self):
        current = self
        while current.is_expanded:
            current = current.maybe_add_child(np.argmax(current.child_score()[0:(len(current.possible_moves))]))
        return current

    def maybe_add_child(self, move):
        # print(move)
        # print("Possible moves")
        # print(len(self.possible_moves))
        if len(self.possible_moves) == move:
            print(self.child_score())
            print(move)
            print("Possible moves")
            print(len(self.possible_moves))
            print(self.child_prior_probability)
        if move not in self.children:
            new_board = checkers.apply_move(self.board, self.possible_moves[move][0],
                                            self.possible_moves[move][1], self.player)
            player2 = checkers.switch_player(self.player)
            if self.is_board_in_MCTS(new_board, player2):
                m = self.child_score()
                m[move] = m.min()-1
                move = np.argmax(m[0:(len(self.possible_moves))])
                new_board = checkers.apply_move(self.board, self.possible_moves[move][0],
                                                self.possible_moves[move][1], self.player)
                player2 = checkers.switch_player(self.player)
            self.children[move] = Node(new_board, checkers.get_all_moves(new_board, player2),
                                       player2, move=move, parent=self)
        #checkers.print_board(self.children[move].board)
        return self.children[move]

    def inject_noise(self):
        dirch = np.random.dirichlet([0.3]*len(self.possible_moves))
        self.child_prior_probability[0: len(self.possible_moves)] = \
            self.child_prior_probability[0: len(self.possible_moves)]*0.75 + dirch*0.25


    def is_board_in_MCTS(self, board, player):
        current = self
        while current.parent is not None:
            if np.array_equal(current.board, board) and current.player == player:
                return True
            current = current.parent
        return False

    def backpropagate(self, value):
        current_node = self
        while current_node.parent is not None:
            current_node.N += 1
            current_node.R += value
            value *= -1
            current_node = current_node.parent

    def expand_and_evaluate(self, child_pr):
        self.is_expanded = True
        for i in range(len(self.possible_moves), len(child_pr)):
            child_pr[i] = 0
        scale = child_pr.sum()
        self.child_prior_probability = child_pr/scale

    def print_tree(self):
        print("The Number of visits of next possible states")
        print(self.child_number_visits)
        print("The Reward of each next possible states")
        print(self.child_simulation_reward)
        print("The prior probability of next possible states")
        print(self.child_prior_probability)
        print("The possible moves already expanded")
        print(self.children)


def MCTS_Search(board, player, num_reads, n_net):
    root = Node(board, checkers.get_all_moves(board, player), player, move=None, parent=ParentRootNode())
    # root.inject_noise()
    for i in range(num_reads):
        leaf = root.select_leaf()
        player = checkers.switch_player(player)
        child_prior_prob, value = n_net(torch.FloatTensor(checkers.get_state2(leaf.board, leaf.player)))
        # print(child_prior_prob)
        # print("The number of reads", i)
        if checkers.isTerminal(leaf.board, leaf.player) or checkers.get_all_moves(leaf.board, leaf.player) == []:
            # print("Finished Game")
            leaf.backpropagate(value)
            # leaf.print_tree()
        else:
            child_prior_prob = child_prior_prob.cpu().detach().numpy().reshape(-1)
            leaf.expand_and_evaluate(child_prior_prob)
            leaf.backpropagate(value)

    # root.print_tree()
    return root


def get_policy(node, temp=1):
    return (node.child_number_visits**(1/temp))/sum(node.child_number_visits**(1/temp))


def MCTS_self_play(nnet, num_games, s_index, iteration):
    data_x = []
    for itt in tqdm(range(s_index, num_games + s_index)):
        board = checkers.initial_board(board_size, board_size)
        #board = checkers.initial_b6()
        player = 1
        data = []
        value = 0
        num_moves = 0
        t = 1
        while checkers.isTerminal(board, player) is not True:
            # if num_moves > 15:
            #     t = 0.1
            root = MCTS_Search(board, player, roll_out, nnet)
            # print("The turn of player {:d} and Moves {:d}".format(player, num_moves))
            # checkers.print_board(root.board)
            policy = get_policy(root, t)
            data.append([board, player, policy])
            move = np.argmax(policy)
            board = checkers.apply_move(root.board, root.possible_moves[move][0], root.possible_moves[move][1],
                                        root.player)
            player = checkers.switch_player(player)
            if len(checkers.get_all_moves(board, player)) == 0:
                # Player == 1 means White pieces
                # print("Game Finished")
                if player == 1:
                    value = -1
                elif player == 2:
                    value = 1
                else:
                    value = 0
                break
            if num_moves == 150:
                value = 0
                break
            num_moves += 1

        for ind, dx in enumerate(data):
            s, pl, po = dx
            if ind == 0:
                data_x.append([checkers.get_state2(s, pl), po, 0])
            else:
                data_x.append([checkers.get_state2(s, pl), po, value])
        del data
        # filename = "MCTS_iteration-{:d}_game-{:d}.p".format(iteration, itt)
        # save_data(filename, data_x)
    return data_x


def MCTS_Play_WithRandom(nnet, num_games):
    number_of_wins = 0
    number_of_draws = 0
    for itt in tqdm(range(num_games)):
        board = checkers.initial_board(board_size, board_size)
        # board = checkers.initial_b6()
        player = 1
        num_moves = 0
        t = 1
        while checkers.isTerminal(board, player) is not True:
            # if num_moves > 15:
            #     t = 0.1
            if player == 1:
                root = MCTS_Search(board, player, roll_out, nnet)
                policy = get_policy(root, t)
                move = np.argmax(policy)
                board = checkers.apply_move(root.board, root.possible_moves[move][0], root.possible_moves[move][1],
                                            root.player)
            else:
                move = checkers.get_random_move(board, player)
                board = checkers.apply_move(board, move[0], move[1], player)
            # print("The turn of player {:d} and Moves {:d}".format(player, num_moves))
            # checkers.print_board(board)

            player = checkers.switch_player(player)
            if len(checkers.get_all_moves(board, player)) == 0:
                # Player == 1 means White pieces
                # print("Game Finished")
                if player == 2:
                    number_of_wins += 1
                break
            if num_moves == 200:
                number_of_draws += 1
                break
            num_moves += 1
    return number_of_wins, number_of_draws


def training(n_net, batch_size, n_epochs, learning_rate, dataset):
    n_net.train()
    criteria = Nnet.ErrorFnc()
    train_set = TrainingData(dataset)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(n_net.parameters(), lr=learning_rate )
    t_time = time.time()
    update_size = len(train_loader)
    loss_data = np.zeros(n_epochs, dtype=np.float )
    for epoch in range(n_epochs):
        total_loss = 0

        for i, data in enumerate(train_loader, 0):
            state, policy, value = data
            optimizer.zero_grad()
            policy_estimate, value_estimate = n_net(state.float())
            loss = criteria(value_estimate, value.float(),  policy_estimate,  policy.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # print("Training Finished, Epoch-{:d} total loss-{:.2f} and took-{:.2f}s"
        #       .format(epoch, total_loss/update_size, time.time() - t_time))
        loss_data[epoch] = total_loss/update_size
    return loss_data


def save_data(name, data):
    data1 = open(name, 'wb')
    pickle.dump(data, data1, protocol=pickle.HIGHEST_PROTOCOL)
    data1.close()


def load_data(name):
    c_name = name + ".p"
    data1 = open(c_name, 'rb')
    return pickle.load(data1)


class TrainingData(Dataset):
    def __init__(self, data_set):
        d = np.array(data_set)
        self.a = d[:, 0]
        self.b = d[:, 1]
        self.c = d[:, 2]

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return np.int64(self.a[i]), self.b[i], self.c[i]

# board_n = checkers.initial_board(8, 8)
# player_n = 1
# possible_moves_n = checkers.get_all_moves(board_n, player_n)
# print(possible_moves_n)
# a = Node(board_n, possible_moves_n, player_n, ParentRootNode())
# MCTS_Search(board_n, player_n, 2000, Nnet.Net())


# d = MCTS_self_play(Nnet.Net(), 1, 0, 1)
# training(Nnet.Net(), 1, 50, 0.001, d, 0)


def evaluate(n_net, n_games):
    variation = 3
    net1 = n_net.Net()
    learning_rate = [0.001,  0.01,  0.1]
    print("Started Self Play")
    data = MCTS_self_play(net1, n_games, 0, 1)
    print("Finished Self Play")
    nnet = [net1, net1, net1]
    for i in range(variation):
        print("Started Training for parameter-{:d}".format(i))
        loss = training(nnet[i], 1, 50, learning_rate[i], data)
        plt.figure()
        plt.title('Learning Curve')
        plt.plot(loss, label="lr={:3f}".format(learning_rate[i]))
        plt.ylabel('Error')
        plt.xlabel('Iteration')
        plt.show()
        print("Finished Training for parameter-{:d}".format(i))
    return nnet[0]

# evaluate(Nnet, 25)


if __name__ == "__main__":
    # board size should be even 6 8 10 12 14
    board_size = 6
    Nnet.board_size = board_size
    roll_out = 50
    num_games = 2
    # Each self play game generates roughly
    # 40 samples board size 6
    # 90 samples for board size 8

    net = evaluate(Nnet, num_games)
    play_with_trained = True
    games = 10
    if play_with_trained:
        # To play with random opponent with trained
        print("Started Playing with random opponent")
        win, draw = MCTS_Play_WithRandom(net, games)
        print("Play using trained with random, the number of wins-{:d}, draws-{:d}- total_games-{:d}".format(win, draw, games))
    # else:
        # To play with random opponent with untrained
        print("Started Playing with random opponent")
        win, draw = MCTS_Play_WithRandom(Nnet.Net(), games)
        print("Played using untrained with random , the number of wins-{:d}, draws-{:d}- total_games-{:d}".format(win, draw, games))















