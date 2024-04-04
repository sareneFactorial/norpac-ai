# lots of this used this as reference: https://joshvarty.github.io/AlphaZero/
import copy
import math
import torch
from operator import add, truediv
import cProfile
import random
import pickle
import numpy
from datetime import datetime
import collections
import torch.optim as optim
import os


import newnorpac

EE_CONSTANT = 1
BUFFER_SIZE = 5000
NUM_SIM = 300

LEARN_RATE = 6e-5

checkpointsDir = os.path.join(os.path.dirname(__file__), "checkpoints/")


def ucbScore(parent, child):
    global EE_CONSTANT
    pass
    return list(map(add, child.averageValue(), [EE_CONSTANT * math.sqrt(math.log(parent.visits+1) / (child.visits + 1))]*6))


class Node:
    state: newnorpac.NorpacGame

    def __init__(self, prior, player, actionID, state: newnorpac.NorpacGame):
        self.prior = prior
        self.player = player

        self.value = [0, 0, 0, 0, 0, 0]
        self.visits = 0
        self.children = []
        self.state = state

        self.expanded = False

        # the action ID from newnorpac.py
        self.actionID = actionID

    def averageValue(self):
        if self.visits == 0:
            return [0, 0, 0, 0, 0, 0]
        return list(map(truediv, self.value, [self.visits] * 6))

    def expand(self, policyModel):
        priors = policyModel.output(self.state.createInput(self.player))
        for i in self.state.allLegalMoves(self.player):
            newState = self.state.doAction(self.state.currentPlayer, i)
            self.children.append(Node(priors[i], newState[0].currentPlayer, i, newState[0]))
        self.expanded = True


class Tree:
    def __init__(self, game, valueModel, policyModel):
        self.game = game
        self.valueModel = valueModel
        self.policyModel = policyModel
        self.currentRoot = None

    def runSimulations(self, numSimulations, player, root=None):
        if root is None and self.currentRoot is None:
            self.currentRoot = Node(0, player, -1, self.game)
            self.currentRoot.expand(self.policyModel)
            self.currentRoot.visits += 1
        if root is None:
            root = self.currentRoot

        for ii in range(numSimulations):
            # if ii % 100 == 0:
            #     print(f"on simulation {ii}")

            node = root
            searchPath = [node]

            while node.expanded:
                # TODO: find a better way to do this, like kotlin's max function, i hate for loops
                maxScore = -99999999
                maxIndex = -1
                for n in node.children:
                    score = ucbScore(node, n)
                    score = score[node.state.players.index(node.player)]
                    if score > maxScore:
                        maxScore = score
                        maxIndex = node.children.index(n)
                if maxIndex == -1:  # TODO: is this correct?
                    break
                node = node.children[maxIndex]
                searchPath.append(node)

            value = self.valueModel.output(node.state.createInput(node.player)).tolist()

            if node.state.terminalState:
                    scores = [(i, node.state.points.get(i, 0), node.state.badInvestments.get(i, 0),
                               node.state.playerOrder.index(i)) for i in node.state.players]
                    scores.sort(key=lambda a: (
                        -a[1], a[2], a[3]))  # sort by points (descending) and bad investments (ascending)
                    winner = scores[0]
                    value = newnorpac.hotOne(scores.index(winner), 6)
            else:
                node.expand(self.policyModel)

            for n in searchPath:
                n.value = list(map(add, value, n.value))
                n.visits += 1

        return root


class PolicyNetwork(torch.nn.Module):
    def __init__(self, greedy=False, rand=False):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1217, 250),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(250, 200),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(200, 100)
        )

        self.greedy = greedy
        self.random = rand

    def output(self, x):
        logits = self.linear_relu_stack(torch.tensor(x))
        return logits


class ValueNetwork(torch.nn.Module):
    def __init__(self, greedy=False, rand=False):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1217, 200),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(200, 100),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(100, 6)
        )

        self.greedy = greedy
        self.random = rand

    def output(self, x):
        logits = self.linear_relu_stack(torch.tensor(x))
        return logits


def test():
    criterion = torch.nn.SmoothL1Loss()
    valueNetwork = ValueNetwork()
    policyNetwork = PolicyNetwork()
    valueOptimizer = optim.AdamW(valueNetwork.parameters(), lr=LEARN_RATE, amsgrad=True)
    policyOptimizer = optim.AdamW(policyNetwork.parameters(), lr=LEARN_RATE, amsgrad=True)
    players = [newnorpac.Player(), newnorpac.Player(), newnorpac.Player()]
    tree = Tree(newnorpac.newGame(players), valueNetwork, policyNetwork)
    policyTraining = collections.deque(maxlen=BUFFER_SIZE)
    valueTraining = collections.deque(maxlen=BUFFER_SIZE)

    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')

    for generation in range(0, 1):
        botWins = 0
        games = 0
        print(f"Generation {generation}:")
        for g in range(0, 3):
            games += 1
            players = []
            # players = [newnorpac.Player()] * random.randrange(3, 6+1)
            for i in range(0, random.randrange(3, 6)):
                players.append(newnorpac.Player())
            tree.game = newnorpac.newGame(players)
            node = Node(-1, tree.game.currentPlayer, -1, tree.game)
            node = tree.runSimulations(1, tree.game.currentPlayer, node)
            originalRoot = node
            nodeHistory = [node]

            _actingPlayer = random.choice(tree.game.players)
            actingPlayerIndex = tree.game.players.index(_actingPlayer)
            # print(f"\n\nacting player is {actingPlayerIndex}\n")
            if generation == 0 or generation == 1:
                print(f"game {g}...")

            while not node.state.terminalState:
                if not node.expanded:
                    node.expand(tree.policyModel)

                if node.state.players.index(node.state.currentPlayer) == actingPlayerIndex:
                    node = tree.runSimulations(NUM_SIM, tree.game.currentPlayer, node)
                    tree.currentRoot = node
                    priorDict = {}
                    priorValues = []
                    for child in node.children:
                        priorDict[child.actionID] = child.visits
                    for i in range(0, 100):
                        priorValues.append(priorDict.get(i, 0))
                    priorValues = [x / sum(priorValues) for x in priorValues]
                    policyTraining.append((node.state.createInput(node.state.currentPlayer), priorValues))
                    move = numpy.argmax(priorValues)
                    if g == 0:
                        print("BOT: " + newnorpac.readOutput(move))
                    node = next(x for x in node.children if x.actionID == move)
                    nodeHistory.append(node)
                    if not node.expanded:
                        node.expand(tree.policyModel)
                else:
                    # TODO: introduce more than fully random actors
                    player = node.state.currentPlayer
                    moves = node.state.allLegalMoves(player)
                    move = random.choice(moves)
                    if g == 0:
                        print(F"Player {node.state.players.index(node.player)}: " + newnorpac.readOutput(move))
                    if not node.expanded:
                        node.expand(tree.policyModel)
                    node = next(x for x in node.children if x.actionID == move) # , default_value)
                    nodeHistory.append(node)
                    if not node.expanded:
                        node.expand(tree.policyModel)

            scores = [(i, node.state.points.get(i, 0), node.state.badInvestments.get(i, 0), node.state.playerOrder.index(i)) for i in node.state.players]
            scores.sort(key=lambda a: (
                -a[1], a[2], a[3]))  # sort by points (descending) and bad investments (ascending)
            winner = scores[0]
            if node.state.players.index(winner[0]) == actingPlayerIndex:
                botWins += 1
            value = newnorpac.hotOne(scores.index(winner), 6)
            for n in nodeHistory:
                valueTraining.append((n.state.createInput(n.player), value))
            nodeHistory.clear()

        print(f"Games done. Winrate: {(botWins/games)*100:.3f}%. Training...")

        for i in range(0, 10):
            # valueDist = random.choices(valueTraining, weights=tdErrors, k=round(len(valueTraining) * 0.4))
            # TODO: add weighting
            valueData = random.choices(valueTraining, k=round(len(valueTraining) * 0.4))
            policyData = random.choices(policyTraining, k=round(len(policyTraining) * 0.4))

            valueBatch = [list(it) for it in zip(*valueData)]
            valueOutput = valueNetwork.output(valueBatch[0])#.gather(1, response.unsqueeze(0))

            valueLoss = criterion(valueOutput, torch.tensor(valueBatch[1]))
            # Optimize the model
            valueOptimizer.zero_grad()
            valueLoss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(valueNetwork.parameters(), 100)
            valueOptimizer.step()
            # print("---")
            # print(valueLoss)
            # print(criterion(valueNetwork.output(valueBatch[0]), torch.tensor(valueBatch[1])))
            # print("---\n")

            # policy time
            policyBatch = [list(it) for it in zip(*policyData)]
            policyOutput = policyNetwork.output(policyBatch[0])
            policyLoss = criterion(policyOutput, torch.tensor(policyBatch[1]))
            policyOptimizer.zero_grad()
            policyLoss.backward()
            torch.nn.utils.clip_grad_value_(policyNetwork.parameters(), 100)
            policyOptimizer.step()

        if generation % 10 == 0 and generation != 0:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
            filename = f"checkpoints/{timestamp}-mcts-{generation}.pkl"
            with open(os.path.join(checkpointsDir, filename), "wb") as f:
                torch.save(actingNetwork, f)
            filename = f"{timestamp}-mcts-{generation}-trainingdata.pkl"
            with open(os.path.join(checkpointsDir, filename), "wb") as f:
                pickle.dump((policyData, valueData), f)
            print(f"Generation done. Saved generation to {filename}.")
        else:
            print(f"Generation done.")

        print("Full average loss over training data:")
        valueBatch = [list(it) for it in zip(*valueTraining)]
        valueOutput = valueNetwork.output(valueBatch[0])
        valueLoss = criterion(valueOutput, torch.tensor(valueBatch[1]))
        print(f"Value: {valueLoss}")
        policyBatch = [list(it) for it in zip(*policyTraining)]
        policyOutput = policyNetwork.output(policyBatch[0])
        policyLoss = criterion(policyOutput, torch.tensor(policyBatch[1]))
        print(f"Policy: {policyLoss}")

    filename = f"{timestamp}-mcts-{generation}-policy.pkl"
    with open(os.path.join(checkpointsDir, filename), "wb") as f:
        torch.save(policyNetwork, f)
    filename = f"{timestamp}-mcts-{generation}-value.pkl"
    with open(os.path.join(checkpointsDir, filename), "wb") as f:
        torch.save(valueNetwork, f)

    print(f"Training done. Saved generation to {filename}.")


test()

# cProfile.run("test()", sort='cumtime')
