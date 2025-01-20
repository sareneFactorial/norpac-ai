# lots of this used this as reference: https://joshvarty.github.io/AlphaZero/
import math
import torch
from operator import add, truediv
import random
import numpy

import newnorpac

NUM_SIM = 300
EE_CONSTANT = 1


def ucbScore(parent, child):
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
        return list(map(truediv, self.value, [self.visits+1] * 6))

    def expand(self, policyModel):
        # global pool
        # priors = policyModel.output(self.state.createInput(self.player))

        # legal = self.state.allLegalMoves(self.player)
        # with multiprocessing.Pool() as pool:
        #     self.children = pool.starmap(expandLoop, [(self, i) for i in legal])
        for i in self.state.allLegalMoves(self.player):
            newState = self.state.doAction(self.state.currentPlayer, i)
            self.children.append(Node(0, newState[0].currentPlayer, i, newState[0]))
        self.expanded = True


class Tree:
    def __init__(self, game):
        self.game = game
        # self.valueModel = valueModel
        # self.policyModel = policyModel

    # TODO: make this static, remove Tree completely
    def runSimulations(self, numSimulations, root, valueModel, policyModel):
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

            value = valueModel.output(node.state.createInput(node.player)).tolist()

            if node.state.terminalState:
                scores = [(i, node.state.points.get(i, 0), node.state.badInvestments.get(i, 0),
                           node.state.playerOrder.index(i)) for i in node.state.players]
                scores.sort(key=lambda a: (
                    -a[1], a[2], a[3]))  # sort by points (descending) and bad investments (ascending)
                winner = scores[0]
                value = newnorpac.hotOne(scores.index(winner), 6)
            else:
                node.expand(policyModel)

            for n in searchPath:
                n.value = list(map(add, value, n.value))
                n.visits += 1

        return root


# TODO: move to actors.py
class MCTSBot:
    def __init__(self, game, policyModel, valueModel, isTraining=False):
        self.tree = Tree(game)
        self.currentNode = Node(-1, self.tree.game.currentPlayer, -1, self.tree.game)
        self.policyModel = policyModel
        self.valueModel = valueModel

        self.isTraining = isTraining

        self.nodeHistory = []
        self.policyTraining = []  # TODO: make this deque
        self.valueTraining = []

    def doAction(self, player: newnorpac.Player, game: newnorpac.NorpacGame) -> tuple[tuple[newnorpac.NorpacGame, str], int]:
        self.currentNode = self.tree.runSimulations(NUM_SIM, self.currentNode, self.valueModel, self.policyModel)
        priorDict = {}
        priorValues = []
        for child in self.currentNode.children:
            priorDict[child.actionID] = child.visits
        for i in range(0, 100):
            priorValues.append(priorDict.get(i, 0))
        priorValues = [x / sum(priorValues) for x in priorValues]
        self.policyTraining.append((self.currentNode.state.createInput(self.currentNode.state.currentPlayer), priorValues))
        move = numpy.argmax(priorValues)
        self.currentNode = next(x for x in self.currentNode.children if x.actionID == move)
        if self.isTraining:
            self.nodeHistory.append(self.currentNode)
        if not self.currentNode.expanded:
            self.currentNode.expand(self.policyModel)
        return (self.currentNode.state, "bot action TODO: fix this text"), self.currentNode.actionID

    def externalAction(self, player, move):
        if player.actor == self:  # skip if action is own
            return
        if not self.currentNode.expanded:
            self.currentNode.expand(self.policyModel)
        self.currentNode = next(x for x in self.currentNode.children if x.actionID == move)
        if self.isTraining:
            self.nodeHistory.append(self.currentNode)
        if not self.currentNode.expanded:
            self.currentNode.expand(self.policyModel)

    def reportWin(self, winner):
        # print("aaaaa")
        if not self.isTraining:
            return
        value = newnorpac.hotOne(winner[3], 6)
        for n in self.nodeHistory:
            self.valueTraining.append((n.state.createInput(n.player), value))
        self.nodeHistory.clear()

    def newGame(self, game):
        players = []
        # players = [newnorpac.Player()] * random.randrange(3, 6+1)
        for i in range(0, random.randrange(3, 6)):
            players.append(newnorpac.Player())
        self.tree.game = game
        self.currentNode = Node(-1, self.tree.game.currentPlayer, -1, self.tree.game)
        self.currentNode = self.tree.runSimulations(1, self.currentNode, self.valueModel, self.policyModel)
        self.nodeHistory = [self.currentNode]

    def trainNetworks(self, n):
        # print(self.valueTraining)
        # print(self.policyTraining)
        for i in range(0, n):
            # TODO: add weighting
            self.valueModel.train_(self.valueTraining)
            self.policyModel.train_(self.policyTraining)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1217, 250),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(250, 200),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(200, 100)
        )

        self.criterion = criterion

    def output(self, x):
        logits = self.linear_relu_stack(torch.tensor(x))
        return logits

    def train_(self, trainingData):
        data = random.choices(trainingData, k=round(len(trainingData) * 0.4))
        batch = [list(it) for it in zip(*data)]
        output = self.output(batch[0])
        loss = self.criterion(output, torch.tensor(batch[1]))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimizer.step()


class ValueNetwork(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1217, 200),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(200, 100),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(100, 6)
        )

        self.criterion = criterion

    def output(self, x):
        logits = self.linear_relu_stack(torch.tensor(x))
        return logits

    def train_(self, trainingData):
        data = random.choices(trainingData, k=round(len(trainingData) * 0.4))
        batch = [list(it) for it in zip(*data)]
        output = self.output(batch[0])
        loss = self.criterion(output, torch.tensor(batch[1]))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimizer.step()