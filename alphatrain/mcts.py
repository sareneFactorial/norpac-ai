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
import actors

EE_CONSTANT = 1
BUFFER_SIZE = 10000
NUM_SIM = 5

LEARN_RATE = 6e-5

GAMES_PER_GEN = 1
NUM_GENS = 20

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
    def __init__(self, game):
        self.game = game
        # self.valueModel = valueModel
        # self.policyModel = policyModel

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
        self.policyTraining = []
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
            self.currentNode.expand(tree.policyModel)
        return (self.currentNode.state, "bot action TODO: fix this text"), self.currentNode.actionID

    def externalAction(self, player, move):
        if player == self.currentNode.player:  # skip if action is already in tree
            return
        if not self.currentNode.expanded:
            self.currentNode.expand(self.policyModel)
        # TODO: thjis fucks up right here fix this
        self.currentNode = next(x for x in self.currentNode.children if x.actionID == move)
        if self.isTraining:
            self.nodeHistory.append(self.currentNode)
        if not self.currentNode.expanded:
            self.currentNode.expand(self.policyModel)

    def reportWin(self, winner):
        print("aaaaa")
        if not self.isTraining:
            return
        value = newnorpac.hotOne(scores.index(winner), 6)
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
        self.nodeHistory = [node]

    def trainNetworks(self, n):
        print(self.valueTraining)
        print(self.policyTraining)
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
        loss = self.criterion(output, torch.tensor(policyBatch[1]))
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
        loss = self.criterion(output, torch.tensor(valueBatch[1]))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimizer.step()


def test():
    criterion = torch.nn.SmoothL1Loss()

    valueNetwork = ValueNetwork(criterion)
    valueOptimizer = optim.AdamW(valueNetwork.parameters(), lr=LEARN_RATE, amsgrad=True)
    valueNetwork.optimizer = valueOptimizer
    valueTraining = collections.deque(maxlen=BUFFER_SIZE)

    policyNetwork = PolicyNetwork(criterion)
    policyOptimizer = optim.AdamW(policyNetwork.parameters(), lr=LEARN_RATE, amsgrad=True)
    policyNetwork.optimizer = policyOptimizer
    policyTraining = collections.deque(maxlen=BUFFER_SIZE)

    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')

    # null game because of the dependency hell i've created with Tree
    nullGame = newnorpac.newGame([newnorpac.Player(actors.RandomAI()), newnorpac.Player(actors.RandomAI()), newnorpac.Player(actors.RandomAI())])

    trainingBot = MCTSBot(nullGame, policyNetwork, valueNetwork, True)
    trainingPlayer = newnorpac.Player(trainingBot)

    guaranteedActors = [
        trainingBot
    ]

    actorsList = [
        actors.RandomAI(),
        actors.RandomAI(),
        actors.RandomAI(),
        actors.RandomAI(),
        actors.RandomAI(),
        actors.RandomAI(),
    ]

    for generation in range(0, NUM_GENS):
        botWins = 0
        games = 0
        print(f"Generation {generation}:")

        # TODO: self-play against previous versions
        for g in range(0, GAMES_PER_GEN):
            games += 1
            players = []
            for i in range(0, random.randrange(3, 5)):
                players.append(newnorpac.Player(actors.RandomAI()))
            players.append(trainingPlayer)

            currentGame = newnorpac.newGame(players)

            # _actingPlayer = random.choice(tree.game.players)
            # actingPlayerIndex = tree.game.players.index(_actingPlayer)

            # run game
            while not currentGame.terminalState:
                # neural network player
                player = currentGame.currentPlayer
                tuple1, moveNum = player.actor.doAction(player, currentGame)
                currentGame, _ = tuple1
                print(newnorpac.readOutput(moveNum))
                print(player)
                print(player.actor)
                print()
                for i in players:
                    if hasattr(i.actor, 'externalAction'):
                        i.actor.externalAction(player, moveNum)

            # check winner
            scores = [(i, currentGame.points.get(i, 0), currentGame.badInvestments.get(i, 0), currentGame.playerOrder.index(i)) for i in currentGame.players]
            scores.sort(key=lambda a: (
                -a[1], a[2], a[3]))  # sort by points (descending) and bad investments (ascending)
            winner = scores[0]
            if winner[0] == trainingPlayer:
                botWins += 1
            for i in players:
                if hasattr(i.actor, 'reportWin'):
                    i.actor.reportWin(winner[0])

        print(f"Games done. Winrate: {(botWins/games)*100:.3f}%. Training...")

        # training policy & value networks
        trainingBot.trainNetworks(10)
        # TODO: also train rainbow dqn along with this for benchmark

        if generation % 10 == 0 and generation != 0:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
            filename = f"checkpoints/{timestamp}-mcts-{generation}.pkl"
            with open(os.path.join(checkpointsDir, filename), "wb") as f:
                torch.save(actingNetwork, f)
            print(f"Generation done. Saved generation to {filename}.")
        else:
            print(f"Generation done.")

        filename = f"{timestamp}-mcts-{generation}-trainingdata.pkl"
        with open(os.path.join(checkpointsDir, filename), "wb") as f:
            pickle.dump((policyData, valueData), f)

        print("Full average loss over training data:")
        valueBatch = [list(it) for it in zip(*valueTraining)]
        valueOutput = valueNetwork.output(valueBatch[0])
        valueLoss = criterion(valueOutput, torch.tensor(valueBatch[1]))
        print(f"Value: {valueLoss}")
        policyBatch = [list(it) for it in zip(*policyTraining)]
        policyOutput = policyNetwork.output(policyBatch[0])
        policyLoss = criterion(policyOutput, torch.tensor(policyBatch[1]))
        print(f"Policy: {policyLoss}")

    # at the end
    filename = f"{timestamp}-mcts-final-policy.pkl"
    with open(os.path.join(checkpointsDir, filename), "wb") as f:
        torch.save(policyNetwork, f)
    filename = f"{timestamp}-mcts-final-value.pkl"
    with open(os.path.join(checkpointsDir, filename), "wb") as f:
        torch.save(valueNetwork, f)

    print(f"Training done. Saved generation to {filename}.")


test()

# cProfile.run("test()", sort='cumtime')
