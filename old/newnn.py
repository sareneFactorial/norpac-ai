import numpy as np
import random
import norpac
from numba import vectorize, njit, jit
import copy


@vectorize
def clip(x, l, u):
    return max(min(x, u), l)


def sigmoid(x):
    s = clip(x, -500, 500)
    return 1 / (1 + np.exp(-s))


def sigmoidDerivative(x):
    s = sigmoid(x)
    return s * (1 - s)


@njit
def relu(x):
    return np.where(x > 0, x, x * 0.1)


@njit
def d_relu(x):
    return np.where(x > 0, 1, 0.1)


@njit
def linear(x):
    return x

@njit
def d_linear(x):
    return 1


def costFun(val, target):
    return 0.5 * (val - target) ** 2


def costFunDerivative(val, target):
    return val - target


SIGMA = 40

@njit
def huberLoss(val, target):
    x = clip(val, -1000, 1000)
    return np.where(abs(x) <= SIGMA, 0.5 * (val - target) ** 2, SIGMA * abs(val - target) - 0.5 * SIGMA**2)


@njit
def d_huberLoss(val, target):
    x = clip(val, -1000, 1000)
    return np.where(abs(x) <= SIGMA, (val - target), SIGMA * (np.sign(x)))


def hotOne(h, n):
    return [1 if i == h else 0 for i in range(0, n)]


class NeuralNet:

    def __init__(self, shape=[1217, 150, 100], biases=None, synapses=None, greedy=False, rand=False, activation=relu,
                 activationDerivative=d_relu, outputActivation=linear, outputActivationDerivative=d_linear):
        self.greedy = greedy
        self.random = rand
        self.shape = shape
        self.activation = activation
        self.activationDerivative = activationDerivative
        self.outputActivation = outputActivation
        self.outputActivationDerivative = outputActivationDerivative

        # Initialize weights for each layer
        if synapses is None:
            self.syn = []
            self.biases = []
            for i in range(len(shape) - 1):
                self.syn.append(0.002 * np.random.random((shape[i], shape[i + 1])) - 0.001)
                self.biases.append(np.zeros(shape[i + 1]))
        else:
            self.syn = synapses
            self.biases = biases

    def copy(self):
        return NeuralNet(self.shape, biases=self.biases.copy(), synapses=self.syn.copy(), activation=relu,
                         activationDerivative=d_relu)

    def output(self, l0):
        li = l0
        for i, s in enumerate(self.syn):
            if i == len(self.syn) - 1:
                li = self.outputActivation(np.dot(li, s) + self.biases[i])
            else:
                li = self.activation(np.dot(li, s) + self.biases[i])
        return li

    def Q_gradientDescent(self, inp, n, target, learnRate):
        activations = [inp]
        zs = []
        for i, s in enumerate(self.syn):
            zs.append(np.dot(activations[-1], s) + self.biases[i])
            activations.append(self.activation(zs[-1]))

        nodeValues = np.zeros_like(activations[-1])
        nodeValues[n] = d_huberLoss(activations[-1][n], target)
        self.syn[-1] -= learnRate * np.outer(activations[-2], nodeValues)
        self.biases[-1] -= learnRate * nodeValues

        for i in reversed(range(len(self.syn) - 1)):
            nodeValues = nodeValues.dot(self.syn[i + 1].T) * self.activationDerivative(zs[i])
            self.syn[i] -= learnRate * np.outer(activations[i], nodeValues)
            self.biases[i] -= learnRate * nodeValues

    def gradientDescent(self, inp, target, learnRate):
        activations = [inp]
        zs = []
        for i, s in enumerate(self.syn):
            zs.append(np.dot(activations[-1], s) + self.biases[i])
            activations.append(self.activation(zs[-1]))

        nodeValues = costFunDerivative(activations[-1], target) * self.outputActivationDerivative(zs[-1])
        self.syn[-1] -= learnRate * np.outer(activations[-2], nodeValues)
        self.biases[-1] -= learnRate * nodeValues

        for i in reversed(range(len(self.syn) - 1)):
            nodeValues = nodeValues.dot(self.syn[i + 1].T) * self.activationDerivative(zs[i])
            self.syn[i] -= learnRate * np.outer(activations[i], nodeValues)
            self.biases[i] -= learnRate * nodeValues

    def vary(self, factor):
        for i in range(0, len(self.syn)):
            self.syn[i] = self.syn[i] + (np.random.random(self.syn[i].shape) * factor)
        return self

    def firstLegal(self, game, weights):
        sortedout = weights.argsort().tolist()[::-1]
        ai = game.findAI(self)
        for i in sortedout:
            if i < 50:
                if game.currentCity.name == norpac.allConnections[i][0]:
                    if (norpac.allConnections[i][1], norpac.allConnections[i][0]) in game.trains:  # if double connection taken
                        continue
                    return i
            elif i < 96:
                j = (i - 50) // 2
                city = game.cities[j - 1]
                if len(city.cubes) >= city.size:  # if city full
                    continue
                if len(ai.cubes) <= 0:  # if ai have no cube :(
                    continue
                if city.name in list(sum(game.trains, ())):  # city connected to!! already
                    continue

                if i % 2 == 1:  # if odd i.e. big cube
                    if not ai.hasBig():
                        continue
                    return i
                # if even i.e. small cube
                if ai.howManySmall() > 0:
                    return i
                else:
                    continue
            else:
                n = i - 96
                if game.currentCity.name == norpac.seattleConnections[n][0]:
                    return i

    # deprecated
    # TODO: move all usages to game.allLegalMoves
    def allLegalMoves(self, game):
        ai = game.findAI(self)
        return game.allLegalMoves(ai)


def readOutput(n):
    if n < 50:
        conn = norpac.allConnections[n]
        return f"Conn {conn[0]} to {conn[1]}"
    elif n < 96:
        j = (n - 50) // 2
        city = norpac.cityIndices[j - 1]
        st = ""
        if n % 2 == 1:  # if odd i.e. big cube
            st += "Big "
        else:
            st += "Small "
        st += f"cube on {city}"
        return st
    else:
        n = n - 96
        return f"Connect {norpac.seattleConnections[n][0]} to Seattle!!!"


def createInput(player, game: norpac.NorpacGame):
    """Create an input layer for a model."""
    a = []
    cities = game.cities[1:len(game.cities) - 1]
    for i in cities:
        for j in i.cubes + [None] * (4 - len(i.cubes)):
            if j is None:
                a.extend([0, 0] * 6)
                continue
            for k in game.players + [None] * (6 - len(game.players)):
                if k is None or j.owner != k:
                    a.extend([0, 0])
                    continue
                a.extend(hotOne(1 if j.big else 0, 2))
    for i in norpac.allConnections:
        a.append(1 if i in game.trains else 0)
    for i in game.players + [None] * (6 - len(game.players)):
        if i is None:
            a.extend([0] * 6)
            continue
        a.extend(hotOne(game.playerOrder.index(i), 6))
    for i in game.players + [None] * (6 - len(game.players)):
        if i is None:
            a.extend([0, 0])
            continue
        n = i.howManySmall()
        a.append(n/20)
        for j in i.cubes:
            if j.big:
                a.append(1)
                break
        else:
            a.append(0)
    for i in game.players + [None] * (6 - len(game.players)):
        if i is None or i.nn != player:
            a.append(0)
            continue
        a.append(1)
    for i in game.players + [None] * (6 - len(game.players)):
        if i is None:
            a.append(0)
            continue
        a.append(i.points / 20)
    a.extend(hotOne(game.roundNumber, 3))

    if len(a) != player.shape[0]:
        print("AAAAAAAAAAAA input length is messed up")
        print(len(a))
        raise Exception("input length is messed up idk why")
    return a
