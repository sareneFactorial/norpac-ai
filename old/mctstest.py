import newnn as neuralnet
import norpac
import copy
import math


def ucbScore(parent, child):
    priorScore = child.prior * math.sqrt(parent.visitCount) / (child.visitCount + 1)
    if child.visit_count > 0:
        valueScore = child.value()
    else:
        valueScore = 0

    if child.trainingPlayer != child.activePlayer:
        valueScore *= -1

    return valueScore + priorScore


class Node:
    def __init__(self, prior, activePlayer, trainingPlayer):
        self.visitCount = 0
        self.activePlayer = activePlayer
        self.prior = prior
        self.valueSum = 0
        self.children = {}
        self.game = None
        self.state = None

        self.trainingPlayer = trainingPlayer

    def value(self):
        return self.valueSum / self.visitCount

    def expand(self):
        pass


def run(state, activePlayer, numSimulations=100, root = None):

    pass