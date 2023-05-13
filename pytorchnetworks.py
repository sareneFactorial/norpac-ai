import random

import numpy as np
import torch
import norpac


def hotOne(h, n):
    return [1 if i == h else 0 for i in range(0, n)]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, greedy=False, rand=False):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1217, 500),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(500, 200),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(200, 100)
        )

        self.greedy = greedy
        self.random = rand

    def output(self, x):
        logits = self.linear_relu_stack(torch.tensor(x))
        return logits

    # game logic stuff
    def createInput(self, game: norpac.NorpacGame):
        return createInput(self, game)

    def doAction(self, game, weights, loud=False, weightsAreActions=False):
        return doAction(self, game, weights, loud, weightsAreActions)

    def doSingleAction(self, game, index):
        return self.doAction(game, [index], weightsAreActions=True)

    def firstLegal(self, game, weights):
        return firstLegal(self, game, weights)


# TODO: figure out how to do polymorphism in python
# code heavily referenced from https://github.com/Curt-Park/rainbow-is-all-you-need
class DuelingNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.greedy = False
        self.random = False

        self.feature_layer = torch.nn.Sequential(
            torch.nn.Linear(1217, 500),
            torch.nn.LeakyReLU(negative_slope=0.1),
        )
        self.advantage_layer = torch.nn.Sequential(
            torch.nn.Linear(500, 200),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(200, 100),
        )
        self.value_layer = torch.nn.Sequential(
            torch.nn.Linear(500, 128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(128, 1),
        )

    def output(self, x) -> torch.Tensor:
        feature = self.feature_layer(torch.tensor(x))

        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q

    def createInput(self, game: norpac.NorpacGame):
        return createInput(self, game)

    def doAction(self, game, weights, loud=False, weightsAreActions=False):
        return doAction(self, game, weights, loud, weightsAreActions)

    def doSingleAction(self, game, index):
        return self.doAction(game, [index], weightsAreActions=True)

    # deprecated
    # TODO: decide if i want to remove usages of this & the other ones i moved out
    def firstLegal(self, game, weights):
        return firstLegal(self, game, weights)


# game state Glue here. the interface between the game and the networks
# here be Bad Code.
# TODO: move these? unsure

def allLegal(theAi, game):
    ai = game.findAI(theAi)
    legal = []
    for i in range(0,100):
        if i < 50:
            if game.currentCity.name == norpac.allConnections[i][0]:
                if (norpac.allConnections[i][1], norpac.allConnections[i][0]) in game.trains:  # if double connection taken
                    continue
                legal.append(i)
                continue
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
                legal.append(i)
                continue
            # if even i.e. small cube
            if ai.howManySmall() > 0:
                legal.append(i)
                continue
            else:
                continue
        else:
            n = i - 96
            if game.currentCity.name == norpac.seattleConnections[n][0]:
                legal.append(i)
                continue
    return legal

# TODO: better variable names here lol
def firstLegal(theAi, game, weights):
    sortedout = weights.argsort().tolist()[::-1]
    ai = game.findAI(theAi)
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


# TODO: decouple the weights from this and make it just return false if it's illegal
# TODO: make greedy/random players their own class without the NN stuff, remove from here
def doAction(self, game, weights, loud=False, weightsAreActions=False):
    if weightsAreActions:
        sortedout = weights
    else:
        sortedout = weights.argsort().tolist()[::-1]

    if self.random:
        random.shuffle(sortedout)
    ai = game.findAI(self)
    if self.greedy:  # TODO: decouple greedy game logic
        for i in game.currentCity.connections:

            if game.findAI(self) in [x.owner for x in game.findCity(i).cubes]:
                st = "connected " + game.currentCity.name + " to " + i + " GREEDILY!"
                game.currentCity.connect(i)
                if loud: print(st)
                return st
        for i in game.cities:
            if i.name in ["Minneapolis", "Seattle"]:
                continue
            if len(ai.cubes) > 0 and 0 < len(i.cubes) < i.size and self not in list(set([x.owner for x in i.cubes])) and i not in game.getUnvisitableCities():
                if ai.hasBig():
                    i.cubes.append(norpac.Cube(ai, True))
                    ai.spendBig()
                    if loud: print("placed big cube on " + i.name + " with GREED.")
                    return "placed big cube on " + i.name + " with GREED."
                if ai.howManySmall() > 0:
                    i.cubes.append(norpac.Cube(ai, False))
                    ai.spendSmall()
                    if loud: print("placed small cube on " + i.name + " with GREED.")
                    return "placed small cube on " + i.name + " with GREED."

        cIndexShuffled = list(range(0, len(game.cities)))
        random.shuffle(cIndexShuffled)
        for i in cIndexShuffled:
            if game.cities[i].name in ["Minneapolis", "Seattle"]:
                continue
            if len(ai.cubes) > 0 and 0 < len(game.cities[i].cubes) < game.cities[i].size and game.cities[i] not in game.getUnvisitableCities():
                if ai.hasBig():
                    game.cities[i].cubes.append(norpac.Cube(ai, True))
                    ai.spendBig()
                    if loud: print("placed big cube on " + game.cities[i].name + " with GREED.")
                    return "placed big cube on " + game.cities[i].name + " with GREED."
                if ai.howManySmall() > 0:
                    game.cities[i].cubes.append(norpac.Cube(ai, False))
                    ai.spendSmall()
                    if loud: print("placed small cube on " + game.cities[i].name + " with GREED.")
                    return "placed small cube on " + game.cities[i].name + " with GREED."
        random.shuffle(weights)

    for i in sortedout:
        if self.greedy:
            weights[i] = -99.0
        if i < 50:
            if game.currentCity.name == norpac.allConnections[i][0]:
                if (norpac.allConnections[i][1], norpac.allConnections[i][0]) in game.trains:  # if double connection taken
                    continue
                game.currentCity.connect(norpac.allConnections[i][1])
                return f"connected {norpac.allConnections[i][0]} to {norpac.allConnections[i][1]} with confidence {weights[i] if not weightsAreActions else -99:2f}"
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
                city.cubes.append(norpac.Cube(ai, True))
                ai.spendBig()
                return f"placed big cube on {city.name} with confidence {weights[i] if not weightsAreActions else -99:.2f}"
            # if even i.e. small cube
            if ai.howManySmall() > 0:
                city.cubes.append(norpac.Cube(ai, False))
                ai.spendSmall()
                return f"placed small cube on {city.name} with confidence {weights[i] if not weightsAreActions else -99:.2f}"
            else:
                continue
        else:
            n = i - 96
            if game.currentCity.name == norpac.seattleConnections[n][0]:
                game.currentCity.connect(norpac.seattleConnections[n][1])
                return f"connected {norpac.seattleConnections[n][0]} to Seattle!!! with confidence {weights[i] if not weightsAreActions else -99:.2f}"
    return False


# TODO: clean this up a bit, include more documentation
def createInput(self, game: norpac.NorpacGame):
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
        if i is None or i.nn != self:
            a.append(0)
            continue
        a.append(1)
    for i in game.players + [None] * (6 - len(game.players)):
        if i is None:
            a.append(0)
            continue
        a.append(i.points / 20)
    a.extend(hotOne(game.roundNumber, 3))

    if len(a) != 1217:
        print("AAAAAAAAAAAA input length is messed up")
        print(len(a))
        raise Exception("input length is messed up idk why")
    return a