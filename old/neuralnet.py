import numpy as np
import random
import norpac


def sigmoid(x):
    s = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-s))


def sigmoidDerivative(x):
    return s * (1 - s)


def costFun(val, target):
    return 0.5 * (val - target) ** 2


def costFunDerivative(val, target):
    return val - target


def hotOne(h, n):
    return [1 if i == h else 0 for i in range(0, n)]


class NeuralNet:

    def __init__(self, syn0=None, syn1=None, syn2=None, greedy=False, rand=False):
        self.syn0 = syn0 if syn0 is not None else 2 * np.random.random((1332, 250)) - 1
        self.syn1 = syn1 if syn1 is not None else 2 * np.random.random((250, 250)) - 1
        self.syn2 = syn2 if syn2 is not None else 2 * np.random.random((250, 100)) - 1
        self.greedy = greedy
        self.random = rand
        self.layers = []  # TODO: n amount of layers

    def copy(self):
        return NeuralNet(syn0=np.copy(self.syn0), syn1=np.copy(self.syn1), syn2=np.copy(self.syn2))

    def output(self, l0):
        l1 = sigmoid(np.dot(l0, self.syn0))
        l2 = sigmoid(np.dot(l1, self.syn1))
        l3 = np.dot(l2, self.syn2)

        return l3

    def Q_gradientDescent(self, inp, n, target, learnRate):

        l0 = inp
        z1 = np.dot(l0, self.syn0)
        l1 = sigmoid(z1)
        z2 = np.dot(l1, self.syn1)
        l2 = sigmoid(z2)
        z3 = np.dot(l2, self.syn2)
        l3 = z3

        nodeValues2 = np.zeros_like(l3)
        nodeValues2[n] = costFunDerivative(l3[n], target)
        dcdw2 = np.outer(l2, nodeValues2)

        nodeValues1 = nodeValues2.dot(self.syn2.T) * sigmoidDerivative(z2)
        dcdw1 = np.outer(l1, nodeValues1)

        nodeValues0 = nodeValues1.dot(self.syn1.T) * sigmoidDerivative(z1)
        dcdw0 = np.outer(l0, nodeValues0)

        self.syn2 -= dcdw2 * learnRate
        self.syn1 -= dcdw1 * learnRate
        self.syn0 -= dcdw0 * learnRate

    def gradientDescent(self, inp, output, learnRate):
        l0 = inp
        z1 = np.dot(l0, self.syn0)
        l1 = sigmoid(z1)
        z2 = np.dot(l1, self.syn1)
        l2 = sigmoid(z2)
        z3 = np.dot(l2, self.syn2)
        l3 = sigmoid(z3)

        nodeValues2 = costFunDerivative(l3, output) * sigmoidDerivative(z3)
        dcdw2 = np.outer(l2, nodeValues2)

        nodeValues1 = nodeValues2.dot(self.syn2.T) * sigmoidDerivative(z2)
        dcdw1 = np.outer(l1, nodeValues1)

        nodeValues0 = nodeValues1.dot(self.syn1.T) * sigmoidDerivative(z1)
        dcdw0 = np.outer(l0, nodeValues0)

        self.syn2 -= dcdw2 * learnRate
        self.syn1 -= dcdw1 * learnRate
        self.syn0 -= dcdw0 * learnRate

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
                a.extend([0] * 21)
                continue
            n = i.howManySmall()
            a.extend([1] * n)
            a.extend([0] * (20 - n))
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

        a.append(1)  # bias

        if len(a) != 1332:
            print("AAAAAAAAAAAA")
            while True:
                pass
        return a

    def doAction(self, game, weights, loud=False):
        sortedout = weights.argsort().tolist()[::-1]
        if self.random:
            random.shuffle(sortedout)
        ai = game.findAI(self)
        if self.greedy:  # TODO: implement greedy better
            for i in game.currentCity.connections:
                if game.findAI(self) in [x.owner for x in game.findCity(i).cubes]:
                    game.currentCity.connect(i)
                    if loud: print("connected " + game.currentCity.name + " to " + i + "GREEDILY!")
                    return "connected " + game.currentCity.name + " to " + i + "GREEDILY!"
            for i in game.cities:
                if i.name in ["Minneapolis", "Seattle"]:
                    continue
                if len(ai.cubes) > 0 and 0 < len(i.cubes) < i.size and (
                        self not in list(set([x.owner for x in i.cubes]))):
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
            random.shuffle(weights)

        for i in sortedout:
            if self.greedy:
                weights[i] = -99.0
            if i < 50:
                if game.currentCity.name == norpac.allConnections[i][0]:
                    if (norpac.allConnections[i][1],
                        norpac.allConnections[i][0]) in game.trains:  # if double connection taken
                        continue
                    game.currentCity.connect(norpac.allConnections[i][1])
                    if loud: print("connected " + norpac.allConnections[i][0] + " to " + norpac.allConnections[i][
                        1] + " with confidence " + str(weights[i]))
                    return "connected " + norpac.allConnections[i][0] + " to " + norpac.allConnections[i][
                        1] + " with confidence " + str(weights[i])
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
                    if loud: print("placed big cube on " + city.name + " with confidence " + str(weights[i]))
                    return "placed big cube on " + city.name + " with confidence " + str(weights[i])
                # if even i.e. small cube
                if ai.howManySmall() > 0:
                    city.cubes.append(norpac.Cube(ai, False))
                    ai.spendSmall()
                    if loud: print("placed small cube on " + city.name + " with confidence " + str(weights[i]))
                    return "placed small cube on " + city.name + " with confidence " + str(weights[i])
                else:
                    continue
            else:
                n = i - 96
                if game.currentCity.name == norpac.seattleConnections[n][0]:
                    game.currentCity.connect(norpac.seattleConnections[n][1])
                    if loud: print(
                        "connected " + norpac.seattleConnections[n][0] + " to Seattle!!! with confidence " + str(
                            weights[i]))
                    return "connected " + norpac.seattleConnections[n][0] + " to Seattle!!! with confidence " + str(
                        weights[i])
        raise Exception(f"should not get here. value: {i}")

    def firstLegal(self, game, weights):
        sortedout = weights.argsort().tolist()[::-1]
        ai = game.findAI(self)
        for i in sortedout:
            if i < 50:
                if game.currentCity.name == norpac.allConnections[i][0]:
                    if (norpac.allConnections[i][1],
                        norpac.allConnections[i][0]) in game.trains:  # if double connection taken
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
