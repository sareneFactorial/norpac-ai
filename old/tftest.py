# conec = ff.mlgraph((1331,100,96))

import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import numpy as np
import random
import norpac
from numba import njit

@njit(cache=True, fastmath=True)
def hotOne(h, n):
	return [1 if i == h else 0 for i in range(0, n)]

@njit(cache=True, fastmath=True)
def nonlin(x):
	return 1 / (1 + np.exp(-x))

def readOutput(n):
	if n < 50:
		conn = norpac.allConnections[n]
		return f"Conn {conn[0]} to {conn[1]}"
	elif n < 96:
		j = (n - 50) // 2
		city = norpac.cityIndices[j-1]
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





class NeuralNet:

	def __init__(self, syn0=None, syn1=None, greedy=False, rand=False):
		self.syn0 = syn0 if syn0 is not None else 2 * np.random.random((1332, 250)) - 1
		self.syn1 = syn1 if syn1 is not None else 2 * np.random.random((250, 100)) - 1
		self.greedy = greedy
		self.random = rand

	def output(self, l0):
		l1 = nonlin(np.dot(l0, self.syn0))
		l2 = nonlin(np.dot(l1, self.syn1))

		return l2

	def gradientDescent(self, inp, output, learnRate):
		l0 = X
		l1 = nonlin(np.dot(l0, syn0))
		l2 = nonlin(np.dot(l1, syn1))

		# how much did we miss the target value?
		l2_error = output - l2

		# in what direction is the target value?
		# were we really sure? if so, don't change too much.
		l2_delta = l2_error * nonlin(l2, deriv=True)

		# how much did each l1 value contribute to the l2 error (according to the weights)?
		l1_error = l2_delta.dot(syn1.T)

		# in what direction is the target l1?
		# were we really sure? if so, don't change too much.
		l1_delta = l1_error * nonlin(l1, deriv=True)

		syn1 += l1.T.dot(l2_delta)
		syn0 += l0.T.dot(l1_delta)

	def createInput(self, game: norpac.NorpacGame):
		a = []
		cities = game.cities[1:len(game.cities) - 1]
		for i in cities:
			for j in i.cubes + [None] * (4 - len(i.cubes)):
				if j is None:
					a += [0, 0] * 6
					continue
				for k in game.players + [None] * (6 - len(game.players)):
					if k is None or j.owner != k:
						a += [0, 0]
						continue
					a += hotOne(1 if j.big else 0, 2)
		for i in norpac.allConnections:
			if i in game.trains:
				a += [1]
			else:
				a += [0]
		for i in game.players + [None] * (6 - len(game.players)):
			if i is None:
				a += [0] * 6
				continue
			a += hotOne(game.playerOrder.index(i), 6)
		for i in game.players + [None] * (6 - len(game.players)):
			if i is None:
				a += [0] * 21
				continue
			n = i.howManySmall()
			a += [1] * n
			a += [0] * (20 - n)
			for j in i.cubes:
				if j.big:
					a += [1]
					break
			else:
				a += [0]
		for i in game.players + [None] * (6 - len(game.players)):
			if i is None or i.nn != self:
				a += [0]
				continue
			a += [1]
		for i in game.players + [None] * (6 - len(game.players)):
			if i is None:
				a += [0]
				continue
			a += [i.points / 20]
		a += hotOne(game.roundNumber, 3)

		a += [1]  # bias

		if len(a) != 1332:
			print("AAAAAAAAAAAA")
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
				if len(ai.cubes) > 0 and 0 < len(i.cubes) < i.size and (self not in list(set([x.owner for x in i.cubes]))):
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
					if (norpac.allConnections[i][1], norpac.allConnections[i][0]) in game.trains:  # if double connection taken
						continue
					game.currentCity.connect(norpac.allConnections[i][1])
					if loud: print("connected " + norpac.allConnections[i][0] + " to " + norpac.allConnections[i][1] + " with confidence " + str(weights[i]))
					return "connected " + norpac.allConnections[i][0] + " to " + norpac.allConnections[i][1] + " with confidence " + str(weights[i])
			elif i < 96:
				j = (i - 50) // 2
				city = game.cities[j-1]
				if len(city.cubes) >= city.size: # if city full
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
					if loud: print("connected " + norpac.seattleConnections[n][0] + " to Seattle!!! with confidence " + str(weights[i]))
					return "connected " + norpac.seattleConnections[n][0] + " to Seattle!!! with confidence " + str(weights[i])

		print("AAAAH I THSOULD NOT GET HERE!!!")
		print("AAAAH I THSOULD NOT GET HERE!!!")
		print("AAAAH I THSOULD NOT GET HERE!!!")
		print("PANIC PANIC PANIC PANIC PANIC PANIC PANIC ")
		print(i)
		while True:
			pass

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
				city = game.cities[j-1]
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



def breed(n1: NeuralNet, n2: NeuralNet, mut=0.05):

	syn0 = n1.syn0 if random.random() > 0.5 else n2.syn0
	syn1 = n1.syn1 if random.random() > 0.5 else n2.syn1

	# syn0 = (n1.syn0 + n2.syn0) / 2
	# syn1 = (n1.syn1 + n2.syn1) / 2
	#

	v0 = ((2 * random.random() - 1) * syn0) + syn0 if (random.random() < mut) else syn0
	v1 = ((2 * random.random() - 1) * syn1) + syn1 if (random.random() < mut) else syn1

	return NeuralNet(v0, v1)


def vary(n: NeuralNet):
	v0 = (n.syn0 * (random.random() - 0.5)) + n.syn0
	v1 = (n.syn1 * (random.random() - 0.5)) + n.syn1

	return NeuralNet(v0, v1)
