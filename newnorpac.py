# norpac.py but immutable this time
import functools
from dataclasses import dataclass
from typing import Self
import copy
import random
from enum import Enum
from functools import cached_property

from util import hotOne


allConnections = [('Oroville', 'Vancouver'), ('Richland', 'Portland'), ('Spokane', 'Oroville'), ('Spokane', 'Richland'),
                  ("Bonner's Ferry", 'Oroville'), ("Bonner's Ferry", 'Spokane'), ('Lewiston', 'Richland'),
                  ('Lewiston', 'Spokane'), ('Shelby', "Bonner's Ferry"), ('Shelby', 'Lewiston'),
                  ('Shelby', 'Great Falls'), ('Butte', 'Great Falls'), ('Butte', 'Lewiston'), ('Great Falls', 'Shelby'),
                  ('Great Falls', 'Butte'), ('Great Falls', 'Lewiston'), ('Chinook', 'Shelby'),
                  ('Chinook', 'Great Falls'), ('Glasgow', 'Chinook'), ('Glasgow', 'Great Falls'), ('Glasgow', 'Terry'),
                  ('Terry', 'Glasgow'), ('Terry', 'Great Falls'), ('Terry', 'Billings'), ('Billings', 'Great Falls'),
                  ('Billings', 'Butte'), ('Casper', 'Billings'), ('Casper', 'Butte'), ('Rapid City', 'Terry'),
                  ('Rapid City', 'Billings'), ('Rapid City', 'Casper'), ('Minot', 'Terry'), ('Minot', 'Glasgow'),
                  ('Bismarck', 'Terry'), ('Bismarck', 'Minot'), ('Aberdeen', 'Bismarck'), ('Aberdeen', 'Rapid City'),
                  ('Sioux Falls', 'Aberdeen'), ('Sioux Falls', 'Rapid City'), ('Grand Forks', 'Minot'),
                  ('Grand Forks', 'Fargo'), ('Duluth', 'Grand Forks'), ('Duluth', 'Fargo'), ('Fargo', 'Grand Forks'),
                  ('Fargo', 'Minot'), ('Fargo', 'Bismarck'), ('Minneapolis', 'Duluth'), ('Minneapolis', 'Fargo'),
                  ('Minneapolis', 'Aberdeen'), ('Minneapolis', 'Sioux Falls')]
seattleConnections = [("Vancouver", "Seattle"), ("Oroville", "Seattle"), ("Richland", "Seattle"),
                      ("Portland", "Seattle")]

cityIndices = ["Seattle",
               "Vancouver",
               "Portland",
               "Oroville",
               "Richland",
               "Spokane",
               "Bonner's Ferry",
               "Lewiston",
               "Shelby",
               "Butte",
               "Great Falls",
               "Chinook",
               "Glasgow",
               "Terry",
               "Billings",
               "Casper",
               "Rapid City",
               "Minot",
               "Bismarck",
               "Aberdeen",
               "Sioux Falls",
               "Grand Forks",
               "Duluth",
               "Fargo",
               "Minneapolis"]

class CityEnum(Enum):
    Seattle = 0
    Vancouver = 1
    Portland = 2
    Oroville = 3
    Richland = 4
    Spokane = 5
    Bonner = 6
    Lewiston = 7
    Shelby = 8
    Butte = 9
    Great = 10
    Chinook = 11
    Glasgow = 12
    Terry = 13
    Billings = 14
    Casper = 15
    Rapid = 16
    Minot = 17
    Bismarck = 18
    Aberdeen = 19
    Sioux = 20
    Grand = 21
    Duluth = 22
    Fargo = 23
    Minneapolis = 24


@dataclass(frozen=True)
class City:
    name: str
    connections: list[str]
    enum: CityEnum

    def __eq__(self, other):
        return self.enum == other.enum


CITIES = [  # TODO: optimize this to remove string comparison
    City("Seattle", [], CityEnum.Seattle),
    City("Vancouver", ["Seattle"], CityEnum.Vancouver),
    City("Portland", ["Seattle"], CityEnum.Portland),
    City("Oroville", ["Seattle", "Vancouver"], CityEnum.Oroville),
    City("Richland", ["Seattle", "Portland"], CityEnum.Richland),
    City("Spokane", ["Oroville", "Richland"], CityEnum.Spokane),
    City("Bonner's Ferry", ["Oroville", "Spokane"], CityEnum.Bonner),
    City("Lewiston", ["Richland", "Spokane"], CityEnum.Lewiston),
    City("Shelby", ["Bonner's Ferry", "Lewiston", "Great Falls"], CityEnum.Shelby),
    City("Butte", ["Lewiston", "Great Falls"], CityEnum.Butte),
    City("Great Falls", ["Shelby", "Butte", "Lewiston"], CityEnum.Great),
    City("Chinook", ["Shelby", "Great Falls"], CityEnum.Chinook),
    City("Glasgow", ["Chinook", "Great Falls", "Terry"], CityEnum.Glasgow),
    City("Terry", ["Glasgow", "Great Falls", "Billings"], CityEnum.Terry),
    City("Billings", ["Great Falls", "Butte"], CityEnum.Billings),
    City("Casper", ["Billings", "Butte"], CityEnum.Casper),
    City("Rapid City", ["Terry", "Billings", "Casper"], CityEnum.Rapid),
    City("Minot", ["Terry", "Glasgow"], CityEnum.Minot),
    City("Bismarck", ["Terry", "Minot"], CityEnum.Bismarck),
    City("Aberdeen", ["Bismarck", "Rapid City"], CityEnum.Aberdeen),
    City("Sioux Falls", ["Aberdeen", "Rapid City"], CityEnum.Sioux),
    City("Grand Forks", ["Minot", "Fargo"], CityEnum.Grand),
    City("Duluth", ["Grand Forks", "Fargo"], CityEnum.Duluth),
    City("Fargo", ["Grand Forks", "Minot", "Bismarck"], CityEnum.Fargo),
    City("Minneapolis", ["Duluth", "Fargo", "Aberdeen", "Sioux Falls"], CityEnum.Minneapolis),
]


class Player:
    # TODO: implement Actor
    def __init__(self, actor=None):
        self.uuid = random.random()
        self.actor = actor

    # def __hash__(self):
    #     return hash(self.uuid)


@dataclass
class Cube:
    owner: Player
    location: City
    big: bool


@dataclass(frozen=True)
class NorpacGame:
    players: list[Player]
    playerOrder: list[Player]
    points: dict[Player, int]
    badInvestments: dict[Player, int]
    playerCubes: dict[Player, tuple[int, int]]  # small, big

    placedCubes: list[Cube]
    trains: list[tuple[str, str]]

    roundNumber: int
    currentPlayer: Player

    cityCubesCache = [None] * 25
    inputCache = {}

    def newState(self, **kwargs) -> Self:
        players = kwargs.get("players", self.players)
        playerOrder = kwargs.get("playerOrder", self.playerOrder)
        points = kwargs.get("points", self.points)
        badInvestments = kwargs.get("badInvestments", self.badInvestments)
        playerCubes = kwargs.get("playerCubes", self.playerCubes)
        placedCubes = kwargs.get("placedCubes", self.placedCubes)
        trains = kwargs.get("trains", self.trains)
        roundNumber = kwargs.get("roundNumber", self.roundNumber)
        currentPlayer = kwargs.get("currentPlayer", self.currentPlayer)
        return NorpacGame(players, playerOrder, points, badInvestments, playerCubes, placedCubes, trains, roundNumber,
                          currentPlayer)

    def clearState(self, **kwargs) -> Self:
        players = kwargs.get("players", self.players)
        cubesDict = {}
        for i in players:
            cubesDict[i] = (3, 1)
        playerOrder = kwargs.get("playerOrder", self.playerOrder)
        points = kwargs.get("points", {})
        badInvestments = kwargs.get("badInvestments", {})
        playerCubes = kwargs.get("playerCubes", cubesDict)
        placedCubes = kwargs.get("placedCubes", [])
        trains = kwargs.get("trains", [])
        roundNumber = kwargs.get("roundNumber", 0)
        currentPlayer = kwargs.get("currentPlayer", self.currentPlayer)
        return NorpacGame(players, playerOrder, points, badInvestments, playerCubes, placedCubes, trains, roundNumber,
                          currentPlayer)

    @cached_property
    def maxCubes(self):
        if len(self.players) == 3:
            return 2
        if len(self.players) == 4 or len(self.players) == 5:
            return 3
        if len(self.players) == 6:
            return 4
        raise Exception("Invalid player count!")

    @property
    def terminalState(self):
        return self.roundNumber > 2  # TODO: custom number of rounds

    @property
    def currentCity(self):
        if len(self.trains) == 0:
            return "Minneapolis"
        return self.trains[-1][1]

    @property
    def reachedCities(self):
        return ["Minneapolis"] + [sub[1] for sub in self.trains]

    @property
    def nextPlayer(self):
        return self.playerOrder[(self.playerOrder.index(self.currentPlayer) + 1) % len(self.playerOrder)]

    def getCityCubes(self, city: City):  # TODO: look into functools.lru_cache
        x = self.cityCubesCache[city.enum.value]
        if x is not None:
            return x
        x = [x for x in self.placedCubes if x.location == city]
        self.cityCubesCache[city.enum.value] = x
        return x   # TODO: optimize

    def bigCubes(self, player: Player):
        return self.playerCubes[player][1]

    def smallCubes(self, player: Player):
        return self.playerCubes[player][0]

    # @functools.lru_cache(maxsize=100, typed=False)
    def allLegalMoves(self, player: Player):  # TODO: optimize this, most time is spent in this function somehow
        # check cache

        legalMoves = []
        for i in range(0, 100):
            if 50 <= i <= 53:
                continue
            if i < 50:
                if self.currentCity == allConnections[i][0]:
                    if (allConnections[i][1], allConnections[i][0]) in self.trains:  # if double connection taken
                        continue
                    legalMoves.append(i)
                    continue
            elif i < 96:
                j = (i - 50) // 2
                city = CITIES[j - 1]
                if len(self.getCityCubes(city)) >= self.maxCubes:  # if city full
                    continue
                if self.playerCubes[player][0] + self.playerCubes[player][1] == 0:  # if ai have no cube :(
                    continue
                if city.name in self.reachedCities:  # city connected to!! already
                    continue

                if i % 2 == 1:  # if odd i.e. big cube
                    if self.bigCubes(player) <= 0:
                        continue
                    legalMoves.append(i)
                    continue
                # if even i.e. small cube
                if self.smallCubes(player) > 0:
                    legalMoves.append(i)
                    continue
                else:
                    continue
            else:
                n = i - 96
                if self.currentCity == seattleConnections[n][0]:
                    legalMoves.append(i)
                    continue
        return legalMoves

    def doAction(self, player: Player, action: int, extraText: str = "") -> tuple[Self, str]:
        """ Does an action, returns new state without modifying original object.
        Does not check for legality individually. """
        # if action not in self.allLegalMoves(player):
        #     raise Exception

        if action < 50:  # connection
            if (self.currentCity == allConnections[action][0] and  # if connection is in current city
                    not allConnections[action] in self.trains and  # if connection is not taken
                    not allConnections[action][::-1] in self.trains):  # if the reverse connection is not taken
                connectedCity = [x for x in CITIES if x.name == allConnections[action][0]][0]
                cityCubes = [x for x in self.placedCubes if x.location == connectedCity]
                cubesToAdd = {}
                bigCubesToAdd = {}
                for x in cityCubes:
                    cubesToAdd[x.owner] = cubesToAdd.get(x.owner, 0) + 2
                    if x.big:
                        bigCubesToAdd[x.owner] = bigCubesToAdd.get(x.owner, 0) + 1
                playerCubes = copy.copy(self.playerCubes)
                for k,v in cubesToAdd.items():
                    newList = list(playerCubes[player])
                    newList[0] += v
                    playerCubes[player] = tuple(newList)
                for k,v in bigCubesToAdd.items():
                    newList = list(playerCubes[player])
                    newList[1] += v  # TODO: this should never be above 1. remove the tuple?
                    playerCubes[player] = tuple(newList)
                newState = self.newState(
                    trains=(self.trains + [allConnections[action]]),
                    playerCubes=playerCubes,
                    currentPlayer=self.nextPlayer)
                return newState, f"connected {allConnections[action][0]} to {allConnections[action][1]}" + extraText
            else:
                raise Exception("invalid move!")
        elif action < 96:
            j = (action - 50) // 2
            city = CITIES[j - 1]
            if action % 2 == 1:  # if odd i.e. big cube
                newDict = copy.copy(self.playerCubes)
                newList = list(newDict[player])
                newList[1] -= 1
                newDict[player] = tuple(newList)
                newState = self.newState(
                    placedCubes=self.placedCubes + [Cube(player, city, True)],
                    playerCubes=newDict,
                    currentPlayer=self.nextPlayer
                )
                return newState, f"placed big cube on {city.name}" + extraText
            # if even i.e. small cube
            newDict = copy.copy(self.playerCubes)
            newList = list(newDict[player])
            newList[0] -= 1
            newDict[player] = tuple(newList)
            newState = self.newState(
                placedCubes=self.placedCubes + [Cube(player, city, False)],
                playerCubes=newDict,
                currentPlayer=self.nextPlayer
            )
            return newState, f"placed small cube on {city.name}" + extraText
        else:
            n = action - 96
            if self.currentCity == seattleConnections[n][0]:
                points = copy.copy(self.points)
                badInvestments = copy.copy(self.badInvestments)
                for p in self.players:
                    points[p] = points.get(p, 0) + (self.smallCubes(player) + self.bigCubes(player))
                    badInvestments[p] = badInvestments.get(p, 0) + len([x for x in self.placedCubes if x.owner == player])
                playerOrder = copy.copy(self.playerOrder)
                random.shuffle(playerOrder)
                newState = self.clearState(points=points, badInvestments=badInvestments, playerOrder=playerOrder,
                                           roundNumber=self.roundNumber + 1, currentPlayer=playerOrder[0])
                return newState, f"connected {seattleConnections[n][0]} to Seattle!!!" + extraText
        raise Exception(f"invalid move number {action}")

    def createInput(self, player):
        """ Creates neural network input for a given player in a given state. """
        # TODO: should be the same for all players except for the "which player" vector - optimize based on that

        # check cache
        x = self.inputCache.get(player, None)
        if x is not None:
            return x

        a = []
        cities = CITIES[1:len(CITIES) - 1]
        # cubes on each cities and which players own them and which type they are
        for i in cities:
            for j in self.getCityCubes(i) + [None] * (4 - len(self.getCityCubes(i))):
                if j is None:
                    a.extend([0, 0] * 6)
                    continue
                for k in self.players + [None] * (6 - len(self.players)):
                    if k is None or j.owner != k:
                        a.extend([0, 0])
                        continue
                    a.extend(hotOne(1 if j.big else 0, 2))
        # which connections are active
        for i in allConnections:
            a.append(1 if i in self.trains else 0)
        # which player is in which player order spot vector (weird i know but i already coded it)
        for i in self.players + [None] * (6 - len(self.players)):
            if i is None:
                a.extend([0] * 6)
                continue
            a.extend(hotOne(self.playerOrder.index(i), 6))
        # how many cubes of each type
        for i in self.players + [None] * (6 - len(self.players)):
            if i is None:
                a.extend([0, 0])
                continue
            n = self.smallCubes(i)
            a.append(n / 20)
            a.append(self.bigCubes(i))  # TODO: is this valid? should always be 1
        # Which Player Are You vector
        for i in self.players + [None] * (6 - len(self.players)):
            if i is None or i != player:
                a.append(0)
                continue
            a.append(1)
        # points vector
        for i in self.players + [None] * (6 - len(self.players)):
            if i is None:
                a.append(0)
                continue
            a.append(self.points.get(i, 0) / 20)
        # round number
        a.extend(hotOne(self.roundNumber, 3))

        if len(a) != 1217:
            print("AAAAAAAAAAAA input length is messed up")
            print(len(a))
            raise Exception("input length is messed up idk why")
        self.inputCache[player] = a
        return a


def newGame(players: list[Player]) -> NorpacGame:
    newPlayers = copy.copy(players)
    random.shuffle(newPlayers)
    cubesDict = {}
    for i in players:
        cubesDict[i] = (3, 1)
    return NorpacGame(players, newPlayers, {}, {}, cubesDict, [], [], 0, newPlayers[0])


def readOutput(n):
    if n < 50:
        conn = allConnections[n]
        return f"Conn {conn[0]} to {conn[1]}"
    elif n < 96:
        j = (n - 50) // 2
        city = cityIndices[j - 1]
        st = ""
        if n % 2 == 1:  # if odd i.e. big cube
            st += "Big "
        else:
            st += "Small "
        st += f"cube on {city}"
        return st
    else:
        n = n - 96
        return f"Connect {seattleConnections[n][0]} to Seattle!!!"


# ACTION NUMBERS

# 0: Conn Oroville to Vancouver
# 1: Conn Richland to Portland
# 2: Conn Spokane to Oroville
# 3: Conn Spokane to Richland
# 4: Conn Bonner's Ferry to Oroville
# 5: Conn Bonner's Ferry to Spokane
# 6: Conn Lewiston to Richland
# 7: Conn Lewiston to Spokane
# 8: Conn Shelby to Bonner's Ferry
# 9: Conn Shelby to Lewiston
# 10: Conn Shelby to Great Falls
# 11: Conn Butte to Great Falls
# 12: Conn Butte to Lewiston
# 13: Conn Great Falls to Shelby
# 14: Conn Great Falls to Butte
# 15: Conn Great Falls to Lewiston
# 16: Conn Chinook to Shelby
# 17: Conn Chinook to Great Falls
# 18: Conn Glasgow to Chinook
# 19: Conn Glasgow to Great Falls
# 20: Conn Glasgow to Terry
# 21: Conn Terry to Glasgow
# 22: Conn Terry to Great Falls
# 23: Conn Terry to Billings
# 24: Conn Billings to Great Falls
# 25: Conn Billings to Butte
# 26: Conn Casper to Billings
# 27: Conn Casper to Butte
# 28: Conn Rapid City to Terry
# 29: Conn Rapid City to Billings
# 30: Conn Rapid City to Casper
# 31: Conn Minot to Terry
# 32: Conn Minot to Glasgow
# 33: Conn Bismarck to Terry
# 34: Conn Bismarck to Minot
# 35: Conn Aberdeen to Bismarck
# 36: Conn Aberdeen to Rapid City
# 37: Conn Sioux Falls to Aberdeen
# 38: Conn Sioux Falls to Rapid City
# 39: Conn Grand Forks to Minot
# 40: Conn Grand Forks to Fargo
# 41: Conn Duluth to Grand Forks
# 42: Conn Duluth to Fargo
# 43: Conn Fargo to Grand Forks
# 44: Conn Fargo to Minot
# 45: Conn Fargo to Bismarck
# 46: Conn Minneapolis to Duluth
# 47: Conn Minneapolis to Fargo
# 48: Conn Minneapolis to Aberdeen
# 49: Conn Minneapolis to Sioux Falls
# 50: Small cube on Minneapolis (INVALID) whoops
# 51: Big cube on Minneapolis (INVALID) whoops
# 52: Small cube on Seattle (INVALID) whoops
# 53: Big cube on Seattle (INVALID) whoops
# 54: Small cube on Vancouver
# 55: Big cube on Vancouver
# 56: Small cube on Portland
# 57: Big cube on Portland
# 58: Small cube on Oroville
# 59: Big cube on Oroville
# 60: Small cube on Richland
# 61: Big cube on Richland
# 62: Small cube on Spokane
# 63: Big cube on Spokane
# 64: Small cube on Bonner's Ferry
# 65: Big cube on Bonner's Ferry
# 66: Small cube on Lewiston
# 67: Big cube on Lewiston
# 68: Small cube on Shelby
# 69: Big cube on Shelby
# 70: Small cube on Butte
# 71: Big cube on Butte
# 72: Small cube on Great Falls
# 73: Big cube on Great Falls
# 74: Small cube on Chinook
# 75: Big cube on Chinook
# 76: Small cube on Glasgow
# 77: Big cube on Glasgow
# 78: Small cube on Terry
# 79: Big cube on Terry
# 80: Small cube on Billings
# 81: Big cube on Billings
# 82: Small cube on Casper
# 83: Big cube on Casper
# 84: Small cube on Rapid City
# 85: Big cube on Rapid City
# 86: Small cube on Minot
# 87: Big cube on Minot
# 88: Small cube on Bismarck
# 89: Big cube on Bismarck
# 90: Small cube on Aberdeen
# 91: Big cube on Aberdeen
# 92: Small cube on Sioux Falls
# 93: Big cube on Sioux Falls
# 94: Small cube on Grand Forks
# 95: Big cube on Grand Forks
# 96: Connect Vancouver to Seattle!!!
# 97: Connect Oroville to Seattle!!!
# 98: Connect Richland to Seattle!!!
# 99: Connect Portland to Seattle!!!
