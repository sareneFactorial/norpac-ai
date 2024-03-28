import random

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
               "Minneapolis", ]


class NorpacGame:
    def __init__(self):
        self.trains = []  # connections are tuples
        self.cities = []
        self.currentCity = None  # late init
        self.reachedCities = []
        self.players = []
        self.playerOrder = []
        self.roundNumber = 0
        self.currentPlayer = None
        self.lastScore = {}

    def findCity(self, name: str):
        for i in self.cities:
            if i.name == name:
                return i
        return None

    def findAI(self, nn):
        for i in self.players:
            if i.nn == nn:
                return i
        return None

    def addCube(self, city: str, player, big: bool):
        realCity = self.findCity(city)
        realCity.cubes.append(Cube(player, big))

    # TODO: this is awful fix this somehow
    def setupGame(self, playerCount: int):
        if playerCount == 3:
            n = 2
        elif playerCount == 4 or playerCount == 5:
            n = 3
        elif playerCount == 6:
            n = 4
        else:
            raise Exception("player count out of range! 3 <= playercount <= 6")

        self.cities.append(City(self, "Seattle", [], 0))
        self.cities.append(City(self, "Vancouver", ["Seattle"], n))
        self.cities.append(City(self, "Portland", ["Seattle"], n))
        self.cities.append(City(self, "Oroville", ["Seattle", "Vancouver"], n))
        self.cities.append(City(self, "Richland", ["Seattle", "Portland"], n))
        self.cities.append(City(self, "Spokane", ["Oroville", "Richland"], n))
        self.cities.append(City(self, "Bonner's Ferry", ["Oroville", "Spokane"], n))
        self.cities.append(City(self, "Lewiston", ["Richland", "Spokane"], n))
        self.cities.append(City(self, "Shelby", ["Bonner's Ferry", "Lewiston", "Great Falls"], n))
        self.cities.append(City(self, "Butte", ["Lewiston", "Great Falls"], n))
        self.cities.append(City(self, "Great Falls", ["Shelby", "Butte", "Lewiston"], n))
        self.cities.append(City(self, "Chinook", ["Shelby", "Great Falls"], n))
        self.cities.append(City(self, "Glasgow", ["Chinook", "Great Falls", "Terry"], n))
        self.cities.append(City(self, "Terry", ["Glasgow", "Great Falls", "Billings"], n))
        self.cities.append(City(self, "Billings", ["Great Falls", "Butte"], n))
        self.cities.append(City(self, "Casper", ["Billings", "Butte"], n))
        self.cities.append(City(self, "Rapid City", ["Terry", "Billings", "Casper"], n))
        self.cities.append(City(self, "Minot", ["Terry", "Glasgow"], n))
        self.cities.append(City(self, "Bismarck", ["Terry", "Minot"], n))
        self.cities.append(City(self, "Aberdeen", ["Bismarck", "Rapid City"], n))
        self.cities.append(City(self, "Sioux Falls", ["Aberdeen", "Rapid City"], n))
        self.cities.append(City(self, "Grand Forks", ["Minot", "Fargo"], n))
        self.cities.append(City(self, "Duluth", ["Grand Forks", "Fargo"], n))
        self.cities.append(City(self, "Fargo", ["Grand Forks", "Minot", "Bismarck"], n))
        self.cities.append(City(self, "Minneapolis", ["Duluth", "Fargo", "Aberdeen", "Sioux Falls"], 0))

        self.clearGame()

    def clearGame(self):
        for i in self.cities:
            i.cubes.clear()
        self.trains.clear()
        self.currentCity = self.findCity("Minneapolis")
        self.playerOrder = [x for x in self.players]
        random.shuffle(self.playerOrder)
        self.currentPlayer = self.playerOrder[0]
        for i in self.players:
            i.cubes.clear()
            i.getCube(3, False)
            i.getCube(1, True)

    def countPoints(self):
        bads = {}
        for i in self.players:
            bads[i] = 0
            i.points += len(i.cubes)
        for i in self.cities:
            for j in i.cubes:
                bads[j.owner] += 1
        for (k, v) in bads.items():
            k.badInvestments += v

    def getBadCubes(self):
        visitable = self.treeSearch(self.currentCity.name, [])
        bads = {}
        for i in self.players:
            bads[i] = 0
        for i in [it for it in self.cities if it.name not in visitable]:
            for cube in i.cubes:
                bads[cube.owner] += 1
        return bads

    def getUnvisitableCities(self):
        visitable = self.treeSearch(self.currentCity.name, [])
        return [it for it in self.cities if it.name not in visitable]

    def treeSearch(self, city, visited):
        """ Gets a list of all cities reachable from this city, assuming empty connections. """
        visited.append(city)
        for i in self.findCity(city).connections:
            if i not in visited:
                self.treeSearch(i, visited)
        return visited

    def allLegalMoves(self, player):
        legalMoves = []
        for i in range(0, 100):
            if i < 50:
                if self.currentCity.name == allConnections[i][0]:
                    if (allConnections[i][1], allConnections[i][0]) in self.trains:  # if double connection taken
                        continue
                    legalMoves.append(i)
                    continue
            elif i < 96:
                j = (i - 50) // 2
                city = self.cities[j - 1]
                if len(city.cubes) >= city.size:  # if city full
                    continue
                if len(player.cubes) <= 0:  # if ai have no cube :(
                    continue
                if city.name in list(sum(self.trains, ())):  # city connected to!! already
                    continue

                if i % 2 == 1:  # if odd i.e. big cube
                    if not player.hasBig():
                        continue
                    legalMoves.append(i)
                    continue
                # if even i.e. small cube
                if player.howManySmall() > 0:
                    legalMoves.append(i)
                    continue
                else:
                    continue
            else:
                n = i - 96
                if self.currentCity.name == seattleConnections[n][0]:
                    legalMoves.append(i)
                    continue

    # see action numbers at bottom of this file
    def doAction(self, player, actionNumber, extraText=""):
        """ Does an action. Returns the logs. Extratext gets added to the end of logs.
        Does not check for illegality."""
        # TODO: should this return state?
        i = actionNumber
        if i < 50:
            if self.currentCity.name == allConnections[i][0]:
                self.currentCity.connect(allConnections[i][1])
                return f"connected {allConnections[i][0]} to {allConnections[i][1]}" + extraText
            else:
                raise Exception("invalid move!")
        elif i < 96:
            j = (i - 50) // 2
            city = self.cities[j - 1]
            # if city full, player has no cubes, or city is already connected to
            if len(city.cubes) >= city.size or len(player.cubes) <= 0 or city.name in list(sum(self.trains, ())):
                raise Exception("invalid move")
            if i % 2 == 1:  # if odd i.e. big cube
                if not player.hasBig():
                    raise Exception("invalid move!!")
                city.cubes.append(Cube(player, True))
                player.spendBig()
                return f"placed big cube on {city.name}" + extraText
            # if even i.e. small cube
            if player.howManySmall() > 0:
                city.cubes.append(Cube(player, False))
                player.spendSmall()
                return f"placed small cube on {city.name}" + extraText
            else:
                raise Exception("invalid move")
        else:
            n = i - 96
            if self.currentCity.name == seattleConnections[n][0]:
                self.currentCity.connect(seattleConnections[n][1])
                return f"connected {seattleConnections[n][0]} to Seattle!!!" + extraText
        raise Exception(f"invalid move number {i}")

    def createInput(self, player):
        """ Creates neural network input for a given player in a given state. """
        a = []
        cities = self.cities[1:len(self.cities) - 1]
        # cubes on each cities and which players own them and which type they are
        for i in cities:
            for j in i.cubes + [None] * (4 - len(i.cubes)):
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
            n = i.howManySmall()
            a.append(n / 20)
            for j in i.cubes:
                if j.big:
                    a.append(1)
                    break
            else:
                a.append(0)
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
            a.append(i.points / 20)
        # round number
        a.extend(hotOne(self.roundNumber, 3))

        if len(a) != 1217:
            print("AAAAAAAAAAAA input length is messed up")
            print(len(a))
            raise Exception("input length is messed up idk why")
        return a

class City:
    def __init__(self, game: NorpacGame, name: str, connections, size: int):
        self.game = game
        self.connections = connections
        self.cubes = []
        self.size = size
        self.name = name

    def score(self):
        for cube in self.cubes:
            self.game.lastScore[cube.owner] = self.game.lastScore.get(cube.owner, 0) + 2
            cube.owner.getCube(2, False)
            if cube.big:
                cube.owner.getCube(1, True)
                self.game.lastScore[cube.owner] += 1
        self.cubes.clear()

    def connect(self, destination: str):
        city = self.game.findCity(destination)
        self.game.trains.append((self.name, city.name))
        self.game.currentCity = city
        city.score()


class Player:
    def __init__(self, nn=None):
        self.cubes = []
        self.getCube(3, False)
        self.getCube(1, True)
        self.nn = nn
        self.points = 0
        self.badInvestments = 0

    def getCube(self, num: int, big: bool):
        for i in range(0, num):
            self.cubes.append(Cube(self, big))

    def spendSmall(self):
        small_cubes = [cube for cube in self.cubes if not cube.big]
        if small_cubes:
            self.cubes.remove(small_cubes[0])

    def spendBig(self):
        big_cube = next((cube for cube in self.cubes if cube.big), None)
        if big_cube:
            self.cubes.remove(big_cube)

    def howManySmall(self):
        return sum(1 for i in self.cubes if not i.big)

    def hasBig(self):
        return any(i.big for i in self.cubes)


class Cube:
    def __init__(self, owner: Player, big: bool):
        self.owner = owner
        self.big = big

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
# 50: Small cube on Minneapolis
# 51: Big cube on Minneapolis
# 52: Small cube on Seattle
# 53: Big cube on Seattle
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
