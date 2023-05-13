import random

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
            raise Exception("player count out of range! 2 <= playercount <= 6")

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
        visited.append(city)
        for i in self.findCity(city).connections:
            if i not in visited:
                self.treeSearch(i, visited)
        return visited



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
