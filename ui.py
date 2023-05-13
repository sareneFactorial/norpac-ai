import sys
import norpac
import pygame
import pickle
import tftest
import newnn as neuralnet
import random
from pytorchnetworks import NeuralNetwork
import numpy
from datetime import datetime
from pygame.locals import *
import copy

pygame.init()

fps = 60
fpsClock = pygame.time.Clock()




pygame.font.init()

# sorry if you don't have this font lol its on google fonts
# light_font = pygame.font.SysFont("Mulish ExtraLight", 20)
# my_font = pygame.font.SysFont("Mulish ExtraLight", 20, bold=True)
#
# small_font = pygame.font.SysFont("Mulish ExtraLight", 12)
# small_bold = pygame.font.SysFont("Mulish ExtraLight", 12, bold=True)
#width, height = 1800, 1000

# for linux
light_font = pygame.font.SysFont("Mulish", 20)
my_font = pygame.font.SysFont("Mulish", 20, bold=True)

small_font = pygame.font.SysFont("Mulish", 12)
small_bold = pygame.font.SysFont("Mulish", 12, bold=True)
width, height = 1595, 800

screen = pygame.display.set_mode((width, height))

imp = pygame.image.load("board.jpg").convert()

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
player_colors = {}

backupnum = 1

# 30 X 30 CITY AREAS

currentPlayer = None
currentAiWeights = None
currentAiWeightsRaw = None

cityCoords = {
    "Seattle": (98, 141),
    "Vancouver": (97, 17),
    "Portland": (52, 272),
    "Oroville": (223, 72),
    "Richland": (183, 255),
    "Spokane": (281, 167),
    "Bonner's Ferry": (366, 83),
    "Lewiston": (322, 263),
    "Shelby": (496, 97),
    "Butte": (486, 296),
    "Great Falls": (550, 195),
    "Chinook": (629, 102),
    "Glasgow": (763, 109),
    "Terry": (817, 206),
    "Billings": (663, 289),
    "Casper": (743, 439),
    "Rapid City": (883, 363),
    "Minot": (948, 89),
    "Bismarck": (976, 197),
    "Aberdeen": (1082, 274),
    "Sioux Falls": (1171, 368),
    "Grand Forks": (1123, 71),
    "Duluth": (1328, 123),
    "Fargo": (1141, 173),
    "Minneapolis": (1318, 254), }

humanMoves = []

game = None
player = None

# with open("2023-05-07-172642-DQN-newnn-generation160.pkl", 'rb') as f:  # open a text file
#     activeNetwork = pickle.load(f)

with open("2023-05-09-205035-DQN-pytorch-generation100.pkl", 'rb') as f:  # open a text file
    activeNetwork = pickle.load(f)

gameLog = []

timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')

newGame = True

# Game loop.
while True:
    screen.fill((0, 0, 0))

    m_x, m_y = pygame.mouse.get_pos()

    if newGame:
        game = norpac.NorpacGame()
        player = norpac.Player()

        # nns = random.sample(population, 2)
        nns = [copy.deepcopy(activeNetwork), NeuralNetwork(greedy=True), activeNetwork, copy.deepcopy(activeNetwork)]
        # game.players.append(player)
        for n in nns:
            game.players.append(norpac.Player(n))

        random.shuffle(game.players)

        random.shuffle(colors)
        for i,v in enumerate(game.players):
            player_colors[v] = colors[i]

        game.setupGame(3)
        newGame = False


    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN and game.currentPlayer == player:
            clicked_city = [s[0] for s in cityCoords.items() if pygame.Rect(s[1], (30, 30)).collidepoint(m_x, m_y)]
            if len(clicked_city) == 0:
                continue

            city = game.findCity(clicked_city[0])
            n = ((game.cities.index(city) - 1) * 2) + 50
            gameState = sampleNN.createInput(game)
            if event.button == 3:  # if odd i.e. big cube
                if not player.hasBig():
                    continue
                city.cubes.append(norpac.Cube(player, True))
                player.spendBig()
                n += 1

            elif event.button == 1:
                if player.howManySmall() > 0:
                    game.findCity(clicked_city[0]).cubes.append(norpac.Cube(player, False))
                    player.spendSmall()
                else:
                    continue
            game.currentPlayer = game.playerOrder[
                (game.playerOrder.index(game.currentPlayer) + 1) % len(game.playerOrder)]
            humanMoves.append((gameState, n))
            gameLog.append("Player: " + neuralnet.readOutput(n))

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN and game.currentCity.name == "Seattle":
                gameLog.clear()
                if game.roundNumber == 2:
                    newGame = True
                    break
                game.countPoints()
                game.clearGame()
                game.roundNumber += 1
                continue
            elif event.key == pygame.K_s:
                # save stuff
                filename = f"{timestamp}humanmoves-backup{backupnum}.pkl"
                with open(filename, 'wb') as f:  # open a text file
                    pickle.dump(humanMoves, f)  # serialize the list
                    print(f"Saved human moves to {filename}")
                pass
            elif event.key == pygame.K_RETURN and game.currentPlayer != player:
                nn = game.currentPlayer.nn
                out = nn.output(nn.createInput(game))
                log = str(game.currentPlayer)[-4:-1]+" "+nn.doAction(game, out)
                game.currentPlayer = game.playerOrder[
                    (game.playerOrder.index(game.currentPlayer) + 1) % len(game.playerOrder)]
                gameLog.append(log)

            elif pygame.key.name(event.key) in "1234":
                if int(pygame.key.name(event.key)) > len(game.currentCity.connections):
                    continue
                city = game.currentCity.connections[int(pygame.key.name(event.key))-1]
                if city not in list(sum(game.trains, ())):
                    gameState = sampleNN.createInput(game)
                    if city != "Seattle":
                        n = norpac.allConnections.index((game.currentCity.name, city))
                    else:
                        n = norpac.seattleConnections.index((game.currentCity.name, city)) + 96
                    game.currentCity.connect(city)
                    game.currentPlayer = game.playerOrder[
                        (game.playerOrder.index(game.currentPlayer) + 1) % len(game.playerOrder)]
                    humanMoves.append((gameState, n))
                    gameLog.append("Player: " + neuralnet.readOutput(n))


    # Update.

    # text_surface = my_font.render(f"{x}, {y}", False, (0, 0, 0))

    # Draw.

    screen.blit(imp, (0, 0))
    for i, p in enumerate(game.playerOrder):
        font = my_font if p == game.currentPlayer else light_font
        st = ""
        if p == player:
            st += "(You) "
        elif p.nn == activeNetwork:
            st += "Our Boy "
        elif p.nn.random:
            st += "Random "
        elif p.nn.greedy:
            st += "Greedy "
        else:
            st += "Bozo "
        st += str(p)[-4:-1] + ": "
        st += str(len(p.cubes))
        if p.hasBig(): st += "!"
        st += f"        Points: {p.points}| Bad Inv.: {p.badInvestments}"
        screen.blit(font.render(st, False, (255, 255, 255)), (15, 560+i*25))
        pygame.draw.rect(screen, player_colors[p], pygame.Rect(0, 560+(i*25)+7, 15, 15))

    if game.currentPlayer == player:
        for i, c in enumerate(game.currentCity.connections):
            screen.blit(light_font.render(f"Press {i+1} to connect to {c}", False, (255, 255, 255)), (400, 560 + i * 25))

    screen.blit(light_font.render(f"Round {game.roundNumber+1}", False, (255, 255, 255)), (0, 700))

    # if game.currentPlayer != player and game.currentCity.name != "Seattle":
    #     nn = game.currentPlayer.nn
    #     out = nn.output(nn.createInput(game))
    #     log = str(game.currentPlayer)[-4:-1] + " " + nn.doAction(game, out)
    #     game.currentPlayer = game.playerOrder[
    #         (game.playerOrder.index(game.currentPlayer) + 1) % len(game.playerOrder)]
    #     gameLog.append(log)



    if game.currentPlayer != currentPlayer:
        currentPlayer = game.currentPlayer
        if currentPlayer.nn:
            out = game.currentPlayer.nn.output(game.currentPlayer.nn.createInput(game))

            sortedout = out.argsort().tolist()[::-1]
            actionList = [(neuralnet.readOutput(n), out[n]) for i, n in enumerate(sortedout)]
            currentAiWeights = actionList
            currentAiWeightsRaw = out
        else:
            currentAiWeights = [("you're the player", 1)]
            currentAiWeightsRaw = numpy.array([1])

    firstLegal = 1
    if currentPlayer.nn:
        firstLegal = currentPlayer.nn.firstLegal(game, currentAiWeightsRaw)

    for i,(v, weight) in enumerate(currentAiWeights):
        sortedout = currentAiWeightsRaw.argsort().tolist()[::-1]
        font = small_font if v != neuralnet.readOutput(firstLegal) else small_bold
        # print(firstLegal)
        # print(neuralnet.readOutput(firstLegal))

        screen.blit(font.render(f"{sortedout[i]} {v}: {weight}", False, (255, 255, 255)), (1407, (i * 9)))

    for i, v in enumerate(gameLog[::-1][:15]):
        screen.blit(my_font.render(v, False, (255, 255, 255)), (800, (i * 20)+560))


    for name, (x, y) in cityCoords.items():
        color = (0, 0, 0)
        if name in list(sum(game.trains, ())):
            color = (255, 255, 255)
        pygame.draw.rect(screen, color, pygame.Rect(x, y, 30, 30))
        # screen.blit(my_font.render(name, False, (0, 255, 0)), (x, y))

    for i in game.cities:
        for j, v in enumerate(i.cubes):
            if j == 0:
                cubeCoords = (0, 0)
            elif j == 1:
                cubeCoords = (15, 0)
            elif j == 2:
                cubeCoords = (0, 15)
            elif j == 3:
                cubeCoords = (15, 15)

            size = (10, 10)

            if v.big:
                size = (15, 15)
            city = cityCoords[i.name]
            newCoords = (city[0] + cubeCoords[0], city[1] + cubeCoords[1])
            pygame.draw.rect(screen, player_colors[v.owner], pygame.Rect(newCoords, size))


    pygame.display.flip()
    fpsClock.tick(fps)
