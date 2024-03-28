import sys
import norpac
import pygame
import pickle
import old.newnn as neuralnet
import random

from old.oldpytorch import pytorchnetworks
import numpy
from datetime import datetime
from pygame.locals import *
import copy
import os
import torch

pygame.init()
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

fps = 60
fpsClock = pygame.time.Clock()

# feel free to change this to any checkpoint you want
with open(os.path.join(os.path.dirname(__file__), "good-checkpoints/2023-05-13-062509-DQN-dueling-cuda-generation1000.pkl"), 'rb') as f:
    activeNetwork = torch.load(f, map_location=device)

# watch the AI or play?
PLAYER_IN_GAME = False

# the fourth player to replace the user
with open(os.path.join(os.path.dirname(__file__), "good-checkpoints/2023-05-10-111231-DQN-pytorch-generation20.pkl"), 'rb') as f:
    otherBozo = pickle.load(f)

pygame.font.init()

# sorry if you don't have this font lol its on google fonts.
# TODO: use a default font or something
# for windows
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
# because my laptop screen is tiny small
width, height = 1595, 800

screen = pygame.display.set_mode((width, height))

imp = pygame.image.load("board.jpg").convert()

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
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

# humanMoves = []

game = None
player = None

gameLog = []

timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')

newGame = True

distribAI = copy.deepcopy(activeNetwork)
top5RandomAI = copy.deepcopy(activeNetwork)

# Game loop.
while True:
    screen.fill((0, 0, 0))

    m_x, m_y = pygame.mouse.get_pos()

    if newGame:
        game = norpac.NorpacGame()
        player = norpac.Player()

        # nns = random.sample(population, 2)
        nns = [distribAI, top5RandomAI, activeNetwork]
        if PLAYER_IN_GAME:
            game.players.append(player)
        else:
            nns.append(otherBozo)
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
            n = ((game.cities.index(city) + 1) * 2) + 50
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
            # elif event.key == pygame.K_s:
            #     # save stuff
            #     filename = f"{timestamp}humanmoves-backup{backupnum}.pkl"
            #     with open(filename, 'wb') as f:  # open a text file
            #         pickle.dump(humanMoves, f)  # serialize the list
            #         print(f"Saved human moves to {filename}")
            #     pass
            elif event.key == pygame.K_RETURN and game.currentPlayer != player:
                nn = game.currentPlayer.nn
                out = nn.output(nn.createInput(game))

                if nn == top5RandomAI:
                    sortedout = out.argsort().tolist()[::-1]
                    legalMoves = pytorchnetworks.allLegal(top5RandomAI, game)
                    sortedLegals = [it for it in sortedout if it in legalMoves]
                    choice = random.choice(sortedLegals[0:min(5, len(sortedLegals))])
                    log = nn.doSingleAction(game, choice)
                elif nn == distribAI:
                    legalMoves = pytorchnetworks.allLegal(distribAI, game)
                    # first zero out any non-legal moves
                    weights = [(float(v)) if i in legalMoves else 0 for i, v in enumerate(out)]
                    # new baseline for zero is the lowest legal action not lowest overall action
                    zero = min(weights)
                    weights = [(it + abs(zero)) ** 3 if it != 0 else 0 for it in weights]
                    if sum(weights) == 0:
                        log = nn.doSingleAction(game, random.choice(legalMoves))
                    else:
                        choice = random.choices(list(range(0, 100)), weights=weights, k=1)[0]
                        log = nn.doSingleAction(game, choice)
                else:
                    log = nn.doAction(game, out)

                log = str(game.currentPlayer)[-4:-1]+" "+log
                game.currentPlayer = game.playerOrder[
                    (game.playerOrder.index(game.currentPlayer) + 1) % len(game.playerOrder)]
                gameLog.append(log)

            elif pygame.key.name(event.key) in "1234":
                if int(pygame.key.name(event.key)) > len(game.currentCity.connections):
                    continue
                city = game.currentCity.connections[int(pygame.key.name(event.key))-1]
                if city not in list(sum(game.trains, ())):
                    # gameState = pytorchnetworks.createInput(game)
                    if city != "Seattle":
                        n = norpac.allConnections.index((game.currentCity.name, city))
                    else:
                        n = norpac.seattleConnections.index((game.currentCity.name, city)) + 96
                    game.currentCity.connect(city)
                    game.currentPlayer = game.playerOrder[
                        (game.playerOrder.index(game.currentPlayer) + 1) % len(game.playerOrder)]
                    # humanMoves.append((gameState, n))
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
        elif p.nn == distribAI:
            st += "Distrib. "
        elif p.nn == top5RandomAI:
            st += "Top5Rnd. "
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
