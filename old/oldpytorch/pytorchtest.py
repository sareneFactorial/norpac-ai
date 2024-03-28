import norpac
import torch
import torch.optim as optim
import random
import collections
import time
from datetime import datetime
import numpy as np
import copy
import pickle
import pytorchnetworks
from pytorchnetworks import NeuralNetwork
import cProfile
import os

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device")

GAMES_PER_GEN = 200
NUM_GENS = 100
LEARN_RATE = 1e-4
TEST_GAMES = 0
LOOKFORWARD_STEPS = 4  # how many steps to look forward with multi step learning

PROGRESS_UPDATES = 5000
BUFFER_SIZE = 20000
CHECKPOINT = 20

EPSILON = 0.65  # likelihood of AI doing a random move. reduced by winrate
GAMMA = 0.95  # how much the next state is counted for the reward

timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
checkpointsDir = os.path.join(os.path.dirname(__file__), "checkpoints/")

fullExperienceBuffer = collections.deque(maxlen=BUFFER_SIZE)



# TODO: plot this on a graph while it's training along with data like average reward, average scores, etc.
# TODO: make this a constant parameter to adjust
# keeping average of last 5 winrates for statistics
aiWinRates = collections.deque(maxlen=5)
greedyWinRates = collections.deque(maxlen=5)
randomWinRates = collections.deque(maxlen=5)
laggingWinRates = collections.deque(maxlen=5)
formidableWinRates = collections.deque(maxlen=5)
distribWinRates = collections.deque(maxlen=5)
top5RandomWinRates = collections.deque(maxlen=5)
perfectWinRates = collections.deque(maxlen=5)

# actingNetwork = DuelingNetwork().to(device)

# feel free to use this instead with the relevant file to resume training:
with open(os.path.join(os.path.dirname(__file__), "good-checkpoints/2023-05-13-062509-DQN-dueling-cuda-generation1000.pkl"), 'rb') as f:
    actingNetwork = torch.load(f, map_location=device)

# older AIs for baseline comparison
laggingAis = collections.deque(maxlen=20)

randomAI = NeuralNetwork(rand=True).to(device)
greedyAI = NeuralNetwork(greedy=True).to(device)
top5RandomAI = copy.deepcopy(actingNetwork)  # picks randomly from its top 5 moves
distribAI = copy.deepcopy(actingNetwork)  # picks randomly using softmax q values as a distribution
perfectAI = copy.deepcopy(actingNetwork)  # copy of the ai that doesn't use epsilon-greedy

with open(os.path.join(os.path.dirname(__file__), "good-checkpoints/2023-05-10-111231-DQN-pytorch-generation20.pkl"), 'rb') as f:
    formidableOpponent = pickle.load(f)


optimizer = optim.AdamW(actingNetwork.parameters(), lr=LEARN_RATE, amsgrad=True)


def newTest():
    global distribAI  # why do only some of these need the global keyword? python vexes me
    global top5RandomAI
    global perfectAI
    for generation in range(0, NUM_GENS+1):
        print(f"\nGeneration {generation}")

        start = time.time()

        # double dqn baybee
        targetNetwork = copy.deepcopy(actingNetwork)

        # baseline opponent networks
        laggingAis.appendleft(copy.deepcopy(actingNetwork))
        laggingNetwork = laggingAis[-1]
        top5RandomAI = copy.deepcopy(actingNetwork)
        distribAI = copy.deepcopy(actingNetwork)
        perfectAI = copy.deepcopy(actingNetwork)

        # just to initialize this because its jank
        otherPlayerBads_new = 0

        # used for calculating a reward that is "immediate" but only applied on the next transition
        persistentReward = 0

        # TODO: make this a dict or something
        randomAiPlays = 0
        randomAiWins = 0
        greedyAiPlays = 0
        greedyAiWins = 0
        laggingAiPlays = 0
        laggingAiWins = 0
        formidableAiPlays = 0
        formidableAiWins = 0
        distribAiPlays = 0
        distribAiWins = 0
        top5RandomAiPlays = 0
        top5RandomAiWins = 0
        perfectAiPlays = 0
        perfectAiWins = 0
        aiWins = 0

        for gameNumber in range(0,GAMES_PER_GEN+1):  # TODO: move the +1 to the visual representations of the number
            playerCount = random.randrange(3, 7)  # player count from 3 to 6

            if gameNumber % PROGRESS_UPDATES == 0 and gameNumber != 0:
                print(f"Game #{gameNumber}....")
            game = norpac.NorpacGame()

            candidates = [laggingNetwork, randomAI, formidableOpponent, greedyAI, top5RandomAI, distribAI, perfectAI, ]

            nns = random.sample(candidates, playerCount-1)
            nns.append(actingNetwork)
            random.shuffle(nns)

            if any(it.random for it in nns):
                randomAiPlays += 1
            if any(it.greedy for it in nns):
                greedyAiPlays += 1
            if any(it == laggingNetwork for it in nns):
                laggingAiPlays += 1
            if any(it == formidableOpponent for it in nns):
                formidableAiPlays += 1
            if any(it == top5RandomAI for it in nns):
                top5RandomAiPlays += 1
            if any(it == distribAI for it in nns):
                distribAiPlays += 1
            if any(it == perfectAI for it in nns):
                perfectAiPlays += 1

            for i in nns:  # TODO: fix inputting players
                game.players.append(norpac.Player(i))
            game.setupGame(playerCount)

            experienceBuffer = []  # (state, action, reward, nextState, done)

            for r in range(0, 3):
                game.clearGame()
                game.roundNumber = r

                activePlayerBads_old = game.getBadCubes()[game.currentPlayer]
                otherPlayerBads_old = sum([v for (k, v) in game.getBadCubes().items() if k != game.currentPlayer])

                incompleteExperienceTuple = ()

                while game.currentCity.name != "Seattle":
                    nn = game.currentPlayer.nn
                    state = nn.createInput(game)
                    out = nn.output(state)

                    if nn == actingNetwork:
                        # epsilon is reduced based on performance of AI
                        # TODO: remove the random.shuffle here for speed
                        if random.random() < (EPSILON * (1 - aiWinRates[0]) if len(aiWinRates) > 0 else EPSILON):
                            random.shuffle(out)

                        otherPlayerPoints_new = sum([v for k, v in game.lastScore.items() if k != game.currentPlayer])
                        activePlayerPoints_new = game.lastScore.get(game.currentPlayer, 0) if game.lastScore.get(
                            game.currentPlayer) is not None else 0
                        game.lastScore.clear()
                        activePlayerBads_new = game.getBadCubes()[game.currentPlayer] - activePlayerBads_old

                        reward = ((activePlayerPoints_new * 1.2) - (otherPlayerPoints_new * 1)) + (
                                    (otherPlayerBads_new * 0.5) - (activePlayerBads_new * 1.3)) * 1 + persistentReward

                        persistentReward = 0

                        if len(incompleteExperienceTuple) != 0:
                            incompleteExperienceTuple += (reward, state, False)
                            experienceBuffer.append(incompleteExperienceTuple)

                        activePlayerBads_old = game.getBadCubes()[game.currentPlayer]
                        otherPlayerBads_old = sum([v for (k, v) in game.getBadCubes().items() if k != game.currentPlayer])

                    chosenAction = nn.firstLegal(game, out)

                    if nn == top5RandomAI:
                        sortedout = out.argsort().tolist()[::-1]
                        legalMoves = pytorchnetworks.allLegal(top5RandomAI, game)
                        sortedLegals = [it for it in sortedout if it in legalMoves]
                        choice = random.choice(sortedLegals[0:min(5, len(sortedLegals))])
                        log = nn.doSingleAction(game, choice)
                    elif nn == distribAI:
                        legalMoves = pytorchnetworks.allLegal(distribAI, game)
                        # first zero out any non-legal moves
                        weights = [(float(v)) if i in legalMoves else 0 for i,v in enumerate(out)]
                        # new baseline for zero is the lowest legal action not lowest overall action
                        zero = min(weights)
                        weights = [(it + abs(zero))**2 if it != 0 else 0 for it in weights]
                        if sum(weights) == 0:
                            log = nn.doSingleAction(game, random.choice(legalMoves))
                        else:
                            choice = random.choices(list(range(0, 100)), weights=weights, k=1)[0]
                            log = nn.doSingleAction(game, choice)
                    else:
                        log = nn.doAction(game, out)

                    if nn == actingNetwork:
                        otherPlayerBads_new = sum([v for k, v in game.getBadCubes().items() if k != game.currentPlayer]) - otherPlayerBads_old
                        # discourage placing useless cubes or making own investments bad
                        if game.getBadCubes()[game.currentPlayer] > activePlayerBads_old:
                            persistentReward -= (game.getBadCubes()[game.currentPlayer] - activePlayerBads_old) * 3
                        # discourage connections without cubes out
                        cubesOut = sum([sum([(1 if jt.owner == game.currentPlayer else 0) for jt in it.cubes]) for it in game.cities])
                        if chosenAction < 50 and cubesOut == 0:
                            persistentReward -= 3

                    # if (aiWinRates[0] if len(aiWinRates) > 0 else 1) > 0.6 and gameNumber == 0 and game.roundNumber == 0:
                    # if gameNumber == 0 and game.roundNumber == 0:
                    if gameNumber == 0 and game.roundNumber == 0 and generation % 20 == 0 and generation != 0:
                        s = ""
                        if game.currentPlayer.nn == actingNetwork:
                            s += "Our Boy"
                        elif game.currentPlayer.nn.greedy:
                            s += "Greedy Player"
                        elif game.currentPlayer.nn.random:
                            s += "Random Player"
                        elif game.currentPlayer == distribAI:
                            s += "Distrib Player"
                        elif game.currentPlayer == top5RandomAI:
                            s += "Top 5 Random AI"
                        elif game.currentPlayer == perfectAI:
                            s += "Perfect AI"
                        else:
                            s += "Weird Bozo"

                        print(f"{s} {log}")

                    if game.currentPlayer == game.findAI(actingNetwork):
                        if game.currentCity.name == "Seattle":
                            if game.roundNumber == 2:
                                # calculate if they won
                                game.countPoints()
                                scores = [(i, i.points, i.badInvestments, game.playerOrder.index(i)) for i in game.players]
                                scores.sort(key=lambda a: (-a[1], a[2], a[3]))  # sort by points (descending) and bad investments (ascending)
                                winner = scores[0][0]
                                reward = 9 if winner.nn == actingNetwork else -8  # TODO: don't discourage the only legal action being connecting(?)
                                experienceBuffer.append((state, chosenAction, reward, None, True))
                            else:
                                scores = [(i, len(i.cubes), game.getBadCubes()[i], game.playerOrder.index(i)) for i in game.players]
                                scores.sort(key=lambda a: (-a[1], a[2], a[3]))
                                winner = scores[0][0]
                                persistentReward += 7 if winner.nn == actingNetwork else -7

                        incompleteExperienceTuple = (state, chosenAction)

                    game.currentPlayer = game.playerOrder[(game.playerOrder.index(game.currentPlayer) + 1) % len(game.playerOrder)]

            scores = [(i, i.points, i.badInvestments, game.playerOrder.index(i)) for i in game.players]
            # sort by points (descending) and bad investments (ascending) and turn order
            scores.sort(key=lambda a: (-a[1], a[2], a[3]))

            winner = scores[0][0]

            if winner.nn.greedy:
                greedyAiWins += 1
            elif winner.nn.random:
                randomAiWins += 1
            elif winner.nn == actingNetwork:
                aiWins += 1
            elif winner.nn == laggingNetwork:
                laggingAiWins += 1
            elif winner.nn == formidableOpponent:
                formidableAiWins += 1
            elif winner.nn == distribAI:
                distribAiWins += 1
            elif winner.nn == top5RandomAI:
                top5RandomAiWins += 1
            elif winner.nn == perfectAI:
                perfectAiWins += 1

            # process experience buffer for games
            for i, v in enumerate(experienceBuffer):
                if LOOKFORWARD_STEPS + i >= len(experienceBuffer):
                    n = len(experienceBuffer) - i
                else:
                    n = LOOKFORWARD_STEPS

                # im so sorry for writing this line like this
                # basically, it takes the sum of all discounted returns (reward * gamma^(steps forward)),
                # then adds the discounted Q value of the step right after all calculated discounted returns,
                # using double DQN to get the target network's evaluation of the action that the active
                # network chose, and if it's the last step it ignores that
                discountedReturns = sum([(GAMMA**it) * experienceBuffer[i+it][2] for it in range(n)]) + (GAMMA ** n) * ((targetNetwork.output(experienceBuffer[i+n][0])[int(np.argmax(actingNetwork.output(experienceBuffer[i + n][0]).detach()))]) if i + n < len(experienceBuffer) else 0)
                fullExperienceBuffer.appendleft(list(v) + [discountedReturns, 0])

        # statistics!
        # TODO: fix possible division by zero here
        greedyWinRates.appendleft(greedyAiWins / greedyAiPlays)
        randomWinRates.appendleft(randomAiWins / randomAiPlays)
        aiWinRates.appendleft(aiWins / GAMES_PER_GEN)
        laggingWinRates.appendleft(laggingAiWins / laggingAiPlays)
        formidableWinRates.appendleft(formidableAiWins / formidableAiPlays)
        distribWinRates.appendleft(distribAiWins / distribAiPlays)
        top5RandomWinRates.appendleft(top5RandomAiWins / top5RandomAiPlays)
        perfectWinRates.appendleft(perfectAiWins / perfectAiPlays)

        tdErrors = [abs(actingNetwork.output(it[0])[it[1]].detach() - it[5]).detach() for it in fullExperienceBuffer]

        print(f"Last 5 Our AI winrate     = {np.average(aiWinRates):.2f} | Current : {aiWins:3}/{GAMES_PER_GEN:3} = {aiWinRates[0]:.2f}")
        print(f"Last 5 Perfect AI winrate = {np.average(perfectWinRates):.2f} | Current : {perfectAiWins:3}/{perfectAiPlays:3} = {perfectWinRates[0]:.2f}")
        print(f"Last 5 Greedy AI Winrate  = {np.average(greedyWinRates):.2f} | Current : {greedyAiWins:3}/{greedyAiPlays:3} = {greedyWinRates[0]:.2f}")
        print(f"Last 5 Random AI Winrate  = {np.average(randomWinRates):.2f} | Current : {randomAiWins:3}/{randomAiPlays:3} = {randomWinRates[0]:.2f}")
        print(f"Last 5 Lagging AI winrate = {np.average(laggingWinRates):.2f} | Current : {laggingAiWins:3}/{laggingAiPlays:3} = {laggingWinRates[0]:.2f}")
        print(f"Last 5 Formidable winrate = {np.average(formidableWinRates):.2f} | Current : {formidableAiWins:3}/{formidableAiPlays:3} = {formidableWinRates[0]:.2f}")
        print(f"Last 5 Distrib winrate    = {np.average(distribWinRates):.2f} | Current : {distribAiWins:3}/{distribAiPlays:3} = {distribWinRates[0]:.2f}")
        print(f"Last 5 top5random winrate = {np.average(top5RandomWinRates):.2f} | Current : {top5RandomAiWins:3}/{top5RandomAiPlays:3} = {top5RandomWinRates[0]:.2f}")
        print(f"Generation time: {(time.time() - start):.2f} seconds")
        print(f"Epsilon: {(EPSILON * (1 - aiWinRates[1]) if len(aiWinRates) > 1 else EPSILON):.3f}")


        print("Games finished. Training...")

        # training time

        # stats
        targetqValues = []

        for i in range(0, 10):
            dist = random.choices(fullExperienceBuffer, weights=tdErrors, k=round(len(fullExperienceBuffer)*0.2))
            for j in dist:
                # increment visits, for statistics
                # TODO: is there a better way to do this? is this slow?
                j[6] += 1

            # TODO: simplify this tuple structure since most are not needed at this point
            batch = [list(it) for it in zip(*dist)]
            state = batch[0]
            response = torch.tensor(batch[1])
            discountedReturns = batch[5]

            targetqValues.extend(discountedReturns)  # statistics

            stateActionValues = actingNetwork.output(state).gather(1, response.unsqueeze(0))

            # Compute Huber loss
            criterion = torch.nn.SmoothL1Loss()
            loss = criterion(stateActionValues, torch.tensor(discountedReturns).unsqueeze(0))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(actingNetwork.parameters(), 100)
            optimizer.step()

        if generation % CHECKPOINT == 0 and generation != 0:
            filename = f"checkpoints/{timestamp}-DQN-dueling-generation{generation}.pkl"  # TODO: serialize as list of floats
            with open(os.path.join(checkpointsDir, filename), "wb") as f:
                torch.save(actingNetwork, f)
                print(f"Generation done. Saved generation to {filename}.")
        else:
            print(f"Generation done.")

        targetqValues = [float(it) for it in targetqValues]
        print(f"Target Q Values min|avg|med|max|var: {min(targetqValues):.2f}|{np.average(targetqValues):.2f}|{np.median(targetqValues):.2f}|{max(targetqValues):.2f}|{np.var(targetqValues):.2f}")
        if len(fullExperienceBuffer) != BUFFER_SIZE:
            print(f"Buffer size: {len(fullExperienceBuffer)}")
        print(f"TD Errors min|avg|med|max|var : {min(tdErrors):.3f}|{np.average(tdErrors):.3f}|{np.median(tdErrors):.3f}|{max(tdErrors):.3f}|{np.var(tdErrors):.3f}")
        visits = [it[6] for it in fullExperienceBuffer]
        print(f"Experience Frame Visits min|avg|med|max|var: {min(visits)}|{np.average(visits):.2f}|{np.median(visits):.2f}|{max(visits)}|{np.var(visits):.2f} | Oldest 1000 visit average: {np.average(visits[-1000:]):.2f}")


# newTest()

cProfile.run("newTest()", sort='cumtime')
