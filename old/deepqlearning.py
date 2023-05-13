import newnn
import norpac
import newnn as neuralnet
import random
import pickle
from datetime import datetime
import numpy as np
import collections
import time
import cProfile


GAMES_PER_GEN = 100
NUM_GENS = 1000
LEARN_RATE = 0.000001
TEST_GAMES = 150
LOOKFORWARD_STEPS = 4  # how many steps to look forward with multi step learning

PROGRESS_UPDATES = 5000
BUFFER_SIZE = 10000
CHECKPOINT = 20

EPSILON = 0.32  # likelihood of AI doing a random move. reduced by winrate
GAMMA = 0.95  # how much the next state is counted for the reward

timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')

fullExperienceBuffer = collections.deque(maxlen=BUFFER_SIZE)



# TODO: plot this on a graph while it's training along with data like average reward, average scores, etc.
aiWinRates = collections.deque(maxlen=5)
greedyWinRates = collections.deque(maxlen=5)
randomWinRates = collections.deque(maxlen=5)
laggingWinRates = collections.deque(maxlen=5)

# older AIs for baseline comparison
laggingAis = collections.deque(maxlen=10)
with open("2023-05-07-172642-DQN-newnn-generation60.pkl", "rb") as f:
    laggingAis.appendleft(pickle.load(f))

with open("2023-05-09-134055-DQN-newnn-generation20.pkl", "rb") as f:
    actingNetwork = pickle.load(f)

# actingNetwork = neuralnet.NeuralNet([1217, 500, 200, 100])

def newTest():
    for generation in range(0, NUM_GENS+1):
        print(f"\nGeneration {generation}")

        start = time.time()

        # double dqn baybee
        targetNetwork = actingNetwork.copy()

        # lagging network for baseline
        laggingAis.appendleft(actingNetwork.copy())
        laggingNetwork = laggingAis[-1]

        # just to initialize this because its jank
        otherPlayerBads_new = 0
        pointsNow = 0

        # used for calculating a reward that is "immediate" but only applied on the next transition
        persistentReward = 0

        for gameNumber in range(0,GAMES_PER_GEN+1):  # TODO: move the +1 to the visual representations of the number
            playerCount = random.randrange(3, 7)  # player count from 3 to 6

            if gameNumber % PROGRESS_UPDATES == 0 and gameNumber != 0:
                print(f"Game #{gameNumber}....")
            game = norpac.NorpacGame()

            # TODO: create better random opponents by using q values as a distribution, or sampling from top n actions
            candidates = [laggingNetwork, actingNetwork.copy().vary(0.1), newnn.NeuralNet([1217, 150, 125, 100]),
                          neuralnet.NeuralNet(rand=True), neuralnet.NeuralNet(greedy=True), neuralnet.NeuralNet(greedy=True),]

            nns = random.sample(candidates, playerCount-1)
            nns.append(actingNetwork)
            random.shuffle(nns)

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
                        if random.random() < (EPSILON * (1 - aiWinRates[0]) if len(aiWinRates) > 0 else EPSILON):
                            random.shuffle(out)

                        otherPlayerPoints_new = sum([v for k, v in game.lastScore.items() if k != game.currentPlayer])
                        activePlayerPoints_new = game.lastScore.get(game.currentPlayer, 0) if game.lastScore.get(
                            game.currentPlayer) is not None else 0
                        game.lastScore.clear()
                        activePlayerBads_new = game.getBadCubes()[game.currentPlayer] - activePlayerBads_old

                        reward = (activePlayerPoints_new - (otherPlayerPoints_new * 1.2)) + (
                                    (otherPlayerBads_new * 0.5) - (activePlayerBads_new * 1.2)) * 1 + persistentReward

                        persistentReward = 0

                        if len(incompleteExperienceTuple) != 0:
                            incompleteExperienceTuple += (reward, state, False)
                            experienceBuffer.append(incompleteExperienceTuple)

                        activePlayerBads_old = game.getBadCubes()[game.currentPlayer]
                        otherPlayerBads_old = sum([v for (k, v) in game.getBadCubes().items() if k != game.currentPlayer])
                        pointsNow = len(game.currentPlayer.cubes)

                    chosenAction = nn.firstLegal(game, out)

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
                                reward = 7 if winner.nn == actingNetwork else -7

                        incompleteExperienceTuple = (state, chosenAction)

                    game.currentPlayer = game.playerOrder[(game.playerOrder.index(game.currentPlayer) + 1) % len(game.playerOrder)]

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
                discountedReturns = sum([(GAMMA**it) * experienceBuffer[i+it][2] for it in range(n)]) + (GAMMA ** n) * ((targetNetwork.output(experienceBuffer[i+n][0])[int(np.argmax(actingNetwork.output(experienceBuffer[i + n][0])))]) if i + n < len(experienceBuffer) else 0)
                fullExperienceBuffer.appendleft(list(v) + [discountedReturns, 0])

        print("Games finished. Training...")

        # training time

        # stats
        targetqValues = []

        for i in range(0, 3):
            dist = random.choices(fullExperienceBuffer, weights=[abs(actingNetwork.output(it[0])[it[1]] - it[5]) for it in fullExperienceBuffer], k=round(len(fullExperienceBuffer)*0.5))
            for j in dist:
                # increment visits, for statistics
                # TODO: is there a better way to do this? is this slow?
                j[6] += 1
            # TODO: simplify this tuple structure since most are not needed at this point
            for frame in dist:
                state = frame[0]
                response = frame[1]
                reward = frame[2]
                nextState = frame[3]
                done = frame[4]
                discountedReturns = frame[5]
                visits = frame[6]
                # if done:  # if done
                #     target_q_value = reward
                # else:
                    # max_next_q_value = np.max(targetNetwork.output(nextState))
                    # target_q_value = reward + GAMMA * max_next_q_value - chosenOne.output(state)[response]

                target_q_value = discountedReturns
                targetqValues.append(target_q_value)

                actingNetwork.Q_gradientDescent(state, response, target_q_value, LEARN_RATE)

        if generation % CHECKPOINT == 0 and generation != 0:
            filename = f"{timestamp}-DQN-newnn-generation{generation}.pkl"  # TODO: serialize as list of floats
            with open(filename, 'wb') as f:  # open a text file
                pickle.dump(actingNetwork, f)  # serialize the list
                print(f"Generation done. Saved generation to {filename}. Testing winrates....")
        else:
            print(f"Generation done. Testing winrates....")

        # TODO: output game logs
        randomAiPlays = 0
        randomAiWins = 0
        greedyAiPlays = 0
        greedyAiWins = 0
        laggingAiPlays = 0
        laggingAiWins = 0
        aiWins = 0
        for test_game in range(0, TEST_GAMES):
            playerCount = random.randrange(3, 7)  # player count from 3 to 6
            game = norpac.NorpacGame()

            candidates = [laggingNetwork, actingNetwork.copy().vary(0.1), newnn.NeuralNet([1217, 150, 125, 100]),
                          neuralnet.NeuralNet(rand=True), neuralnet.NeuralNet(greedy=True),
                          neuralnet.NeuralNet(greedy=True), ]

            nns = random.sample(candidates, playerCount - 1)
            nns.append(actingNetwork)
            random.shuffle(nns)

            if any(it.random for it in nns):
                randomAiPlays += 1
            if any(it.greedy for it in nns):
                greedyAiPlays += 1
            if any(it == laggingNetwork for it in nns):
                laggingAiPlays += 1

            otherBozo = None

            for i in nns:
                game.players.append(norpac.Player(i))
                # one random non-acting network is semi-random sometimes
                if otherBozo is None and i != actingNetwork and random.random() < 0.1 and not i.greedy and not i.random:
                    otherBozo = i
            game.setupGame(playerCount)

            for r in range(0, 3):
                game.clearGame()
                game.roundNumber = r

                while game.currentCity.name != "Seattle":
                    nn = game.currentPlayer.nn
                    out = nn.output(nn.createInput(game))

                    # other bozo is a copy of the active network with semi-random actions
                    if nn == otherBozo and random.random() < 0.35:
                        random.shuffle(out)

                    log = nn.doAction(game, out)
                    game.currentPlayer = game.playerOrder[
                        (game.playerOrder.index(game.currentPlayer) + 1) % len(game.playerOrder)]
                game.countPoints()

            scores = [(i, i.points, i.badInvestments, game.playerOrder.index(i)) for i in game.players]
            scores.sort(
                key=lambda a: (-a[1], a[2], a[3]))  # sort by points (descending) and bad investments (ascending) and turn order

            winner = scores[0][0]

            if winner.nn.greedy:
                greedyAiWins += 1
            elif winner.nn.random:
                randomAiWins += 1
            elif winner.nn == actingNetwork:
                aiWins += 1
            elif winner.nn == laggingNetwork:
                laggingAiWins += 1

        # TODO: possible divide by zero error here. fix it
        greedyWinRates.appendleft(greedyAiWins / greedyAiPlays)
        randomWinRates.appendleft(randomAiWins / randomAiPlays)
        aiWinRates.appendleft(aiWins/TEST_GAMES)
        laggingWinRates.appendleft(laggingAiWins / laggingAiPlays)

        tdErrors = [abs(actingNetwork.output(it[0])[it[1]] - it[5]) for it in fullExperienceBuffer]

        # statistics!
        print(f"Last 5 Our AI winrate     = {np.average(aiWinRates):.2f} | Current : {aiWins}/{TEST_GAMES} = {aiWinRates[0]:.2f}")
        print(f"Last 5 Greedy AI Winrate  = {np.average(greedyWinRates):.2f} | Current : {greedyAiWins}/{greedyAiPlays} = {greedyWinRates[0]:.2f}")
        print(f"Last 5 Random AI Winrate  = {np.average(randomWinRates):.2f} | Current : {randomAiWins}/{randomAiPlays} = {randomWinRates[0]:.2f}")
        print(f"Last 5 Lagging AI winrate = {np.average(laggingWinRates):.2f} | Current : {laggingAiWins}/{laggingAiPlays} = {laggingWinRates[0]:.2f}")
        print(f"Generation time: {(time.time() - start):.2f} seconds")
        print(f"Epsilon: {(EPSILON * (1 - aiWinRates[1]) if len(aiWinRates) > 1 else EPSILON)}")
        if len(fullExperienceBuffer) != BUFFER_SIZE:
            print(f"Buffer size: {len(fullExperienceBuffer)}")
        print(f"TD Errors min|avg|med|max|var : {min(tdErrors):.3f}|{np.average(tdErrors):.3f}|{np.median(tdErrors):.3f}|{max(tdErrors):.3f}|{np.var(tdErrors):.3f}")
        visits = [it[6] for it in fullExperienceBuffer]
        print(f"Experience Frame Visits min|avg|med|max|var: {min(visits):}|{np.average(visits):.2f}|{np.median(visits):}|{max(visits)}|{np.var(visits):.2f} | Oldest 1000 visit average: {np.average(visits[-1000:]):.2f}")
        print(f"Target Q Values min|avg|med|max|var: {min(targetqValues):.2f}|{np.average(targetqValues):.2f}|{np.median(targetqValues):.2f}|{max(targetqValues):.2f}|{np.var(targetqValues):.2f}")


newTest()

# cProfile.run("newTest()", sort='cumtime')
