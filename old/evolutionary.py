import norpac
import neuralnet
import random
import pickle
from datetime import datetime
import cProfile
import time

MAX_POP = 25
GAMES_PER_GEN = 250
NUM_GENS = 200
MUT_RATE = 0.15
population = [neuralnet.NeuralNet() for x in range(0, MAX_POP)]

timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')

allRandomAiWinrates = []
allGreedyWinrates = []
alltimes = []


def fitnessFunction(score, gamesPlayed, totalWinMargin, badInvestments):
    return (score / gamesPlayed)          * 1.2               \
         + (totalWinMargin / (score + 1)) / 10                \
         - (badInvestments / gamesPlayed) / 15

def test():
    for generation in range(0,NUM_GENS+1):
        print(f"{datetime.now().strftime('%H:%M:%S.%f')} | Generation {generation}")
        start_time = time.time()
        if generation % 50 == 0 and generation != 0:
            filename = f"{timestamp}generation{generation}.pkl"
            with open(filename, 'wb') as f:  # open a text file
                pickle.dump(population, f)  # serialize the list
                print(f"Saved generation to {filename}")

        # performance stats
        randomAiWins = 0
        randomAiPlays = 0
        greedyAiWins = 0
        greedyAiPlays = 0

        tally = {}
        for i in population:
            tally[i] = {"score": 0, "gamesPlayed": 0, "totalWinMargin": 0, "badInvestments": 0}

        for game_number in range(0,GAMES_PER_GEN):
            game = norpac.NorpacGame()

            debugGame = generation % 5 == 0 and game_number == GAMES_PER_GEN-1

            if debugGame:
                fitness = [(k, fitnessFunction(v["score"], v["gamesPlayed"], v["totalWinMargin"], v["badInvestments"])) for k, v in tally.items()]
                fitness.sort(key=lambda a: a[1])
                fitness = fitness[::-1]
                nns = [fitness[0][0], fitness[-1][0], fitness[10][0], fitness[9][0]]
            elif random.random() < 0.75:  # about 75% of the time, one player will be a random ai
                nns = random.choices(population, weights=[1/(it["gamesPlayed"]+1) for _,it in tally.items()], k=3)
                if random.random() < 0.3:  # random ai
                    nns.append(neuralnet.NeuralNet())
                    randomAiPlays += 1
                else:  # greedy ai
                    nns.append(neuralnet.NeuralNet(greedy=True))
                    greedyAiPlays += 1
            else:
                nns = random.choices(population, weights=[1 / (it["gamesPlayed"] + 1) for _, it in tally.items()], k=4)

            for i in nns:  # TODO: fix inputting players
                game.players.append(norpac.Player(i))
                try:
                    tally[i]["gamesPlayed"] += 1
                except:
                    pass
            game.setupGame(3)

            f = None
            if debugGame:
                f = open(f"LOG-{timestamp}-generation{generation}.txt", "w")

            for r in range(0, 3):
                if debugGame:
                    f.write(f"Round {r}:\n")
                game.clearGame()
                game.roundNumber = r

                while game.currentCity.name != "Seattle":
                    nn = game.currentPlayer.nn
                    out = nn.output(nn.createInput(game))
                    log = nn.doAction(game, out)
                    game.currentPlayer = game.playerOrder[(game.playerOrder.index(game.currentPlayer) + 1) % len(game.playerOrder)]
                    if debugGame:
                        f.write("Player: " + str(game.currentPlayer)[::-1][1:6][::-1] + " " + log + "\n")

                game.countPoints()

            scores = []
            for i in game.players:
                scores += [(i, i.points, i.badInvestments)]
            scores.sort(key=lambda a: a[1])
            scores = scores[::-1]

            winners = list(filter(lambda x: (x[1] == scores[0][1]), scores))
            winner = None
            if len(winners) > 1:
                winners.sort(key=lambda a: a[2])  # bad investments tiebreaker
                # TODO: turn order tiebreaker
                realWinners = list(filter(lambda x: (x[2] == winners[0][2]), winners))
                winner = realWinners[0]
            else:
                winner = winners[0]

            try:
                tally[winner[0].nn]["score"] += 1
                uniqueScores = list(set([x[2] for x in scores]))
                uniqueScores.sort(reverse=True)
                if len(uniqueScores) <= 1:
                    score_diff = 0
                else:
                    score_diff = uniqueScores[0] - uniqueScores[1]
                tally[winner[0].nn]["totalWinMargin"] += score_diff
            except KeyError:
                if winner[0].nn.greedy:
                    greedyAiWins += 1
                else:
                    randomAiWins += 1
            for i in game.players:
                try:
                    tally[i.nn]["badInvestments"] += i.badInvestments
                except:
                    continue

            if debugGame:
                st = ""
                for i in game.cities:
                    st += f"{i.name}: {len(i.cubes)}|"
                f.write(st + "\n")
                for i in game.players:
                    f.write(f"player {str(i)[::-1][1:6][::-1]} (aka AI {str(i.nn)[::-1][1:6][::-1]}): {i.points} points\n")
                f.write(f"Winner: {str(winner[0])[::-1][1:6][::-1]} with {winner[0].points} points")
            if f is not None:
                f.close()
                print("Game log saved")

        fitness = [(k, fitnessFunction(v["score"], v["gamesPlayed"], v["totalWinMargin"], v["badInvestments"])) for k,v in tally.items()]
        fitness.sort(key=lambda a: a[1])
        fitness = fitness[::-1]

        i = 2
        dead = []
        deadIndices = []
        while i < len(fitness):
            dieChance = 0.70 * (i / len(fitness)) + 0.10
            if random.random() < dieChance:
                dead += [fitness[i]]
                # print(f"died! {i}")
            i += 1
        for i in dead:
            deadIndices.append(fitness.index(i))
        for i in dead:
            population.remove(i[0])
            fitness.remove(i)

        for i in range(0,2):
            j = random.randrange(2, len(fitness))
            population.remove(fitness[j][0])
            fitness.remove(fitness[j])

        for i in range(0, MAX_POP-len(population)):
            roll = random.random()
            if roll > 0.3:
                species = random.choices(population, weights=[1/(x+2) for x in range(0,len(population))], k=2)
                population.append(neuralnet.breed(species[0], species[1], MUT_RATE))
                # breed
            elif roll > 0.1:
                species = random.choices(population, weights=[1 / (x + 1.5) for x in range(0, len(population))], k=1)
                population.append(neuralnet.vary(species[0]))
            else:
                population.append(neuralnet.NeuralNet())

        end_time = time.time()

        # statistics
        deadIndices.sort()
        duration = end_time - start_time
        winrar = tally[fitness[0][0]]
        allRandomAiWinrates.append(randomAiWins/max(1, randomAiPlays))
        allGreedyWinrates.append(greedyAiWins/max(1, greedyAiPlays))
        alltimes.append(duration)
        print(f"Total time taken: {duration} seconds")
        print(f"Most Fit AI stats: Winrate: {winrar['score']}/{winrar['gamesPlayed']} = {winrar['score']/winrar['gamesPlayed']} | Average win margin: {winrar['totalWinMargin']/winrar['score']} | Average Bad Investments: {winrar['badInvestments'] / winrar['gamesPlayed']}")
        print(f"Fitness function thereof: {fitnessFunction(winrar['score'], winrar['gamesPlayed'], winrar['totalWinMargin'], winrar['badInvestments'])}")
        print(f"Random AI Winrate: {randomAiWins}/{randomAiPlays} = {randomAiWins/randomAiPlays}")
        print(f"Greedy AI Winrate: {greedyAiWins}/{greedyAiPlays} = {greedyAiWins/greedyAiPlays}")
        print(f"Population deaths: {len(dead)}. Indices: {deadIndices}")
        if generation % 5 == 0 and generation != 0:
            print()
            print("AVERAGE STATS LAST 5 GENERATIONS")
            last5RandomAi = sum(allRandomAiWinrates[-5:])/5
            last5Greedy = sum(allGreedyWinrates[-5:])/5
            print(f"Last 5 Random AI Winrate: {last5RandomAi} | Last 5 Greedy Winrate: {last5Greedy}")
            last5Times = sum(alltimes[-5:])/5
            print(f"Last 5 generations time: {last5Times} seconds")
        print()
        print("-------------------------------------")
        print()




    fitness = [(k, v["score"]/v["gamesPlayed"]) for k, v in tally.items()]
    fitness.sort(key=lambda a: a[1])
    fitness = fitness[::-1]



    print(fitness[0])

    game = norpac.NorpacGame()
    game.players.append(norpac.Player(fitness[0][0]))
    game.players.append(norpac.Player(fitness[1][0]))
    game.players.append(norpac.Player(fitness[2][0]))
    game.players.append(norpac.Player(fitness[3][0]))
    game.setupGame(3)

    for r in range(0, 3):
        print(f"Round {r}")
        game.clearGame()
        game.roundNumber = r

        while game.currentCity.name != "Seattle":
            print("Player: " + str(game.currentPlayer)[::-1][1:4][::-1])
            nn = game.currentPlayer.nn
            out = nn.output(nn.createInput(game))
            nn.doAction(game, out, True)
            game.currentPlayer = game.playerOrder[
                (game.playerOrder.index(game.currentPlayer) + 1) % len(game.playerOrder)]

        game.countPoints()

    for i in game.players:
        print(f"player {str(i)[::-1][1:4][::-1]}: {i.points} points")

test()

# cProfile.run("test()")
