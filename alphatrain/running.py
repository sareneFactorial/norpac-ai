import os
import pickle
import random
from datetime import datetime

import torch
from torch import optim as optim

import newnorpac
from alphatrain import actors
from alphatrain.tk import LEARN_RATE, NUM_GENS, GAMES_PER_GEN, checkpointsDir
from alphatrain.mcts import ValueNetwork, PolicyNetwork, MCTSBot


def test():
    criterion = torch.nn.SmoothL1Loss()

    valueNetwork = ValueNetwork(criterion)
    valueOptimizer = optim.AdamW(valueNetwork.parameters(), lr=LEARN_RATE, amsgrad=True)
    valueNetwork.optimizer = valueOptimizer

    policyNetwork = PolicyNetwork(criterion)
    policyOptimizer = optim.AdamW(policyNetwork.parameters(), lr=LEARN_RATE, amsgrad=True)
    policyNetwork.optimizer = policyOptimizer

    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')

    # null game because of the dependency hell i've created with Tree
    # TODO: fix this dependency hell
    nullGame = newnorpac.newGame([newnorpac.Player(actors.RandomAI()), newnorpac.Player(actors.RandomAI()), newnorpac.Player(actors.RandomAI())])

    trainingBot = MCTSBot(nullGame, policyNetwork, valueNetwork, True)
    trainingPlayer = newnorpac.Player(trainingBot)

    guaranteedActors = [
        trainingBot
    ]

    actorsList = [
        actors.RandomAI(),
        actors.RandomAI(),
        actors.RandomAI(),
        actors.RandomAI(),
        actors.RandomAI(),
        actors.RandomAI(),
    ]

    for generation in range(0, NUM_GENS):
        botWins = 0
        games = 0
        print(f"Generation {generation}:")

        # TODO: self-play against previous versions
        for g in range(0, GAMES_PER_GEN):
            games += 1
            players = []
            for i in range(0, random.randrange(3, 5)):
                players.append(newnorpac.Player(actors.RandomAI()))
            players.append(trainingPlayer)

            currentGame = newnorpac.newGame(players)
            for i in players:
                if hasattr(i.actor, "newGame"):
                    i.actor.newGame(currentGame)

            # _actingPlayer = random.choice(tree.game.players)
            # actingPlayerIndex = tree.game.players.index(_actingPlayer)

            # run game
            while not currentGame.terminalState:
                # neural network player
                player = currentGame.currentPlayer
                tuple1, moveNum = player.actor.doAction(player, currentGame)
                currentGame, _ = tuple1
                # print(newnorpac.readOutput(moveNum))
                # print(player)
                # print(player.actor)
                # print()
                for i in players:
                    if hasattr(i.actor, 'externalAction'):
                        i.actor.externalAction(player, moveNum)

            # check winner
            scores = [(i, currentGame.points.get(i, 0), currentGame.badInvestments.get(i, 0), currentGame.playerOrder.index(i)) for i in currentGame.players]
            scores.sort(key=lambda a: (
                -a[1], a[2], a[3]))  # sort by points (descending) and bad investments (ascending)
            winner = scores[0]
            if winner[0] == trainingPlayer:
                botWins += 1
            for i in players:
                if hasattr(i.actor, 'reportWin'):
                    i.actor.reportWin(winner)

        print(f"Games done. Winrate: {(botWins/games)*100:.3f}%. Training...")

        # training policy & value networks
        trainingBot.trainNetworks(10)
        # TODO: also train rainbow dqn along with this for benchmark

        if generation % 10 == 0 and generation != 0:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
            filename = f"{timestamp}-mcts-{generation}.pkl"
            with open(os.path.join(checkpointsDir, filename), "wb") as f:
                torch.save(trainingBot, f)
            print(f"Generation done. Saved generation to {filename}.")
        else:
            print(f"Generation done.")

        filename = f"{timestamp}-mcts-{generation}-trainingdata.pkl"
        with open(os.path.join(checkpointsDir, filename), "wb") as f:
            pickle.dump((trainingBot.policyTraining, trainingBot.valueTraining), f)

        print("Full average loss over training data:")
        valueBatch = [list(it) for it in zip(*trainingBot.valueTraining)]
        valueOutput = trainingBot.valueModel.output(valueBatch[0])
        valueLoss = criterion(valueOutput, torch.tensor(valueBatch[1]))
        print(f"Value: {valueLoss}")
        policyBatch = [list(it) for it in zip(*trainingBot.policyTraining)]
        policyOutput = trainingBot.policyModel.output(policyBatch[0])
        policyLoss = criterion(policyOutput, torch.tensor(policyBatch[1]))
        print(f"Policy: {policyLoss}")

    # at the end
    filename = f"{timestamp}-mcts-final-policy.pkl"
    with open(os.path.join(checkpointsDir, filename), "wb") as f:
        torch.save(policyNetwork, f)
    filename = f"{timestamp}-mcts-final-value.pkl"
    with open(os.path.join(checkpointsDir, filename), "wb") as f:
        torch.save(valueNetwork, f)

    print(f"Training done. Saved generation to {filename}.")
