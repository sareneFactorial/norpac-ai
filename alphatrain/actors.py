import random

import numpy as np
import torch

import newnorpac
import norpac

# i hate python not having interfaces and "duck typing" (terrible way to do it)
# here is the Actor duck type:
# it has a function of signature:
# doAction(self, Player, NorpacGame) -> tuple[tuple[NorpacGame, str], int]:
# where the internal tuple is the result of doAction, and the int in the external tuple is the action number done


class DeepQNetwork(torch.nn.Module):
    def __init__(self, greedy=False, rand=False):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1217, 500),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(500, 200),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(200, 100)
        )

        self.greedy = greedy
        self.random = rand

    def output(self, x):
        logits = self.linear_relu_stack(torch.tensor(x))
        return logits

    def doAction(self, player: newnorpac.Player, game: newnorpac.NorpacGame) -> tuple[tuple[newnorpac.NorpacGame, str], int]:
        out = self.output(game.createInput(player))
        sortedout = out.argsort().tolist()[::-1]
        legal = game.allLegalMoves()
        for i in sortedout:
            if i in legal:
                return game.doAction(player, i, " -DQN"), i
        raise Exception("ai has no valid moves")


# TODO: figure out how to do polymorphism in python
# code heavily referenced from https://github.com/Curt-Park/rainbow-is-all-you-need
class DuelingNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.greedy = False
        self.random = False

        self.feature_layer = torch.nn.Sequential(
            torch.nn.Linear(1217, 500),
            torch.nn.LeakyReLU(negative_slope=0.1),
        )
        self.advantage_layer = torch.nn.Sequential(
            torch.nn.Linear(500, 200),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(200, 100),
        )
        self.value_layer = torch.nn.Sequential(
            torch.nn.Linear(500, 128),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(128, 1),
        )

    def output(self, x) -> torch.Tensor:
        feature = self.feature_layer(torch.tensor(x))

        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q

    def doAction(self, player: newnorpac.Player, game: newnorpac.NorpacGame) -> tuple[tuple[newnorpac.NorpacGame, str], int]:
        out = self.output(game.createInput(player))
        sortedout = out.argsort().tolist()[::-1]
        legal = game.allLegalMoves()
        for i in sortedout:
            if i in legal:
                return game.doAction(player, i, " -dueling DQN"), i
        raise Exception("ai has no valid moves")


class RandomAI:
    def doAction(self, player: newnorpac.Player, game: newnorpac.NorpacGame) -> tuple[tuple[newnorpac.NorpacGame, str], int]:
        legal = game.allLegalMoves(player)
        random.shuffle(legal)
        i = legal[0]
        return game.doAction(player, i), i

# TODO: implement greedy bot
