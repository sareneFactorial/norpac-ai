from abc import ABC, abstractmethod

import torch

from ..newnorpac import NorpacGame, Player

# this will replace actors.py in the GUI refactor

# reportWin(Player)
# takes the winner
# used for the bots to apply their own rewards etc.

# train()
# trains the networks
# TODO: any parameters for this one?


class Agent(ABC):

    @abstractmethod
    def doAction(self, player: Player, game: NorpacGame) -> tuple[tuple[NorpacGame, str], int]:
        """
        takes the player to do an action and the current game state
        returns a tuple where the internal tuple is the state after doing the action + an optional log string, and the int in the external tuple is the action number done
        this internal tuple is returned by NorpacGame's action resolver
        """
        pass

    @abstractmethod
    def reportAction(self, player: Player, move: int):
        """
        takes the player and the move that they did
        this is used for the bots to keep track of game state
        """
        pass

    @abstractmethod
    def reportWin(self, winner: Player):
        """
        takes the winner
        used for the bots to apply their own rewards etc.
        """
        pass

    @abstractmethod
    def train(self):
        """
        train the networks
        TODO: any parameters for this one?
        """
        pass


class DeepQNetwork(Agent, torch.nn.module):

    def __init__(self, greedy=False, rand=False, torch=None):
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

    def doAction(self, player: Player, game: NorpacGame) -> tuple[tuple[NorpacGame, str], int]:
        out = self.output(game.createInput(player))
        sortedout = out.argsort().tolist()[::-1]
        legal = game.allLegalMoves()
        for i in sortedout:
            if i in legal:
                return game.doAction(player, i, " -DQN"), i
        raise Exception("ai has no valid moves")

    def reportAction(self, player: Player, move: int):
        pass

    def reportWin(self, winner: Player):
        pass

    def train(self):
        pass

