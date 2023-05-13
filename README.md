# norpac-ai
a test neural network for the board game Northern Pacific


# requirements
```
numpy
pytorch
pygame (optional)
```
for the older stuff you will also need `numba`

# how to use
run `pytorchtest.py` to start training- hyperparameters are in constants near the top of the file.
checkpoints saved to checkpoints/ every 20 generations by default

the network structure is semi-rainbow DQN

to see progress, `ui.py` lets you play against the AI. you might need to figure this one out yourself for now, the way you set it up is janky


# todo
- clean up code
- document a bit more
- implement distributional DQN
- implement noise layer
- add more checkpoints
- decouple game logic from AIs
- optimize code