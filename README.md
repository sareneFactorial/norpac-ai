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
there are three kinds of opponents:
normal AI: does top valued action
distrib AI: uses values as a distribution
top5random AI: takes a random action from the top 5 highest valued actions


# todo
- clean up code
- document a bit more
- implement distributional DQN
- implement noise layer
- add more checkpoints
- decouple game logic from AIs
- optimize code
- add cuda alternative for pytorchtest or make it work for both
- make all the variable/function names more comprehensible to anyone that's not me
- revise reward function; probably don't penalize losing
- let ui.py track & save an experience buffer for further training