# Bachelor-Thesis---Autonomous-Land-Sailing
In this code I use reinforcement learning to train an abstract modell of a land sailing robot. Further details are explained in my thesis.

# Use
## Sailing_Env_v0.py
Running this file will simply run through an episode, where all actions are chosen at random.

## Sailing_Agent.py
Running this file will give a promt suggesting two opptions: train and test. <br>
### Training
When train, an agent will be trained. The corresponding hyperparameters can be chosen in main, although the stock hyperparameters already give sufficient results.

### Testing
When testing, the Agent performes an episode which is rendered to the human, where all actions are chosen optimal. For this process, the Q-Matrix from the last training process will be used. The stock Q-Matrix already gives sufficient results.
