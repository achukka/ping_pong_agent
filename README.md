## Ping Pong Agent

## Overview
This repository contains code for simulating ping pong game using [pygame](http://www.pygame.org/wiki/GettingStarted). In this setup, we have an Evil Agent and an RL agent who learns the actions with help of **[Deep Q Network](https://deepmind.com/research/dqn/)**. The agent gets better by maximising its score through trial and error.
The DQN is a *Convolutional Neural Network* that reads raw pixel data from the pygame and game score. With these parameters it learns the actions(moves) that maximizes its score. 

## Dependencies
This code is written in python. To use it you will need:
- Python 2.7
- [Tensorflow] (https://www.tensorflow.org)
- [OpenCV] (http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html)
- [Pygame](http://www.pygame.org/wiki/GettingStarted)
- [NumPy](https://www.scipy.org/install.html)

## Usage
You can run the agent by the following command:
```python
python rl_algo.py
```

## Credits
This code was by [malreddysid](https://github.com/malreddysid/pong_RL) I've merely wrapped and updated it. 
