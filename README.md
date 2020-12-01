# RL LevelUp
Implementations of well known reinforcement learning algorithms in Python. Based
on [OpenAI's Spinning Up](https://github.com/openai/spinningup/tree/master/spinup). 
The goal of this repository is to learn to implement 
reinforcement learning algorithms from scratch, and to provide a set of experiments
to compare to other implementations using [OpenAI Gym](https://gym.openai.com/).

## Setup

Install by cloning the repository and running ```pip install -e .``` from the 
repository root.
 
Run ```jupyter-lab``` in a command line to start a local Jupyter notebook server. 
You can use it to evaluate experiments. Run `jupyter labextension install jupyterlab-plotly@4.13.0`
 to use plotly together with jupyter lab. Figures created with plotly do not
 show up on github. The experiments notebooks need to be rendered locally
 or should be viewed through a [notebook viewer](https://nbviewer.jupyter.org/).

### Windows
Make sure you install [Visual Studio C++ build tools](https://visualstudio.microsoft.com/downloads/) for OpenAI Gym if you are running Windows 10. 
You might also have to install [Swig](https://stackoverflow.com/questions/51811263/problems-pip-installing-box2d).

## Structure
- ```levelup``` - Implementations of reinforcement learning algorithms.
- ```experiments``` - Jupyter notebooks detailing experiments on implemented algorithms.
- ```warmup``` - Tutorial steps on how to use PyTorch. Check the
[PyTorch documentation pages](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)
for more details.

