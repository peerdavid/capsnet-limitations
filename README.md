# Limitation of capsule networks
A recent development in deep learning groups multiple neurons to capsules such that each capsule represents an object or part of an object. Routing algorithms route the output of capsules from lower-level layers to upper-level layers. We proved that state-of-the-art routing procedures decrease the expressivity of capsule networks. More precisely, it can be shown that EM-routing and routing-by-agreement hinder capsule networks from distinguishing inputs and their negative counterpart. This also affects the training of deep capsule networks negatively. Therefore, we add a bias term to state-of-the-art routing algorithms that (1) solves the aforementioned limitation (2)) stabilizes the training of capsule networks and additionally capsule networks with a bias term generalize better and faster. All experiments can be easily executed using this repository.

*This is the official implementation of the paper "Limitation of Capsule Networks".*

    @misc{peer2021limitationofcapsulenetworks,
        title = "Limitation of Capsule Networks",
        journal = "Pattern Recognition Letters",
        year = "2021",
        issn = "0167-8655",
        doi = "https://doi.org/10.1016/j.patrec.2021.01.017",
        url = "http://www.sciencedirect.com/science/article/pii/S0167865521000301",
        author = "David Peer and Sebastian Stabinger and Antonio Rodríguez-Sánchez",
    }


# Setup
For this installation we assume that python3, pip3 and all nvidia drivers
(GPU support) are already installed. Then execute the following
to create a virtual environment and install all necessary packages:

1. Create virtual environment: ```python3 -m venv env```
2. Activate venv: ```source env/bin/activate```
3. Update your pip installation: ```pip3 install --upgrade pip```
4. Install all requirements. Use requirements-gpu if a gpu is available, requirements-cpu otherwise: ```pip3 install -r requirements.txt```


# Execute experiments
To run the sign experiments use ```train_sgn.py``` and for the depth 
experiment the ```train.py``` file. All parameters (depth, num capsules, 
dimensions etc.) can be changed using command arguments. For example to 
enable a bias term run ```train.py --use_bias=True```

To evaluate the results after the training tensorboard can be used.
Note also that we assume a multi GPU setup for the training and therefore we use the 
distr. API from TensorFlow 2.0. To run experiments on multiple nodes the run_depth_experiment.sh
file can be split (e.g. depth per node or RBA/EM per node etc.). 
We also watn to mention that the same source will also be uploaded 
to GitHub to make all experiments, the architecture and all hyperparameters available and
reproducable.
