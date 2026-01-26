# Demo training task : MNIST

## Overview

**We are going to train a common model between three simulated clients on the
classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/), deploying the
server and clients on Grid5000**. The input of the
model is a set of images of handwritten digits, and the model needs to
determine to which digit between $0$ and $9$ each image corresponds.  

This example is an adaptation of the `mnist/` example (so you can refer to it
for more details on the experiment), but with a deployment
on the cloud environment [Grid5000](https://www.grid5000.fr/w/Grid5000:Home)
using the
[`enoslib`](https://discovery.gitlabpages.inria.fr/enoslib/) Python library.


## Prerequisites

To run this experiment :

- You need to have access to Grid5000 and be able to reserve machines on it.
See the
[Getting Started instructions](https://www.grid5000.fr/w/Getting_Started) for
details.

- Then, you need to have access to a Grid5000 network
[storage](https://www.grid5000.fr/w/Storage) to share the experiment resources 
across all involved machines. The best setup is to have access to a
[Group Storage](https://www.grid5000.fr/w/Group_Storage), which is accessible 
from both Grid5000 front-end machines and reserved nodes, no matter the site.  

- Once, you have access to a network storage, log in to a front-end machine,
move to the storage path and create a directory specific to this Declearn
experiment.

- Take note of its absolute path, it will be used to run the experiment.  

Example of network storage path : 
`/srv/storage/my_storage_name@storage1.lille.grid5000.fr/my_user/declearn-grid5k`.


## Example contents

In this example folder :

The main script `run_grid5k.py` deploys on Grid5000 a FL experiment using the
MNIST dataset machines. **It is the only script you will need to run**.  

All the other scripts are not aimed to be launched on your local machine. They
are here to be easily found, downloaded (using git) and run by Grid5000 machines
reserved for the experiment :
- the `generate_ssl.py`, `prepare_data.py` and `run_server.py` scripts will be
run by the server machine.
- the `run_client.py` script will be run by each client machine.

#### Notes on default deployment
In the script `run_grid5k.py`, with default configuration, we run an
experiment with a server and 3 clients, using 3 machine clusters :  
- `nova` (Lyon)
- `dahu` (Grenoble)
- `econome` (Nantes)

The server runs on `nova`, one client runs on `dahu` and two clients run on two
distinct `econome` machines.



## Deploy the experiment on Grid5000

To run a FL experiment with a Grid5000 deployment :

- Clone the declearn repo (you may specify a given release branch or tag) :

```bash
git clone git@gitlab.inria.fr:magnet/declearn/declearn2.git declearn
```

- Create a dedicated virtual environment.

- Install declearn in it from the local repo :

```bash
cd declearn && pip install ".[g5k]" && cd ..
```

- Set up your
[EnOSlib configuration](https://discovery.gitlabpages.inria.fr/enoslib/tutorials/grid5000.html#configuration).

- Optionally, you can modify the constants at the top of the `run_grid5k.py`
script, if you want to change the configuration of the default experiment
(number of clients, clusters on which they are deployed, etc.).  
See the documentation inside the script for more details.

- Run the main script with your network storage path as argument :
```bash
cd declearn/examples/mnist_g5k
python run_grid5k.py STORAGE_PATH
```

- When the experiment is finished, you can retrieve the logs and results on your
network storage path, in the `results/result_{timestamp}` folder.
