# Demo training task : MNIST

## Overview

**We are going to train a common model between three simulated clients on the
classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/)**. The input of the
model is a set of images of handwritten digits, and the model needs to
determine to which digit between $0$ and $9$ each image corresponds.

## Setup

To be able to experiment with this tutorial:

* Clone the declearn repo (you may specify a given release branch or tag):

```bash
git clone git@gitlab.inria.fr:magnet/declearn/declearn2.git declearn
```

* Create a dedicated virtual environment.
* Install declearn in it from the local repo:

```bash
cd declearn && pip install .[websockets,tensorflow] && cd ..
```

## Contents

This script runs a FL experiment using MNIST. The folder is structured
the following way:

```
mnist/
│   generate_ssl.py - generate self-signed ssl certificates
|   prepare_data.py - fetch and split the MNIST dataset for FL use
|   run_client.py   - set up and launch a federated-learning client
│   run_demo.py     - simulate the entire FL process in a single session
│   run_server.py   - set up and launch a federated-learning server
└─── data           - data folder, containing raw and split MNIST data
└─── results_<time> - saved results from training procedures
```

## Run training routine

The simplest way to run the demo is to run it locally, using multiprocessing.
For something closer to real life implementation, we also show a way to run
the demo from different terminals or machines.

### Locally, for testing and experimentation

**To simply run the demo**, use the bash command below. You can follow along
the code in the `hands-on` section of the package documentation. For more
details on what running the federated learning processes imply, see the last
section.

```bash
cd declearn/examples/mnist/
python run_demo.py  # note: python declearn/examples/mnist/run.py works as well
```

The `run_demo.py` scripts collects the server and client routines defined under
the `run_server.py` and `run_client.py` scripts, and runs them concurrently
under a single python session using multiprocessing. It also prepares the data
by calling the `prepare_data.py` script, passing along input arguments, which
users are encouraged to play with - notably to vary the number of clients and
the data splitting scheme.

This is the easiest way to launch the demo, e.g. to see the effects of
tweaking some learning parameters (by editing the `run_server.py` script).

### On separate terminals or machines

**To run the examples from different terminals or machines**,
we first ensure data is appropriately distributed between machines,
and the machines can communicate over network using SSL-encrypted
communications. We give the code to simulate this on a single machine.
We then sequentially run the server then the clients on separate terminals.

1. **Prepare the data**:<br/>
   First, clients' data should be generated, by fetching and splitting the
   MNIST dataset. This can be done in any way you want, but a practical and
   easy one is to use the `prepare_data.py` script. Data may either be
   prepared at a single location and then shared across clients (in the case
   when distinct computers are used), or prepared redundantly at each place
   using the same random seed and agreeing on clients' ordering.

   To use the `prepare_data.py` script, simply run:
   ```bashand
   python prepare_data.py <NB_CLIENTS> [--scheme=SCHEME] [--seed=SEED]
   ```
   where `SCHEME` must be in `{"iid", "labels", "biased"}`
   and `SEED` may be any int.

   Alternatively, you may use the `declearn-split` command-line utility, with
   similar arguments:
   ```bash
   declearn-split --n_shards=<NB_CLIENTS> [--scheme=SCHEME] [--seed=SEED]
   ```

2. **Set up SSL certificates**:<br/>
   Create a signed SSL certificate for the server and share the CA file that
   signed it with each and every clients. That CA may be self-signed.

   When testing locally, execute the `generate_ssl.py` script, to create a
   self-signed root CA and an SSL certificate for "localhost":

   ```bash
   python generate_ssl.py
   ```

   Note that in real-life applications, one would most likely use certificates
   certificates signed by a trusted certificate authority instead.

   Alternatively, `declearn.test_utils.gen_ssl_certificates` may be used to
   generate a self-signed CA and a signed certificate for a given domain name
   or IP address.

3. **Run the server**:<br/>
   Open a terminal and launch the server script for the desired number of
   clients, specifying the path to the SSL certificate and private key files,
   and network parameters. By default, things will run on the local
   host, looking for `generate_ssl.py`-created PEM files.

   E.g., to use 2 clients:

    ```bash
    python run_server.py 2  # use --help for details on network and SSL options
    ```

    Note that you may edit that script to change the model learned, the FL
    and optimization algorithms used, and/or the training hyper-parameters,
    including the introduction of sample-level differential privacy.

3. **Run each client**:<br/>
   Open a new terminal and launch the client script, specifying the path to
   the main data folder (e.g. "data/mnist_iid") and the client's name (e.g.
   "client_0"), which are both used to determine where to get the prepared
   data. Additional network parameters may also be passed; by default, things
   will run on the localhost, looking for a `generate_ssl.py`-created CA PEM
   file.

   E.g., to launch the first client after preparing iid-split data with the
   `prepara_data.py` script, call:

    ```bash
    python run_client.py client_0 data/mnist_iid
     # use --help for details on options
    ```

Note that the server should be launched before the clients, otherwise the
latter might fail to connect which would cause the script to terminate. A
few seconds' delay is tolerable as clients will make multiple connection
attempts prior to failing.

**To run the example in a real-life setting**, follow the instructions from
this section, after having generated and shared the appropriate PEM files to
set up SSL-encryption, and using additional script parameters to specify the
network host and port to use.
