# Demo training task : heart disease prediction

## Overview

**We use data from the UCI ML repository** - Heart disease dataset, available
[here](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). The goal is to
predict a binary variable, indicating heart disease, from a set of health
indicators.

**To simply run the demo**, use the bash command below. You can follow along
the code in the `hands-on` section of the package documentation. For more
details on what running the federated learning processes imply, see the last
section.

```bash
python run.py
```

## Folder structure

```
heart-uci/
│   client.py  - set up and launch a federated-learning client
│   data.py    - download and preapte the dataset
│   gen_ssl.py - generate self-signed ssl certificates
│   run.py     - launch both the server and clients in a single session
│   server.py  - set up and launch a federated-learning server
|   setup.sh   - bash script to prepare client-wise and server isolated folders
└─── data      - saved datasets as csv files
└─── results   - saved results from training procedure
```

## Run training routine

The simplest way to run the demo is to run it locally, using multiprocessing.
For something closer to real life implementation, we also show a way to run
the demo from different terminals or machines.

### Locally, for testing and experimentation

Use :

```bash
python run.py  # note: python examples/heart-uci/run.py works as well
```

The `run.py` scripts collects the server and client routines defined under
the `server.py` and `client.py` scripts, and runs them concurrently under
a single python session using multiprocessing.

This is the easiest way to launch the demo, e.g. to see the effects of
tweaking some learning parameters.

### On separate terminals or machines

**To run the examples from different terminals or machines**,
We first ensure data is appropriately distributed between machines,
and the machines can communicate over network using SSL-encrypted
communications. We give the code to simulate this on a single machine.

We then sequentially run the server then the clients on separate terminals.

1. **Set up self-signed SSL certificates**:<br/>
   Start by running executing the `gen_ssl.py` script.
   This creates self-signed SSL certificates:
   ```bash
   python gen_ssl.py
   ```
   Note that in real-life applications, one would most likely use certificates
   signed by a trusted certificate authority instead.

2. **Run the server**:<br/>
   Open a terminal and launch the server script for 1 to 4 clients,
   using the generated SSL certificates:
    ```bash
    python server.py 2  # use --help for details on SSL files options
    ```

3. **Run each client**:<br/>
   Open a new terminal and launch the client script, using one of the
   dataset's location name and the generated SSL certificate, e.g.:
    ```bash
    python client.py cleveland   # use --help for details on SSL files options
    ```

Note that the server should be launched before the clients, otherwise the
latter would fail to connect which might cause the script to terminate.
