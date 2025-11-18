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
│   data.py    - download and prepare the dataset
│   gen_ssl.py - generate self-signed ssl certificates
│   run.py     - launch both the server and clients in a single session
│   server.py  - set up and launch a federated-learning server
│   setup.sh   - bash script to prepare client-wise and server isolated folders
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
we first ensure data is appropriately distributed between machines,
and the machines can communicate over network using SSL-encrypted
communications. We give the code to simulate this on a single machine.
We then sequentially run the server then the clients on separate terminals.

1. **Set up SSL certificates**:<br/>
   Start by creating a signed SSL certificate for the server and sharing the
   CA file with each and every clients. The CA may be self-signed.

   When testing locally, execute the `gen_ssl.py` script, to create a
   self-signed root CA and an SSL certificate for "localhost":
   ```bash
   python gen_ssl.py
   ```

   Note that in real-life applications, one would most likely use certificates
   signed by a trusted certificate authority instead.
   Alternatively, `declearn.test_utils.gen_ssl_certificates` may be used to
   generate a self-signed CA and a signed certificate for a given domain name
   or IP address.

2. **Run the server**:<br/>
   Open a terminal and launch the server script for 1 to 4 clients,
   specifying the path to the SSL certificate and private key files,
   and network parameters. By default, things will run on the local
   host, looking for `gen_ssl.py`-created PEM files.

   E.g., to use 2 clients:
    ```bash
    python server.py 2  # use --help for details on network and SSL options
    ```

3. **Run each client**:<br/>
   Open a new terminal and launch the client script, specifying one of the
   dataset-provider names, and optionally the path the CA file and network
   parameters. By default, things will run on the local host, looking for
   a `gen_ssl.py`-created CA PEM file.

   E.g., to launch a client using the "cleveland" dataset:
    ```bash
    python client.py cleveland   # use --help for details on other options
    ```

Note that the server should be launched before the clients, otherwise the
latter might fail to connect which would cause the script to terminate. A
few seconds' delay is tolerable as clients will make multiple connection
attempts prior to failing.

**To run the example in a real-life setting**, follow the instructions from
this section, after having generated and shared the appropriate PEM files to
set up SSL-encryption, and using additional script parameters to specify the
network host and port to use.
