# Demo Training Task: Time-Series Masked Autoencoder for Hand Pose sEMG

This example demonstrates how to use [DecLearn](https://github.com/DecLearn/declearn) to perform federated learning for signal reconstruction.

We use time-series data from [Surface Electromyography (sEMG)](https://en.wikipedia.org/wiki/Electromyography), specifically the dataset:  
*"[EMG datasets: Two datasets with EMG signals of people making gestures](https://www.rovit.ua.es/dataset/emgs/)"* by Nasri et al. (2019).

The model implemented in this example is a simple masked [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) following a butterfly architecture. The core idea is to randomly mask points within each signal slice according to a predefined masking ratio, and train the model to reconstruct the full slice.

For simplicity, the loss is currently computed over the entire slice rather than restricted to the masked points.

## Technical Setup

First, clone the DecLearn repository:

```bash
git clone https://gitlab.inria.fr/magnet/declearn/declearn.git
```

Create and activate a virtual environment and install the required dependencies listed in `pyproject.toml`:
Ex:  
```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
cd declearn && pip install ".[torch, websockets]" && cd ..
```

## Project Structure

The example directory is organized as follows:

```text
examples/time-series/
├── dataset.py       # Dataset definition
├── model.py         # Model definition
├── prepare_data.py  # Client data generation script
├── run_client.py    # Launch a federated learning client
├── run_demo.py      # Run the full demo locally
└── run_server.py    # Launch the server
```

## Execution

The demo can be run either from a single terminal (local simulation) or across multiple terminals or machines (closer to real-world deployment).

### Single-Terminal Execution

First, generate the dataset using:

```bash
cd declearn/examples/time-series
python prepare_data.py --nb_clients <NUMBER_OF_CLIENTS>
```

You can experiment with the number of clients, up to 8 for now.

Then, run the full demo locally using multiprocessing:

```bash
python run_demo.py
```

To view available options:

```bash
python run_demo.py --help
```

### Multi-Terminal Execution

In this setup, the server and clients are launched independently, potentially on different machines, using SSL-secured communication.

#### 1. Prepare the Data

Client datasets must be generated in advance by downloading and splitting the sEMG dataset. In this example, each client is assigned one sensor, which limits the number of clients to 8.

- `_time_series_emg.py` handles filtering, splitting, and normalization.
- `prepare_data.py` automates dataset generation.

To generate client data:

```bash
cd declearn/examples/time-series
python prepare_data.py --nb_clients <NUMBER_OF_CLIENTS>
```

#### 2. Set Up SSL Certificates

Generate SSL certificates for secure communication. For local testing:

```bash
python ../common/generate_ssl.py
```

Note that in real-life applications, one would most likely use certificates
certificates signed by a trusted certificate authority instead.

Also note that `generate_ssl.py` may be used to generate a self-signed CA and
a signed certificate for a given domain name or IP address.  
See the script documentation for more details.

#### 3. Run the server

Start the server first:

```bash
python run_server.py <NUMBER_OF_CLIENTS>
```

Use `--help` to configure networking and SSL options.

You may also modify this script to adjust:
- The model architecture
- The federated learning strategy
- The optimization algorithm
- The training hyper-parameters, including differential privacy


#### 4. Run the clients

Launch each client in a separate terminal:

```bash
python run_client.py client_<CLIENT_NUMBER>
```

Use `--help` for additional options.

## Notes

- The server must be started before the clients; otherwise, connection attempts may fail.
- Clients will retry connections for a short period before exiting.
