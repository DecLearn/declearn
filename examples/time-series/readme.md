# Demo training task: Time-series masked auto-encoder for hand pose surface elecrtomyography (sEMG)
The following example is built to illustrate the usage of [declearn](https://github.com/DecLearn/declearn) to perform
Federated learning for signal reconstruction.
In this example we use [surfce electromyography](https://en.wikipedia.org/wiki/Electromyography) time-series data on the dataset
*"[EMGs datasets:
Two datasets with EMGs signals of people making gestures](https://www.rovit.ua.es/dataset/emgs/)"*
proposed by (N.Nasri and al, 2019). 

The used model for this example, is a super basic Masked [Auto-Encoder](https://en.wikipedia.org/wiki/Autoencoder) following the butterfly architecture
in which the idea is to mask random points of every slice of the signal following a specified masking ratio, then attempt to reconstrcut the whole slice.

For now the loss is computed over the whole slice instead of just the masked points, this choice is to simplify the first implementation of this example.

## Technical setup 
Start first by cloning the declearn library
```bash
git clone git@gitlab.inria.fr:magnet/declearn/declearn.git declearn
```
Create a virual env and install the necessary packages listed in the `pyproject.toml` within `declearn/`

```bash
python3 -m venv <your_venv_name>
```
```bash
cd declearn && pip install . && cd ./examples/time-series/
```
If you use `uv` python package manager: 
```bash
uv venv <your_venv_name> && cd declearn 
```
```bash
uv pip install -r pyroject.toml
```
## Structure
The folder is structured the following way:
```
examples/time-series/
├── dataset.py       # Dataset definition
├── gen_ssl.py       # Self-signed SSL certificate generation
├── model.py         # Model definition
├── prepare_data.py  # Client data generation script
├── readme.md        # Documentation
├── run_client.py    # Launch a federated learning client
├── run_demo.py      # Run the full demo locally
└── run_server.py    # Launch the server
```
## Execution
The simplest way to run the demo is to run it locally, using multiprocessing.
```bash
cd declearn && python examples/time-series/prepare_data.py --nb_clients <NUMBER_OF_CLIENTS>
```
To know more about the flags for the example, you can use: 
```bash
cd examples/time-series && python3 run run_demo.py --help
```
Similarly with `uv`:
```bash
cd examples/time-series && uv run run_demo.py --help
```

### Multi-Terminal Execution

1. **Prepare the data**:<br/>
   First, clients' data should be generated, by fetching and splitting the
   *sEMG hand poses* dataset. This can be done in any way you want, but a practical and
   easy one is to use the `dataset/examples/_time-series.py` script. Data may either be
   prepared at a single location and then shared across clients (in the case
   when distinct computers are used), or prepared redundantly at each place
   using the same random seed and agreeing on clients' ordering.

   To use the `prepare_data.py` script, simply run:
   ```bashand
   python dataset/examples/_time-series.py --folder <data_folder>
   ```
    To see and play with data prep arguments further, run the same file with `--help` flag.


3. **Set up SSL certificates**:<br/>
   Create a signed SSL certificate for the server and share the CA file that
   signed it with each and every clients. That CA may be self-signed.

   When testing locally, execute the `generate_ssl.py` script, to create a
   self-signed root CA and an SSL certificate for "localhost":

```bash
python examples/time-series/data.py --nb_clients <NUMBER_OF_CLIENTS>
```
#### 2. Set Up SSL Certificates

   Alternatively, `declearn.test_utils.generate_ssl_certificates` may be used to
   generate a self-signed CA and a signed certificate for a given domain name
   or IP address.  
   To achieve this easily with the provided example script, update 
   `generate_ssl.py` so that it calls the `generate_ssl_certificates` function
   with custom arguments, more precisely :
   - If you use a domaine name as host (e.g. `mymachine.mydomain.fr`), set this
   value for the `c_name` argument (or in a list, for the `alt_dns` argument).
   - If you use an IP address as host (e.g. `192.0.2.1`), set this value in a
   list and pass it to the `alt_ips` argument.

      Example (IP address):  
      `generate_ssl_certificates(FILEDIR, alt_ips=["192.0.2.1"])`


For more advanced setups, you can use `declearn.utils.generate_ssl_certificates`.

   E.g., to use 2 clients:

    ```bash
    python run_server.py 2  # use --help for details on network and SSL options
    ```

    Note that you may edit that script to change the model learned, the FL
    and optimization algorithms used, and/or the training hyper-parameters,
    including the introduction of sample-level differential privacy.

3. **Run each client**:<br/>
   Open a new terminal and launch the client script, specifying the path to
   the main data folder (e.g. `"data/15Subjects-7Gestures"`) and the client's name (e.g.
   "client_0"), which are both used to determine where to get the prepared
   data. Additional network parameters may also be passed; by default, things
   will run on the localhost, looking for a `generate_ssl.py`-created CA PEM
   file.

   E.g., to launch the first client after preparing iid-split data with the
   `prepara_data.py` script, call:

```bash
python run_server.py 
```

Note that: 
- the server should be launched before the clients, otherwise the
latter might fail to connect which would cause the script to terminate. A
few seconds' delay is tolerable as clients will make multiple connection
attempts prior to failing.
- In this example we use the same data file for both clients (for now). But feel free
add sEMG data that follows the same structure normally it should be processed just as fine.



Launch each client in a separate terminal:

```bash
python run_client.py --name client_<CLIENT_NUMBER>
```

Use `--help` for additional options.

## Notes

- The server must be started before the clients; otherwise, connection attempts may fail.
- Clients will retry connections for a short period before exiting.
