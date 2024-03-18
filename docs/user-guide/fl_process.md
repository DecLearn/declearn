# Overview of the Federated Learning process

This overview describes the way the `declearn.main.FederatedServer`
and `declearn.main.FederatedClient` pair of classes implement the
federated learning process. It is however possible to subclass
these and/or implement alternative orchestrating classes to define
alternative overall algorithmic processes - notably by overriding
or extending methods that define the sub-components of the process
exposed here.

## Overall process orchestrated by the server

- Initially:
    - the clients connect to the server and register for training
    - the server may collect targetted metadata from clients when required
    - the server sets up the model, optimizers, aggregator and metrics
    - all clients receive instructions to set up these objects as well
- Iteratively:
    - perform a training round
    - perform an evaluation round
    - decide whether to continue, based on the number of
      rounds taken or on the evolution of the global loss
- Finally:
    - restore the model weights that yielded the lowest global loss
    - notify clients that training is over, so they can disconnect
      and run their final routine (e.g. save the "best" model)
    - optionally checkpoint the "best" model
    - close the network server and end the process

## Detail of the process phases

### Registration process

- Server:
    - open up registration (stop rejecting all received messages)
    - handle and respond to client-emitted registration requests
    - await criteria to have been met (exact or min/max number of clients
      registered, optionally under a given timeout delay)
    - close registration (reject future requests)
- Client:
    - connect to the server and send a request to join training
    - await the server's response (retry after a timeout if the request
      came in too soon, i.e. registration is not opened yet)

### Post-registration initialization

#### (Optional) Metadata exchange

This step is optional, and depends on the trained model's requirement
for dataset information (typically, features shape and/or dtype).

- Server:
    - query clients for targetted metadata about the local training datasets
- Client:
    - collect and send back queried metadata
- messaging: (MetadataQuery <-> MetadataReply)
- Server:
    - validate and aggregate received information
    - pass it to the model so as to finalize its initialization

#### Initialization of the federated optimization problem

- Server:
    - set up the model, local and global optimizer, aggregator and metrics
    - send specs to the clients so that they set up local counterpart objects
- Client:
    - instantiate the model, optimizer, aggregator and metrics based on specs
- messaging: (InitRequest <-> InitReply)

#### (Optional) Local differential privacy setup

This step is optional; a flag in the InitRequest at the previous step
indicates to clients that it is to happen, as a secondary substep.

- Server:
    - send hyper-parameters to set up local differential privacy, including
      dp-specific hyper-parameters and information on the planned training
- Client:
    - adjust the training process to use sample-wise gradient clipping and
      add gaussian noise to gradients, implementing the DP-SGD algorithm
    - set up a privacy accountant to monitor the use of the privacy budget
- messaging: (PrivacyRequest <-> GenericMessage)

### Training round

- Server:
    - select clients that are to participate
    - send data-batching and effort constraints parameters
    - send shared model trainable weights and (opt. client-specific) optimizer
      auxiliary variables
- Client:
    - update model weights and optimizer auxiliary variables
    - perform training steps based on effort constraints
    - step: compute gradients over a batch; compute updates; apply them
    - finally, send back the local model weights' updates and optimizer
      auxiliary variables
- messaging: (TrainRequest <-> TrainReply)
- Server:
    - unpack and aggregate clients' model weights updates into global updates
    - unpack and process clients' optimizer auxiliary variables
    - run global updates through the server's optimizer to modify and finally
      apply them

### Evaluation round

- Server:
    - select clients that are to participate
    - send data-batching parameters and effort constraints
    - send shared model trainable weights
- Client:
    - update model weights
    - perform evaluation steps based on effort constraints
    - step: update evaluation metrics, including the model's loss, over a batch
    - optionally checkpoint the model, local optimizer and evaluation metrics
    - send results to the server: optionally prevent sharing detailed metrics;
      always include the scalar validation loss value
- messaging: (EvaluateRequest <-> EvaluateReply)
- Server:
    - aggregate local loss values into a global loss metric
    - aggregate all other evaluation metrics and log their values
    - optionally checkpoint the model, optimizer, aggregated evaluation
      metrics and client-wise ones
