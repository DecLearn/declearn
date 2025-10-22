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
    - additional setup phases optionally occur to set up advanced features
      (secure aggregation, differential privacy and/or group fairness)
- Iteratively:
    - (optionally) perform a fairness-related round
    - perform a training round
    - (optionally) perform an evaluation round
    - decide whether to continue, based on the number of
      rounds taken or on the evolution of the global loss
- Finally:
    - (optionally) evaluate the last model, if it was not already done
    - restore the model weights that yielded the lowest global validation loss
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
    - verify that (optional) secure aggregation algorithm choice is coherent
      with that of the server
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
- messaging: (PrivacyRequest <-> PrivacyReply)

#### (Optional) Fairness-aware federated learning setup

This step is optional; a flag in the InitRequest at a previous step
indicates to clients that it is to happen, as a secondary substep.

See our [guide on Fairness](./fairness.md) for further details on
what (group) fairness is and how it is implemented in DecLearn.

When Secure Aggregation is to be used, it is also set up as a first step
to this routine, ensuring exchanged values are protected when possible.

- Server:
    - send hyper-parameters to set up a controller for fairness-aware
      federated learning
- Client:
    - set up a controller based on the server-emitted query
    - send back sensitive group definitions
- messaging: (FairnessSetupQuery <-> FairnessGroups)
- Server:
    - define a sorted list of sensitive group definitions across clients
      and share it with clients
    - await associated sample counts from clients and (secure-)aggregate them
- Client:
    - await group definitions and send back group-wise sample counts
- messaging: (FairnessGroups <-> FairnessCounts)
- Server & Client: run algorithm-specific additional setup steps, that
  may have side effects on the training data, model, optimizer and/or
  aggregator; further communication may occur.

### (Optional) Secure Aggregation setup

When configured to be used, Secure Aggregation may be set up any number of
times during the process, as fresh controllers will be required each and
every time the participating clients to a round differs from those chosen
at the previous round.

By default however, all clients participate to each and every round, so
that a single setup will occur early in the overall FL process.

See our [guide on Secure Aggregation](./secagg.md) for further details on
what secure aggregation is and how it is implemented in DecLearn.

- Server:
    - send an algorithm-specific SecaggSetupQuery message to selected clients
    - trigger an algorithm-dependent setup routine
- Client:
    - parse the query and execute the associated setup routine
- Server & Client: perform algorithm-dependent computations and communication;
  eventually, instantiate and assign respective encryption and decryption
  controllers.
- messaging: (SecaggSetupQuery <-> (algorithm-dependent Message))

### (Optional) Fairness round

This round only occurs when a fairness controller was set up, and may be
configured to be periodically skipped.
If fairness is set up, the first fairness round will always occur.
If checkpointing is set up on the server side, the last model will undergo
a fairness round, to evaluate its fairness prior to ending the FL process.

- Server:
    - send a query to clients, including computational effort constraints,
      and current shared model weights (when not already held by clients)
- Client:
    - compute metrics that account for the fairness of the current model
- messaging: (FairnessQuery <-> FairnessReply)
- Server & Client: take any algorithm-specific additional actions to alter
  training based on the exchanged values; further, communication may happen.

### Training round

- Server:
    - select clients that are to participate
    - send data-batching and effort constraints parameters
    - send current shared model trainable weights (to clients that do not
      already hold them) and optimizer auxiliary variables (if any)
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

### (Optional) Evaluation round

This round may be configured to be periodically skipped.
If checkpointing is set up on the server side, the last model will always be
evaluated prior to ending the FL process.

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
