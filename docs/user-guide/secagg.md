# Secure Aggregation

## Overview

### What is Secure Aggregation?

Secure Aggregation (often and hereafter abbreviated as SecAgg) is a generic
term to describe methods that enable aggregating client-emitted information
without revealing said information to the server in charge of this aggregation.
In other words, SecAgg is about computing a public aggregate of private values
in a secure way, limiting the amount of trust put into the server.

Various methods have been proposed in the litterature, that may use homomorphic
encryption, one-time pads, pseudo-random masks, multi-party computation methods
and so on. These methods come with various costs (both in terms of computation
overhead and communication costs, usually increasing messages' size and/or
frequency), and various features (_e.g._ some support a given amount of loss
of information due to clients dropping from the process; some are best-suited
for some security settings than others; some require more involved setup...).

### General capabilities

DecLearn implements both a generic API for SecAgg and some practical solutions
that are ready-for-use. In the current state of things however, some important
hypotheses are enforced:

- The current SecAgg implementations require clients to generate Ed25519
  identity keys and share the associate public keys with other clients prior
  to using DecLearn. The API also leans towards this requirement. In practice,
  it would not be difficult to make it so that the server distributes clients'
  public keys across the network, but we believe that the incurred loss in
  security partially defeats the purpose of SecAgg, hence we prefer to provide
  with the current behavior (that requires sharing keys via distinct channels)
  and leave it up to end-users to set up alternatives if they want to.
- There is no resilience to clients dropping from the process. If a client was
  supposed to participate in a round but does not send any information, then
  some new setup and secure aggregation would need to be performed. This may
  change in the future, but is on par with the current limitations of the
  framework.
- There are no countermeasures to dishonest clients, i.e. there is no effort
  put in verifying that outputs are coherent. This is not specific to SecAgg
  either, and is again something that will hopefully be tackled in future
  versions.

### Details and caveats

At the moment, the SecAgg API and its integration as part of the main federated
learning orchestration classes has the following characteristics:

- SecAgg is jointly parametrized by the server and clients, that must use
  coherent parameters.
    - The choice of using SecAgg and of a given SecAgg method must be the same
      across all clients and the server, otherwise an error is early-raised.
    - Some things are up to clients, notably the specification of identity keys
      used to set up short-lived secrets, and more generally any parameter that
      cannot be verified to be trustworthy if the server sets it up.
    - Some things are up to the server, notably the choice of quantization
      hyper-parameters, that is coupled with details on the expected magnitude
      of shared quantities (gradients, metrics...).
- SecAgg is set up every time participating clients to a training or validation
  round change.
    - Whenever participating clients do not match those in the previous round,
      a SecAgg setup round is triggered to as to set up controllers anew across
      participating clients.
    - This is designed to guarantee that controllers are properly aligned and
      there is no aggregation of mismatching encrypted values.
    - This may however not be as optimal as theoretically-achievable in terms
      of setup communication and computation costs.
- SecAgg is used to protect training and evaluation rounds' results.
    - Model updates, optimizer auxiliary variables and metrics are covered.
    - Some metadata are secure-aggregated (number of steps), some are discarded
      (number of epochs), some remain cleartext but are reduced (time spent for
      computations is sent in cleartext and max-aggregated).
- SecAgg does not (yet) cover other computations.
    - Metadata queries are not secured (meaning the number of data samples may
      be sent in cleartext).
    - The topic of protecting specific or arbitrary quantities as part of
      processes implemented by subclassing the current main classes remains
      open.
- SecAgg of `Aggregate`-inheriting objects (notably `ModelUpdates`, `AuxVar`
  and `MetricState` instances) is based on their `prepare_for_secagg` method,
  that may not always be defined. It is up to end-users to ensure that the
  components they use (and custom components they might add) are properly
  made compatible with SecAgg.

## How to setup and use SecAgg

From the end-user perspective, setting up and using SecAgg requires:

- Having generated and shared clients' (long-lived) identity keys across
  trusted peers prior to running DecLearn.
- Having all peers specify coherent SecAgg parameters, that are passed to
  the main `FederatedClient` and `FederatedServer` classes at instantiation.
- Ensuring that the `Aggregator`, `OptiModule` plug-ins and `Metrics` used
  for the experiment are compatible with SecAgg.
- _Voilà!_

### Available SecAgg algorithms

At the moment, DecLearn provides with the following SecAgg algorithms:

- Masking-based SecAgg (`declearn.secagg.masking`), that uses pseudo-random
  number generators (PRNG) to generate masks over a finite integer field so
  that the sum of clients' masks is known to be zero.
    - This is based on
      [Bonawitz et al., 2016](https://dl.acm.org/doi/10.1145/3133956.3133982).
    - The setup that produces pairwise PRNG seeds is conducted using the
      [X3DH](https://www.signal.org/docs/specifications/x3dh/) protocol.
    - This solution has very limited computation and commmunication overhead
      and should be considered the default SecAgg solution with DecLearn.

- Joye-Libert sum-homomorphic encryption (`declearn.secagg.joye-libert`), that
  uses actual encryption, modified summation operator, and aggregate-decryption
  primitives that operate on a large biprime-defined integer field.
    - This is based on
      [Joye & Libert, 2013](https://marcjoye.github.io/papers/JL13aggreg.pdf).
    - The setup that compute the public key as a sum of arbitrary private keys
      involves the [X3DH](https://www.signal.org/docs/specifications/x3dh/)
      protocol as well as
      [Shamir Secret Sharing](https://dl.acm.org/doi/10.1145/359168.359176).
    - This solution has a high computation and commmunication overhead. It is
      not really suitable for model with many parameters (including few-layers
      artificial neural networks).

### Hands-on example

If we use the [MNIST example](https://gitlab.inria.fr/magnet/declearn/declearn2/-/tree/develop/examples/mnist/)
implemented via Python scripts and want to use the DecLearn-provided
masking-based algorithm for SecAgg (see below), we merely have to apply the
following modifications:

**1. Generate client Ed25519 identity keys**

This may be done using dedicated tools, but here is a script to merely
generate and dump identity keys to files on a local computer:

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from declearn.secagg.utils import IdentityKeys

# Generate some private Ed25519 keys and gather their public counterparts.
n_clients = 5  # adjust to the number of clients
private_keys = [Ed25519PrivateKey.generate() for _ in range(n_clients)]
public_keys = [key.public_key() for key in private_keys]

# Export all public keys as a single file with custom format.
IdentityKeys(private_keys[0], trusted=public_keys).export_trusted_keys_to_file(
    "trusted_public.keys"
)

# Export private keys as PEM files without password protection.
for idx, key in enumerate(private_keys):
    dat = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with open(f"private_{idx}.pem", "wb") as file:
        file.write(dat)
```

**2. Write client-side SecAgg config**

Add the following code as part of the client routine to setup and run the
federated learning process. Paths and parameters may and should be adjusted
for practical use cases.

```python
from declearn.secagg import parse_secagg_config_client
from declearn.secagg.utils import IdentityKeys

# Add this as part of the `run_client` function in the `run_client.py` file.

client_idx = int(client_name.rsplit("_", 1)[-1])
secagg = parse_secagg_config_client(
    secagg_type="masking",
    id_keys=IdentityKeys(
        prv_key=f"client_{client_idx}.pem",
        trusted="trusted_public.keys",
    )
)
# Alternatively, write `secagg` as the dict of previous kwargs,
# or as a `declearn.secagg.masking.MaskingSecaggConfigClient` instance.

# Overload the `FederatedClient` instantation in step (5) of the function.

client = declearn.main.FederatedClient(
    # ... what is already there
    secagg=secagg,
)
```

**3. Write server-side SecAgg config**

Add the following code as part of the server routine to setup and run the
federated learning process. Parameters may and should be adjusted to practical
use cases.

```python
from declearn.secagg import parse_secagg_config_server

# Add this as part of the `run_server` function in the `run_server.py` file.

secagg = parse_secagg_config_server(
    secagg_type="masking",
    # You may tune hyper-parameters, controlling values' quantization.
)
# Alternatively, write `secagg` as the dict of previous kwargs,
# or as a `declearn.secagg.masking.MaskingSecaggConfigServer` instance.

# Overload the `FederatedServer` instantation in step (4) of the function.

server = declearn.main.FederatedServer(
    # ... what is already there
    secagg=secagg,
)
```

Note that if the configurations do not match (whether as to the use of SecAgg
or not, the choice of algorithm, its hyper-parameters, or the validity of
trusted identity keys), an error will be raised at some (early) point when
attempting to run the Federated Learning process.

## How to implement a new SecAgg method

The API for SecAgg relies on a number of abstractions, that may be divided in
two categories:

- backend controllers that provide with primitives to encrypt, aggregate
  and decrypt values;
- user-end controllers that parse configuration parameters and implement
  a setup protocol to instantiate backend controllers.

As such, it is possible to write a new setup for existing controllers; but in
general, adding a new SecAgg algorithm to DecLearn will involve writing it all
up. The first category is somewhat decoupled from the rest of the framework (it
only needs to know what the `Vector` and `Aggregate` data structures are, and
how to operate on them), while the second is very much coupled with the network
communication and messaging APIs.

### `Encrypter` and `Decrypter` controllers

Primitives for the encryption of private values, aggregation of encrypted
values and decryption of an aggregated value are to be implemented by a pair
of `Encrypter` and `Decrypter` subclasses.

These classes may (and often will) use any form of (private) time index, that
is not required to be shared with encrypted values, as values are assumed to be
encrypted in the same order by each and every client and decrypted in that same
order by the server once aggregated.

`Encrypter` has two abstract methods that need implementing:

- `encrypt_uint`, that encrypts a scalar uint value (that may arise from the
  quantization of a float value). It is called when encrypting floats, numpy
  arrays and declearn `Vector` or `Aggregate` instances.
- `wrap_into_secure_aggregate`, that is called to wrap up encrypted values
  and other metadata and information from an `Aggregate` instance into a
  `SecureAggregate` one. This should in most cases merely correspond to
  instantiating the proper `SecureAggregate` subclass with a mix of input
  arguments to the method and attributes from the `Encrypter` instance.

`Decrypter` has an abstract class attribute and two abstract methods that
need implementing:

- `sum_encrypted`, that aggregates a list of two or more encrypted values that
  need aggregation into a single one (in a way that makes their aggregate
  decryptable into the sum of cleartext values).
- `decrypt_uint`, that decrypts an input into a scalar uint value. It is called
  when decrypting inputs into any supported type, and is the counterpart to the
  `Encrypter.encrypt_uint` method.
- `secure_aggregate_cls`, that is a class attribute that is merely the type of
  the `SecureAggregate` subclass emitted by the paired `Encrypter`'s
  `wrap_into_secure_aggregate` method.

`SecureAggregate` is a third class that is used by the former two, and usually
not directly accessed by end-users. It is an `Aggregate`-like wrapper for
encrypted counterparts to `Aggregate` objects. The only abstract method that
needs defining is `aggregate_encrypted`, which should mostly be the same as
`Decrypter.sum_encrypted`. Subclasses may also embark additional metadata
about the SecAgg algorithm's parameters, and conduct associate verification
of coherence across peers.

### `SecaggConfigClient` and `SecaggConfigServer` endpoints

Routines to set up matching `Encrypter` and `Decrypter` instances across a
federated network of peers are to be implemented by a pair of
`SecaggConfigClient` and `SecaggConfigServer` subclasses. In addition, a
dedicated `SecaggSetupQuery` subclass (itself a `Message`) should be defined.

These classes define a setup that is bound to be initiated by the server, that
emits a `SecaggSetupQuery` message triggering the client's call to the setup
routine. After that, any number of network communication exchanges may be run,
depending on the setup being implemented.

Both `SecaggConfigClient` and `SecaggConfigServer` subclasses must be decorated
as `dataclasses.dataclass`, so as to benefit from TOML-parsing capabilities.
They are automatically type-registered (which may be prevented by passing the
`register=False` parameter at inheritance).

`SecaggConfigClient` has an abstract class attribute and an abstract method:

- `secagg_type`, that is a string class attribute that must be unique across
  subclasses (for type-registration) and match that of the server-side class.
- `setup_encrypter`, that takes a `NetworkClient` and a received serialized
  `SecaggSetupQuery` message, and conducts any steps towards setting up and
  returning an `Encrypter` instance.

`SecaggConfigServer` has an abstract class attribute and two abstract methods:

- `secagg_type`, that is a string class attribute that must be unique across
  subclasses (for type-registration) and match that of the client-side class.
- `prepare_secagg_setup_query`, that returns a `SecaggSetupQuery` to be sent
  to (the subset of) clients that are meant to participate in the setup.
- `finalize_secagg_setup`, that takes a `NetworkServer` and an optional set
  of client names, is called right after sending the setup query to (these)
  clients and may thereafter conduct any steps towards setting up and returning
  a `Decrypter` instance.
