# Adding a custom optimizer module: RMSProp example

## Introduction

Rather than implementing myriads of optimizers, `declearn` provides with a very
simple base `Optimizer` designed for stochastic gradient descent, that can be
made to use any arbitrary **pipeline of plug-in modules**, the latter of which
implement algorithms that change the way model updates are computed from input
gradients, e.g. to use acceleration or correction schemes. These **modules can
be stateful**, and **states can be shared between the server and its clients**
as part of a federated learning process. This tutorial leaves apart the latter
property, to focus on an example that may be run either on the client or server
side but does not require any synchronicity as part of federated learning.

## Overview

**To create a new optimizer module**, create a new class inheriting from the
`OptiModule` abstract class. This class was designed to encapsulate any
transformation taking in gradients in the form of a `Vector`, applying a set
of transformations, and outputting the transformed gradient as a `Vector`
that preserves input specifications.

**Two key conceptual elements** need to be included :

* The parameters of the module need to be defined and accessible, included
in the code in the `__init__` and `get_config` method.
* The transformations applied to the gradients, corresponding to the `run`
method.

**If you are contributing** to `declearn`, please write your code to an appropriate
file under `declearn.optimizer.modules`, include it to the `__all__` global
variable and import it as part of the `__init__.py` file at its import level.
Note that the basic tests for modules will automatically cover your module
thanks to its type-registration.

## RMSProp

**We here show how the RMSProp optimizer was added** to the codebase. This
optimizer introduces diagonal scaling to the gradient updates, allowing apt
re-scaling of the step-size for each dimension of the gradients.

The Root Mean Square Propagation (RMSProp) algorithm, introduced by
[Tieleman and Hinton, 2012](https://www.cs.toronto.edu/~tijmen/csc321/slides/\
lecture_slides_lec6.pdf), scales down the gradients by the square-root of the
momentum-corrected sum of the past squared gradients.

* $`\text{Init}(\beta, \epsilon):`$
  * $\hat{\nabla} = 0$
* $`\text{Step}(\nabla, step):`$
  * $`\hat{\nabla} = \beta*\hat{\nabla} + (1-\beta)*(\nabla^2)`$
  * $`\nabla /= (\sqrt{\hat{\nabla}} + \epsilon)`$

## RMSProp commented code

The RMSProp optimizer is part of the adaptative optimizer family, and was thus
added to `_adaptative.py` file.

```python
from declearn.optimizer.modules import OptiModule, EWMAModule


class RMSPropModule(OptiModule):
    """[Docstring removed for conciseness]"""

    # Identifier, that must be unique across modules for type-registration
    # purposes. This enables specifying the module in configuration files.

    name:ClassVar[str] = "rmsprop"

    # Define optimizer parameters, here beta and eps

    def __init__(self, beta: float = 0.9, eps: float = 1e-7) -> None:
        """Instantiate the RMSProp gradients-adaptation module.

        Parameters
        ----------
        beta: float
            Beta parameter for the momentum correction
            applied to the adaptive scaling term.
        eps: float, default=1e-7
            Numerical-stability improvement term, added
            to the (divisor) adapative scaling term.
        """

        # Reuse the existing EWMA module, see below

        self.mom = EWMAModule(beta=beta)
        self.eps = eps

    # Allow access to the module's parameters

    def get_config(self,) -> Dict[str, Any]:
        return {"beta": self.ewma.beta, "eps": self.eps}

    # Define the actual transformations of the gradient

    def run(self, gradients: Vector) -> Vector:
        v_t = self.ewma.run(gradients**2)
        scaling = (v_t**0.5) + self.eps
        return gradients / scaling

    # Define the state-access methods; here states are handled by the EWMA

    def get_state(self) -> Dict[str, Any]:
        return self.ewma.get_state()

    def set_state(self, state: Dict[str, Any],) -> None:
        self.ewma.set_state(state)
```

We here reuse the EWMA module, defined in the `modules/_momentum.py` file. As a
module, it takes in a `Vector` and outputs a `Vector`. It has one parameter,
$`\beta`$, manages a state vector `state` and its `run` method looks like this:

```python
    def run(self, gradients: Vector) -> Vector:
        self.state = (self.beta * self.state) + ((1 - self.beta) * gradients)
        return self.state
```
