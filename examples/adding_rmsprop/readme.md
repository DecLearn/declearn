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

**Also make sure** to register your new `OptiModule` subtype, as demonstrated
below. This is what makes your module (de)serializable using `declearn`'s
internal tools.

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
from declearn.optimizer.modules import OptiModule
from declearn.utils import register_type


# Start by registering the new optimzer using the dedicated decorator

@register_type(name="RMSProp", group="OptiModule")
class RMSPropModule(OptiModule):
    """[Docstring removed for conciseness]"""

    # Convention, used when a module uses synchronized server
    # and client elements, that need to share the same name

    name = "rmsprop"

    # Define optimizer parameter, here beta and eps

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

        # Reuse the existing momemtum module, see below

        self.mom = MomentumModule(beta=beta)
        self.eps = eps

    # Allow access to the module's parameters

    def get_config(self,) -> Dict[str, Any]:
        """Return a JSON-serializable dict with this module's parameters."""
        return {"beta": self.mom.beta, "eps": self.eps}

    # Define the actual transformations of the gradient

    def run(self, gradients: Vector) -> Vector:
        """Apply RMSProp adaptation to input (pseudo-)gradients."""
        v_t = self.mom.run(gradients**2)
        scaling = (v_t**0.5) + self.eps
        return gradients / scaling
```

We here reuse the Momemtum module, defined in `modules/_base.py`. As a
module, it takes in a `Vector` and outputs a `Vector`. It has one parameter,
$`\beta`$, and its `run` method looks like this:

```python
    def run(self, gradients: Vector) -> Vector:
        """Apply Momentum acceleration to input (pseudo-)gradients."""

        # Iteratively update the state class attribute with input gradients

        self.state = (self.beta * self.state) + ((1 - self.beta) * gradients)
        return self.state

```
