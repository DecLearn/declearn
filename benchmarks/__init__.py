"""ASV benchmark suite for declearn.

This package holds the ASV benchmarks that track the runtime and memory
cost of common declearn federated-learning scenarios across versions.

The package is organized into the following modules:

* [suite][benchmarks.suite]:
    The ASV benchmark classes themselves, which ASV discovers and runs.
* [workload][benchmarks.workload]:
    Construction and execution of the FL workload each benchmark drives.

The classes are deliberately *not* re-exported here: ASV discovers
benchmarks by walking submodules, so re-importing them into this file
would have ASV count each one twice (once per module it appears in).
"""
