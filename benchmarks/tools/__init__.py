"""Investigation tooling that lives alongside the ASV benchmark suite.

This subpackage holds scripts that exercise the same workload as the
ASV cells (via `benchmarks.workload.build_benchmark` /
`benchmarks.workload.run_benchmark`) but step outside ASV's sweep model
to support targeted investigation — e.g. py-spy flame graphs for a
specific configuration when ASV flags a regression.
"""
