"""Backend-specific model factories used by the declearn benchmark suite.

Each module exposes a `build_model() -> declearn.model.api.Model` function.
Backends are imported lazily by `workload.build` so that missing optional
dependencies only surface when the corresponding backend is selected.
"""

__all__: list[str] = []
