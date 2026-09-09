"""In-process pipeline steps.

Each compute step lives in its own module and exposes a pure
``<step>_output_paths()`` helper plus (added in later PRs) a ``run_<step>_step()``
function. Both the ``manifold-genetics`` CLI subcommands and the pipeline
orchestrator call these directly, in-process.
"""
