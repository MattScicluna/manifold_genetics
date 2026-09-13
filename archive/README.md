# archive

Documentation that is **not published** on
<https://mattscicluna.github.io/manifold_genetics/>.

The site is deliberately four pages — home, install, quickstart, tutorial — and
the reference material is being rewritten and reintroduced one page at a time.
Everything here is either waiting for that, or is internal to how the project is
developed and was never meant for users.

Nothing here is built by MkDocs, so links between these files resolve on GitHub
but are not checked by the docs build. Treat the content as **last accurate on
2026-09-13** unless a file says otherwise.

## Waiting to be rewritten and reintroduced

| file | what it covers |
|---|---|
| `cli.md` | all 14 subcommands, what each reads and writes |
| `configuration.md` | every config key, and the memory budgets |
| `api.md` | the `__all__` surface, for library use |
| `concepts.md` | fit vs project, the PCA contract, landmarking |

`cli.md` is still checked by `tests/unit/test_docs_match_cli.py`, which fails if
a subcommand is undocumented or documented but absent — so it cannot drift out
of date while it waits.

## Internal: how the project is developed

| file | what it covers |
|---|---|
| `working-with-agents.md` | the two controlled-access incidents, and the guardrails added after them |
| `testing-real-cohorts.md` | the preflight and cohort suites, and the All of Us procedure |
| `releasing.md` | the tag-triggered release workflow and trusted publishing |
| `hpc.md` | SLURM, and the memory arithmetic |
| `controlled-access.md` | UK Biobank and All of Us specifics |
| `migrations/` | the `transform` → `project` rename |
| `superpowers/` | design specs and implementation plans |

`working-with-agents.md` is referenced by `.sensitive-allow`,
`.github/workflows/ci.yml` and `.pre-commit-config.yaml`. **Moving it again means
updating the allowlist**, or the leak check will flag it and fail CI.

## Superseded

| file | why |
|---|---|
| `UKBB_README.md` | an early walkthrough, superseded by `controlled-access.md` and `examples/ukbb/*/config.yaml` |
