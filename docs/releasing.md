# Releasing

A release is the one action in this repository that cannot be undone. PyPI will
not accept a second upload of the same version, and will not let you replace one
— a mistake is fixed only by publishing another version on top of it. Everything
below is arranged around that.

Publishing happens in one place: `.github/workflows/release.yml`, triggered by a
tag push. There is no API token in this repository and no way to publish from a
laptop.

## One-time setup

Trusted publishing (OIDC) must be configured once per index, by someone with
owner rights on the PyPI project. There is no secret to store.

At <https://pypi.org/manage/account/publishing/> (and the same page on
<https://test.pypi.org>), add a pending publisher:

| field | value |
|---|---|
| PyPI project name | `manifold-genetics` |
| Owner | `MattScicluna` |
| Repository name | `manifold_genetics` |
| Workflow name | `release.yml` |
| Environment name | `pypi` (on TestPyPI: `testpypi`) |

Then create the matching GitHub environments under **Settings → Environments**:
`pypi` and `testpypi`. Adding a required reviewer to `pypi` is worth doing — it
makes the final upload a deliberate click rather than a consequence of a tag.

## Cutting a release

1. **Bump the version** in `pyproject.toml`. `src/manifold_genetics/__init__.py`
   must match; `tests/unit/test_package_metadata.py` fails if it does not.
2. **Write the changelog entry** under a `## [x.y.z]` heading in `CHANGELOG.md`.
   The same test fails if the version has no section, and so does the workflow.
3. **Run `uv lock`** — it records the version, so CI's drift check fails
   otherwise. Do this where there is internet; compute nodes have none.
4. **Open a PR and merge it.** The version bump is a reviewable change like any
   other.
5. **Rehearse on TestPyPI**, from the Actions tab: run *Release* manually with
   `target=testpypi`. TestPyPI accepts a version only once, exactly like PyPI, so
   rehearse with the version you intend to ship — not a throwaway.
6. **Verify the rehearsal** from a machine that is not a checkout of this
   repository:

   ```bash
   python -m venv /tmp/rehearsal && /tmp/rehearsal/bin/pip install \
       --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ \
       manifold-genetics
   /tmp/rehearsal/bin/manifold-genetics --help
   ```

   The extra index is needed because TestPyPI does not mirror dependencies.
7. **Tag and push:**

   ```bash
   git tag -a v0.2.0 -m "manifold-genetics 0.2.0"
   git push origin v0.2.0
   ```

   The workflow refuses to continue if the tag does not match the declared
   version, so a mistyped tag costs nothing.

## What the workflow checks before publishing

- The tag matches `pyproject.toml`'s version, and the changelog has a section
  for it.
- `tests/unit/test_package_metadata.py` — the declared licence matches the
  LICENSE file, the version is declared once, `py.typed` ships, and every
  `project.urls` entry points at the real repository. Each of those was wrong
  before 0.2.0.
- `tests/integration/test_sdist_contents.py` — the sdist contains no untracked
  file. Left unchecked, hatchling's default would have published
  `.claude/settings.local.json` and a script headed "NOT for external users".
- `twine check --strict`.
- A **clean-room install** on Linux and macOS, Python 3.10 and 3.12: a fresh
  virtualenv, the wheel and nothing else, then the console script and the public
  API. This is the check that `pip install manifold-genetics` actually works —
  it did not before 0.2.0, because PCA required a Linux-only binary.

Artefacts are built from a fresh checkout rather than a working tree, which is
the structural reason a stray local file cannot reach an index. The test above
is the belt to that braces.

## Yanking

If a release is broken, `yank` it rather than trying to delete it: the version
stays resolvable for anyone who pinned it, but stops being chosen by new
installs. Then fix forward with a patch release.
