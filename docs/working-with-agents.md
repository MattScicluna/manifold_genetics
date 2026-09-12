# Working safely with agents and collaborators

This repository is public. It is developed on an HPC cluster that also holds UK
Biobank and All of Us genotypes, under data-use agreements that do not permit
redistribution. Those two facts are in tension every time anything is committed,
and the tension is worse, not better, when work is being done quickly by an
agent or by a collaborator who does not know which files are which.

This page is about that risk, what went wrong here, and what now stops it.

## What happened

Twice, files from the working directory reached somewhere they should not have.
Neither was caught by review.

**A bulk add swept private files into a pull request.** A `git add -A examples/`
picked up seven untracked working files along with the intended change. One was
`prepare_data_create_labels.sh`, whose own header reads:

```bash
# Private script to create UKBB sample lists and labels using internal metadata
# This is NOT for external users - they should create sample lists manually
```

It was caught by a CI lint failure on two files nobody had edited — an accident.
Nothing in the process was looking for it.

**The source distribution contained untracked working files.** `uv build` builds
from the working tree, and hatchling's default is to include everything that is
not gitignored — which is *not* the same as everything tracked. The 0.2.0 sdist
contained thirteen untracked files, including `.claude/settings.local.json` with
absolute cluster paths and session identifiers in it, and the same private
script. It was caught while auditing the artefact before publishing, one step
before an upload that could not have been undone.

## What that says

Three things, none of which is "be more careful".

**A working directory on a machine with controlled-access data is not a safe
thing to bulk-add from.** `git add -A`, `git add <dir>`, and any build that
reads the working tree will pick up whatever is sitting there. On a cluster,
what is sitting there is intermediate files from real cohorts.

**`.gitignore` is not a safety boundary.** It stops files being *tracked*. It
does not stop them being added deliberately, and a build tool that respects it
still includes untracked-but-not-ignored files. Both incidents lived in exactly
that gap.

**Review does not find this.** A reviewer looks at the diff they were asked
about. Seven extra files in a large change, or a file list inside a tarball, is
not where attention goes. The check has to be mechanical and it has to run
before the mistake is durable.

## The guardrails

### Before the commit exists

`.pre-commit-config.yaml`, installed once per clone:

```bash
uv sync --frozen --extra dev
uv run pre-commit install
```

!!! note "On a cluster login node"

    `gitleaks` is a Go binary that asks the runtime for a thread per core. On a
    64-core login node with a 4096-process `ulimit` that dies with
    `fatal error: newosproc`, so constrain it:

    ```bash
    GOMAXPROCS=4 uv run pre-commit run --all-files
    ```

    Laptops and CI runners have few enough cores not to hit this.

The formatters run as `language: system` hooks — whatever the synced environment
provides — rather than versions pinned a second time in the hook config. A
second set of pins drifts from `uv.lock`, and then a commit that passes locally
fails the lint job for no visible reason. (Pinning the newest `black` also
pulled in a Rust build that will not compile against this cluster's Cargo: a
formatter hook that cannot be installed on the machine people commit from is
worse than none.)

The hook that matters is `scripts/check_sensitive_files.py`, written against
this repository's own failure modes rather than a generic secret scanner. It
flags:

| rule | what it catches |
|---|---|
| `private-marker` | a file whose header says it is private, internal, or not for external users |
| `genotype-data` | `.bed` `.bim` `.fam` `.ped` `.map` `.bgen` `.pgen` `.psam` `.pvar` `.vcf` `.gz` |
| `sample-identifiers` | a long first column of 6–9 digit biobank IDs — the shape of a label file |
| `local-state` | `.claude/`, `.superpowers/`, `.vscode/`, `.idea/` |
| `private-name` | a filename containing private, secret, credential, token, password |
| `cluster-path` | absolute paths under `/lustre*`, `/scratch`, `/project/ctb-…` |

Alongside it: `gitleaks` for credentials, `detect-private-key`, and a 2 MB
added-file size limit — genotype files are large before they are anything else,
so size catches them even when a suffix is unfamiliar.

The identifier rule is the one worth understanding. It does **not** fire on
public cohort IDs (`HG00096`, `NA12878`), which appear throughout the examples
and are meant to. It fires on fifty or more rows beginning with a bare numeric
ID, which is what a UK Biobank or All of Us label file looks like and what a
public HGDP one does not.

### After the push

Two CI jobs, on every pull request:

- **`leak-check`** runs the same script over *every tracked file*, so something
  committed without the hook installed still fails before merge.
- **`package`** builds the distributions and asserts that neither contains a
  file git does not track — which is what the sdist incident actually was. See
  `tests/integration/test_sdist_contents.py`.

The release workflow runs both again before it uploads anything, and builds from
a fresh checkout rather than a working tree, so the 0.2.0 failure mode is
structurally impossible there.

### Overriding it

Add an entry to `.sensitive-allow` with a comment saying who decided and why:

```
# Reviewed 2026-09-12.
docs/working-with-agents.md:private-marker
docs/working-with-agents.md:cluster-path
tests/unit/test_sensitive_file_check.py:private-marker
tests/unit/test_sensitive_file_check.py:cluster-path
```

That is the whole file: this page and the check's own test, each exempt from
only the two rules its subject matter trips.

Prefer that scoped `path:rule` form over a bare path. A bare path waives every
rule on the file, which means a later edit that adds something genuinely
sensitive to it goes unnoticed; the scoped form still catches everything else.

The allowlist is deliberately a file in the repository rather than a
command-line flag. A flag gets typed once when the check is inconvenient,
forgotten, and the check is then dead. A file shows up in a diff and can be
questioned. If that list ever grows without comments, the check is being worked
around rather than used.

### Calibration

A check that mostly produces false positives gets switched off, so this one was
tuned against the real repository rather than in the abstract. On its first run
over every tracked file it produced three findings — and all three were files
*describing* the leak it exists to prevent: the changelog, the release guide, and
the test that checks the sdist.

That is a real failure mode, not a curiosity: a check whose only findings are
documentation of itself teaches people to ignore it. So the `private-marker`
rule now looks only at a file's first forty lines, where such a notice is
actually written, and skips lines that merely quote the phrase.

Two files could not be fixed that way, and both are self-referential: this
page, which quotes the private header inside a code block where the quoting
heuristic cannot help, and the check's own test file, which must contain the
things it detects. Each carries two rule-scoped exemptions. Nothing else in the
repository does, so a finding means something.

## Working with an agent specifically

None of the above is agent-specific — a collaborator with shell access can make
every one of these mistakes. But there are two things that raise the stakes.

**An agent moves faster than review.** Between the bulk add and the CI failure
that caught it there were only minutes, and the intervening commits were
plausible. The mitigation is not slower work; it is that the mechanical checks
run at commit time, where speed does not matter.

**An agent cannot tell which files are sensitive from their contents alone.**
It sees a `.sh` file in an examples directory. Whether that file embeds a path to
internal metadata is a fact about your data agreement, not about the code. So
that judgement is encoded in `scripts/check_sensitive_files.py` where it can be
read, tested and argued with, instead of being re-derived each time by whoever
is at the keyboard.

Two habits that cost nothing:

- Stage deliberately. `git add <specific paths>`, and `git status` before the
  commit. `git add -A` on this repository is how the first incident happened.
- After any bulk add, run `uv run pre-commit run --all-files`.

## Known gap

These guardrails protect against leaking data. They do not protect against
**silent degradation of scientific output** — an embedding that gets gradually
worse, a metric that drifts, a preprocessing change that quietly reduces the
sample count. The cohort test suite
([Testing against real cohorts](testing-real-cohorts.md)) asserts chance-corrected
statistics on real data and is the beginning of an answer, but it runs on demand
rather than as a gate, and it does not compare against previous runs.

That is tracked as open work rather than claimed as solved.
