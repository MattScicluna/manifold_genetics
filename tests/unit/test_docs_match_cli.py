"""The command-line page must describe the command line that exists.

The CLI is how nearly everyone uses this package, so its page is the one where
drift costs the most -- and documentation drift is invisible to every other test
in this suite. Two real examples this guards against, both found on 2026-09-12:
``--n-components`` was documented as controlling embedding dimensionality and is
not a flag on any subcommand, and ``setup`` was described as downloading into
``bin/`` two releases after it moved to a per-user cache.

This checks the cheap, mechanical half -- that the set of subcommands agrees.
Prose still needs a human.
"""

import re
from pathlib import Path

import pytest

from manifold_genetics.cli import main

DOCS = Path(__file__).resolve().parents[2] / "docs" / "cli.md"


def documented_commands() -> set:
    """Subcommands named on the page, either in its table or in a code block."""
    text = DOCS.read_text()
    in_table = set(re.findall(r"^\| `([a-z][a-z-]*)` \|", text, re.MULTILINE))
    in_prose = set(re.findall(r"manifold-genetics ([a-z][a-z-]*)\b", text))
    return in_table | in_prose


def actual_commands(capsys) -> set:
    """Subcommands argparse itself lists, read from the real ``--help``.

    Taken from the CLI rather than from a hand-kept list, which would be a third
    copy to drift. ``main`` builds its parser internally, so the help text is the
    available seam; the ``{a,b,c}`` choices block is stable argparse output.
    """
    with pytest.raises(SystemExit):
        main(["--help"])

    text = capsys.readouterr().out
    block = re.search(r"\{([a-z0-9,\-]+)\}", text)
    assert block, "could not find the subcommand list in --help output"
    return set(block.group(1).split(","))


def test_every_subcommand_is_documented(capsys):
    undocumented = actual_commands(capsys) - documented_commands()

    assert undocumented == set(), (
        f"subcommands missing from docs/cli.md: {sorted(undocumented)}. "
        "A command nobody can find is a command nobody uses."
    )


def test_no_subcommand_is_documented_that_does_not_exist(capsys):
    """A documented command that was renamed or removed sends people to an error."""
    phantom = documented_commands() - actual_commands(capsys)

    assert phantom == set(), f"docs/cli.md documents commands that do not exist: {sorted(phantom)}"
