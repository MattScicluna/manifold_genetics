"""What the package says about itself.

Everything here is metadata a user never sees until it is wrong on PyPI, at
which point it is wrong permanently: a release cannot be re-uploaded under the
same version. The license declaration is the sharp case -- the LICENSE file is
BSD 3-Clause and ``pyproject.toml`` claimed MIT, in both the license field and
the classifier, which would have published this package under a licence its
author never chose.

These are also the facts that live in three places at once (pyproject, the
package ``__init__``, the changelog) and so drift silently.
"""

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "manifold_genetics"

# The canonical repository. Written out because a wrong slug in project.urls is
# a 404 on the PyPI sidebar and nothing in a checkout would notice.
SLUG = "MattScicluna/manifold_genetics"


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((REPO / "pyproject.toml").read_text())


@pytest.fixture(scope="module")
def project(pyproject):
    return pyproject["project"]


class TestLicense:
    def test_the_declared_license_matches_the_license_file(self, project):
        declared = project["license"]
        if isinstance(declared, dict):
            declared = declared.get("text", "")

        first_line = (REPO / "LICENSE").read_text().splitlines()[0].strip()

        # "BSD 3-Clause License" -> the SPDX identifier "BSD-3-Clause".
        assert first_line.startswith("BSD 3-Clause")
        assert (
            declared == "BSD-3-Clause"
        ), f"pyproject declares {declared!r} but LICENSE is {first_line!r}"

    def test_no_classifier_contradicts_the_license_file(self, project):
        license_classifiers = [c for c in project["classifiers"] if c.startswith("License ::")]

        assert not [c for c in license_classifiers if "MIT" in c], license_classifiers


class TestVersion:
    def test_the_package_and_the_project_agree(self, project):
        declared = re.search(
            r'^__version__ = "([^"]+)"$', (PACKAGE / "__init__.py").read_text(), re.M
        )

        assert declared, "src/manifold_genetics/__init__.py declares no __version__"
        assert declared.group(1) == project["version"]

    def test_it_is_a_release_version(self, project):
        # PEP 440, and no leftover dev/rc suffix on something about to be tagged.
        assert re.fullmatch(r"\d+\.\d+\.\d+", project["version"]), project["version"]

    def test_the_changelog_documents_it(self, project):
        changelog = (REPO / "CHANGELOG.md").read_text()

        assert (
            f"## [{project['version']}]" in changelog
        ), f"CHANGELOG.md has no section for {project['version']}"


class TestTypeInformation:
    def test_the_package_ships_a_py_typed_marker(self):
        """Without it, the annotations in this package are invisible to callers.

        PEP 561: a type checker ignores a dependency's inline types unless the
        distribution says they are there.
        """
        assert (PACKAGE / "py.typed").exists()


class TestUrls:
    def test_every_url_points_at_the_real_repository(self, project):
        wrong = {
            name: url
            for name, url in project["urls"].items()
            if "github.com" in url and SLUG not in url
        }

        assert not wrong, f"these do not point at github.com/{SLUG}: {wrong}"

    def test_it_names_a_homepage_and_an_issue_tracker(self, project):
        assert {"Homepage", "Issues"} <= set(project["urls"])


class TestDependencies:
    def test_the_toml_parser_the_tests_need_is_a_dev_dependency(self, pyproject):
        """This test file imports it. Present transitively is not the same as
        declared, which is the argument that put pyyaml in the dependencies."""
        dev = pyproject["project"]["optional-dependencies"]["dev"]

        assert any(spec.startswith("tomli") for spec in dev), dev

    def test_no_runtime_dependency_is_unpinned_at_the_bottom(self, project):
        # A dependency with no lower bound resolves to whatever is on the index,
        # including versions predating the API this package uses.
        unbounded = [d for d in project["dependencies"] if not re.search(r"[><=~]", d)]

        assert not unbounded, unbounded
