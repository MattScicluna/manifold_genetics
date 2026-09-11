"""--verbose must make *this package* verbose, not every library it imports.

``setup_logging`` called ``logging.basicConfig(level=DEBUG)``, which sets the
**root** logger, so ``--verbose`` turned on DEBUG for numba, pynndescent and
everything else. One embed run in 2026-09 put 665 KB of numba SSA dumps into a
log file in two minutes, burying the pipeline's own output.

Logging is global state, so each test restores it.
"""

import logging

import pytest

from manifold_genetics.cli import setup_logging

PACKAGE = "manifold_genetics"
THIRD_PARTY = "numba.core.ssa"


@pytest.fixture(autouse=True)
def restore_logging():
    """Isolate these tests from logging's global state.

    setup_logging deliberately reuses its own handler rather than stacking
    duplicates, so a handler left on root by an earlier test would be reused
    here -- still bound to that test's stderr, which capsys no longer watches.
    Stripping it first makes each test get a handler bound to its own stream.
    """
    root = logging.getLogger()
    saved = (root.level, list(root.handlers), logging.getLogger(PACKAGE).level)
    root.handlers = [
        h for h in root.handlers if not getattr(h, "_manifold_genetics_handler", False)
    ]
    yield
    root.level, root.handlers = saved[0], saved[1]
    logging.getLogger(PACKAGE).setLevel(saved[2])


class TestVerbose:
    def test_enables_debug_for_this_package(self):
        setup_logging(verbose=True)

        assert logging.getLogger(PACKAGE).isEnabledFor(logging.DEBUG)

    def test_does_not_enable_debug_for_third_party_libraries(self):
        setup_logging(verbose=True)

        assert not logging.getLogger(THIRD_PARTY).isEnabledFor(logging.DEBUG)

    def test_does_not_enable_info_for_third_party_libraries(self):
        setup_logging(verbose=True)

        assert not logging.getLogger(THIRD_PARTY).isEnabledFor(logging.INFO)

    def test_a_submodule_logger_inherits_the_package_level(self):
        setup_logging(verbose=True)

        assert logging.getLogger(f"{PACKAGE}.pca.flashpca").isEnabledFor(logging.DEBUG)


class TestQuiet:
    def test_enables_info_for_this_package(self):
        setup_logging(verbose=False)

        assert logging.getLogger(PACKAGE).isEnabledFor(logging.INFO)

    def test_does_not_enable_debug_for_this_package(self):
        setup_logging(verbose=False)

        assert not logging.getLogger(PACKAGE).isEnabledFor(logging.DEBUG)

    def test_third_party_warnings_still_get_through(self):
        # Quieting libraries must not silence their warnings, which are the ones
        # worth seeing -- graphtools' duplicate-sample warning, for instance.
        setup_logging(verbose=False)

        assert logging.getLogger(THIRD_PARTY).isEnabledFor(logging.WARNING)


class TestOutput:
    def test_package_records_still_reach_a_handler(self, capsys):
        """Scoping the level must not silence us: the records still need emitting."""
        setup_logging(verbose=True)

        logging.getLogger(f"{PACKAGE}.demo").debug("hello from the package")

        assert "hello from the package" in capsys.readouterr().err
