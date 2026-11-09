"""Smoke test: confirms the CI environment installs and imports correctly.

pytest fails when it finds no tests at all, so this keeps CI green until the
real tests (feature builder, label leakage, API contract) are added. Once those
exist, this file can stay as a quick check that the core libraries install.
"""


def test_core_libraries_import():
    import fastapi  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
