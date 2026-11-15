"""Keep test temporaries on the configured scratch disk.

`BONSAI_FIXTURE_TMP` is the documented scratch directory for captures; the
suite writes multi-hundred-megabyte temp files, which do not fit the small
tmpfs that /tmp usually is.
"""
import os
import tempfile


def pytest_configure(config):
    root = os.environ.get('BONSAI_FIXTURE_TMP')
    if root:
        os.makedirs(root, exist_ok=True)
        tempfile.tempdir = root
