# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Console encoding helper.

Several scripts carry mathematical notation (Phi, sigma, nabla, arrows) in
their module docstrings, which argparse prints for ``--help``. On Windows the
console defaults to a legacy code page (cp1252) that cannot encode those
characters, so ``--help`` would raise ``UnicodeEncodeError`` before printing
anything.

``enable_unicode_stdout()`` reconfigures the standard streams to replace
unencodable characters instead of raising. The text degrades to a placeholder
glyph on a legacy console and is unchanged on a UTF-8 one.
"""
from __future__ import annotations

import sys


def enable_unicode_stdout() -> None:
    """Make stdout/stderr tolerant of characters the console cannot encode."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # redirected to a non-text stream; nothing to do
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            pass
