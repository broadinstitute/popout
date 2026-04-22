"""Tests for CLI subcommand dispatch."""

import subprocess
import sys


def test_convert_requires_to():
    """popout convert without --to should fail."""
    result = subprocess.run(
        [sys.executable, "-m", "popout.cli", "convert"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "--to" in result.stderr or "required" in result.stderr


def test_convert_help():
    """popout convert --help should show convert-specific options."""
    result = subprocess.run(
        [sys.executable, "-m", "popout.cli", "convert", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--popout-prefix" in result.stdout
    assert "--input-vcf" in result.stdout
    assert "--thinned-sites" in result.stdout
