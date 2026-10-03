"""The ``check`` command: pre-flight verification of the data archive.

Development checks for the experiment machine only; they are not a prerequisite
for training. ``run_check`` accepts only the released recordings: the tiny
fixture archive is schema-correct and loads, but its trial counts differ from
``si_hom.EXPECTED_RELEASE``, so the check must stop with ``ValueError`` after
loading it. The complete check on the released archive runs only with
``AGFL_SPEECH_REAL_DATA=1``.

Run from the project folder: ``python -m pytest tests/test_check.py -q``.
"""
import os

import pytest

from agfl_speech.check import run_check
from agfl_speech.cli import main
from agfl_speech.datasets import load_dataset
from tests.si_hom_fixture import write_si_hom_fixture


def test_check_stops_on_an_archive_that_is_not_the_released_recordings(tmp_path, capsys):
    data_dir = write_si_hom_fixture(tmp_path / 'data')
    # The archive itself is valid, so the failure below is not a loading error.
    bundle = load_dataset('si_hom', {'data_dir': data_dir})
    with pytest.raises(ValueError):
        run_check(data_dir=data_dir)
    output = capsys.readouterr().out
    # The archive was loaded and described before the counts were compared.
    assert f'Dataset fingerprint: {bundle.fingerprint}' in output
    assert 'All checks passed' not in output


def test_check_command_passes_the_data_folder_to_the_check(tmp_path, capsys):
    data_dir = write_si_hom_fixture(tmp_path / 'data')
    with pytest.raises(ValueError):
        main(['check', '--data-dir', data_dir])
    assert 'All checks passed' not in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        main(['check', '--data-dir', str(tmp_path / 'no_archive_here')])


@pytest.mark.skipif(os.environ.get('AGFL_SPEECH_REAL_DATA') != '1',
                    reason='Set AGFL_SPEECH_REAL_DATA=1 on a machine holding the released archive')
def test_check_completes_on_the_released_archive(capsys):
    run_check(device='cpu')
    output = capsys.readouterr().out
    assert 'Trial, sample and subject counts match the released recordings.' in output
    assert 'All checks passed. Nothing was trained or saved.' in output
