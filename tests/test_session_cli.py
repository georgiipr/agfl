"""Session CLI routing checks; execute on the cluster only, without training."""
import json
from pathlib import Path
import shutil

import pytest

from agfl_speech.cli import main
from agfl_speech.session import mirror_run, session_paths


def _saved_run(directory, *, status='completed', failure=False):
    directory.mkdir(parents=True)
    for name, record in {
        'config.json': {'dataset': 'si_hom', 'model': 'eegnet', 'attention': 'agfl'},
        'split.json': {'split_id': 'unchanged'},
        'history.json': [{'epoch': 1}],
    }.items():
        (directory / name).write_text(json.dumps(record))
    if status is not None:
        (directory / 'result.json').write_text(json.dumps({'status': status}))
    if failure:
        (directory / 'failure.json').write_text('{"type": "KeyboardInterrupt"}')
    (directory / 'checkpoint.pt').write_bytes(b'checkpoint: do not deserialize in CLI routing checks')
    (directory / 'predictions.npz').write_bytes(b'compact predictions')
    return directory


def _capture_analysis(monkeypatch):
    import agfl_speech.analysis
    import agfl_speech.visualization.results

    calls = []
    aggregation = {'from': 'mocked saved-artifact analysis'}

    def analyze(input_root, output_dir):
        calls.append(('analyze', Path(input_root), Path(output_dir)))
        return aggregation

    def plot(actual, output_dir, *, diagnostics_root=None):
        assert actual is aggregation
        calls.append(('plots', Path(output_dir), Path(diagnostics_root) if diagnostics_root else None))
        return {'figures': []}

    monkeypatch.setattr(agfl_speech.analysis, 'analyze_results', analyze)
    monkeypatch.setattr(agfl_speech.visualization.results, 'generate_result_plots', plot)
    return calls


def _capture_diagnostics(monkeypatch):
    import agfl_speech.visualization.diagnostics

    calls = []

    def diagnose(run_dir, output_dir, **kwargs):
        calls.append((Path(run_dir), Path(output_dir), kwargs))
        return {'figures': []}

    monkeypatch.setattr(agfl_speech.visualization.diagnostics, 'diagnose_run', diagnose)
    return calls


def test_organize_command_migrates_files_without_loading_checkpoint(tmp_path, capsys):
    root = tmp_path / 'existing session'
    relative = Path('selected/si_hom/subject_S03/seed_0')
    old_run = _saved_run(root / relative)
    checkpoint = (old_run / 'checkpoint.pt').read_bytes()
    figures = root / 'analysis' / 'figures'
    figures.mkdir(parents=True)
    (figures / 'figure.png').write_bytes(b'previously generated figure')

    main(['organize-results', str(root)])

    assert not old_run.exists()
    assert (root / 'artifacts' / relative / 'checkpoint.pt').read_bytes() == checkpoint
    assert (root / 'report' / 'runs' / relative / 'result.json').is_file()
    assert not list((root / 'report').rglob('checkpoint.pt'))
    assert (root / 'report' / 'analysis' / 'figures' / 'figure.png').is_file()
    output = capsys.readouterr().out
    assert str(root / 'report') in output
    assert 'python -m agfl_speech diagnose-session' in output
    assert 'python -m agfl_speech analyze' in output


@pytest.mark.parametrize('input_section', ['', 'artifacts', 'report', 'report/runs'])
def test_analyze_defaults_to_session_report_and_finds_all_report_diagnostics(tmp_path, monkeypatch, input_section):
    session = session_paths(tmp_path / 'session', create=True)
    run = _saved_run(session.artifacts / 'si_hom' / 'seed_0')
    mirror_run(session.root, run)
    calls = _capture_analysis(monkeypatch)
    source = session.root / input_section

    main(['analyze', str(source), '--plots'])

    assert calls == [
        ('analyze', source, session.report / 'analysis'),
        ('plots', session.report / 'analysis' / 'figures', session.report),
    ]


def test_renamed_downloaded_report_keeps_analysis_inside_portable_folder(tmp_path, monkeypatch):
    session = session_paths(tmp_path / 'cluster-session', create=True)
    mirror_run(session.root, _saved_run(session.artifacts / 'si_hom' / 'seed_0'))
    downloaded = tmp_path / 'review-download'
    shutil.copytree(session.report, downloaded)
    calls = _capture_analysis(monkeypatch)

    main(['analyze', str(downloaded), '--plots'])

    assert calls == [
        ('analyze', downloaded, downloaded / 'analysis'),
        ('plots', downloaded / 'analysis' / 'figures', downloaded),
    ]
    assert not (downloaded / 'artifacts').exists()


def test_analyze_explicit_output_and_diagnostic_root_are_honored(tmp_path, monkeypatch):
    session = session_paths(tmp_path / 'session', create=True)
    calls = _capture_analysis(monkeypatch)
    output, diagnostics = tmp_path / 'custom analysis', tmp_path / 'custom diagnostics'

    main(['analyze', str(session.root), '--plots', '--output-dir', str(output),
          '--diagnostics-root', str(diagnostics)])

    assert calls == [('analyze', session.root, output), ('plots', output / 'figures', diagnostics)]


@pytest.mark.parametrize('command', ['analyze', 'diagnose', 'diagnose-session'])
def test_in_session_output_outside_report_is_rejected_before_writing(tmp_path, monkeypatch, capsys, command):
    session = session_paths(tmp_path / 'session', create=True)
    run = _saved_run(session.artifacts / 'si_hom' / 'seed_0')
    mirror_run(session.root, run)
    calls = _capture_analysis(monkeypatch) if command == 'analyze' else _capture_diagnostics(monkeypatch)
    source = run if command == 'diagnose' else session.root
    invalid = session.root / 'analysis'

    with pytest.raises(SystemExit) as error:
        main([command, str(source), '--output-dir', str(invalid)])

    assert error.value.code == 2
    assert 'Outputs inside this session must be under' in capsys.readouterr().err
    assert not invalid.exists()
    assert not calls
    # A rejected old-style output must not turn an initialized session into a
    # mixed layout that breaks the next training or plotting command.
    assert session_paths(session.root, create=True) == session
    main([command, str(source)])
    assert len(calls) == 1
    output = calls[0][2] if command == 'analyze' else calls[0][1]
    assert output.is_relative_to(session.report)


@pytest.mark.parametrize('custom_output', [False, True])
def test_diagnose_session_uses_completed_canonical_runs_once(tmp_path, monkeypatch, custom_output):
    session = session_paths(tmp_path / 'session', create=True)
    good = [
        _saved_run(session.artifacts / 'si_hom' / 'subject_S03' / 'eegnet-agfl-example' / 'seed_0'),
        _saved_run(session.artifacts / 'selected' / 'si_hom' / 'subject_S03' / 'seed_1'),
    ]
    for run in good:
        mirror_run(session.root, run)
    candidate = _saved_run(session.artifacts / 'candidates' / 'dense' / 'S03' / 'seed_0', status=None)
    (candidate / 'selection.json').write_text('{"status": "validation_completed"}')
    mirror_run(session.root, candidate)
    _saved_run(session.artifacts / 'si_hom' / 'seed_2', status='validation_completed')
    _saved_run(session.artifacts / 'si_hom' / 'seed_3', failure=True)
    calls = _capture_diagnostics(monkeypatch)
    destination = tmp_path / 'custom diagnostics' if custom_output else session.report / 'diagnostics'
    args = ['diagnose-session', str(session.root), '--device', 'cuda', '--partition', 'validation',
            '--max-samples', '32', '--embedding', 'pca', '--data-dir', '../relocated-data']
    if custom_output:
        args.extend(['--output-dir', str(destination)])

    main(args)

    assert [run for run, _, _ in calls] == sorted(good)
    for run, output, options in calls:
        assert output == destination / run.relative_to(session.artifacts)
        assert options == {'device': 'cuda', 'partition': 'validation', 'max_samples': 32,
                           'embedding': 'pca', 'data_dir': '../relocated-data'}


@pytest.mark.parametrize('report_input', [False, True])
@pytest.mark.parametrize('custom_output', [False, True])
def test_single_diagnose_resolves_report_run_and_chooses_output(tmp_path, monkeypatch, report_input, custom_output):
    session = session_paths(tmp_path / 'session', create=True)
    run = _saved_run(session.artifacts / 'selected' / 'si_hom' / 'subject_S03' / 'seed_0')
    published = mirror_run(session.root, run)
    calls = _capture_diagnostics(monkeypatch)
    destination = (tmp_path / 'custom diagnostics' if custom_output else
                   session.report / 'diagnostics' / run.relative_to(session.artifacts))
    args = ['diagnose', str(published if report_input else run), '--device', 'cuda']
    if custom_output:
        args.extend(['--output-dir', str(destination)])

    main(args)

    assert len(calls) == 1
    assert calls[0][:2] == (run, destination)
    assert calls[0][2] == {'device': 'cuda', 'partition': 'validation', 'max_samples': 256,
                           'embedding': 'tsne', 'data_dir': None}


@pytest.mark.parametrize('command', ['diagnose', 'diagnose-session'])
@pytest.mark.parametrize('flag', ['--labels-dir', '--data-path'])
def test_removed_relocation_flags_are_rejected(tmp_path, monkeypatch, command, flag):
    session = session_paths(tmp_path / 'session', create=True)
    run = _saved_run(session.artifacts / 'si_hom' / 'seed_0')
    mirror_run(session.root, run)
    calls = _capture_diagnostics(monkeypatch)

    with pytest.raises(SystemExit) as error:
        main([command, str(run if command == 'diagnose' else session.root), flag, 'elsewhere'])

    assert error.value.code == 2
    assert not calls


def test_diagnose_session_rejects_report_only_download(tmp_path, monkeypatch, capsys):
    session = session_paths(tmp_path / 'cluster-session', create=True)
    downloaded = tmp_path / 'downloaded-report'
    shutil.copytree(session.report, downloaded)
    calls = _capture_diagnostics(monkeypatch)

    with pytest.raises(SystemExit) as error:
        main(['diagnose-session', str(downloaded)])

    assert error.value.code == 2
    assert not calls
    assert 'complete session with artifacts' in capsys.readouterr().err


def test_diagnose_session_rejects_no_completed_runs(tmp_path, monkeypatch, capsys):
    session = session_paths(tmp_path / 'session', create=True)
    _saved_run(session.artifacts / 'si_hom' / 'seed_0', failure=True)
    calls = _capture_diagnostics(monkeypatch)

    with pytest.raises(SystemExit) as error:
        main(['diagnose-session', str(session.root)])

    assert error.value.code == 2
    assert not calls
    assert 'No completed runs' in capsys.readouterr().err
