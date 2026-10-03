"""Target-machine routing checks for optional reports after training."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import agfl_speech.cli as cli


def _config(root, *, device='cpu', attention='agfl'):
    # Only the keys the launcher itself reads; configuration resolution is mocked.
    return {'dataset': 'si_hom', 'model': 'eegnet', 'attention': attention,
            'model_variant': 'eeg', 'subject_id': None, 'seeds': [0],
            'split': {'protocol': 'stratified'},
            'device': device, 'training': {'epochs': 250}, 'output_dir': str(root)}


def _mock_launch(monkeypatch, configs, *, fail_command=None):
    import agfl_speech.engine

    calls = []
    launch = cli.main
    document = configs[0] if len(configs) == 1 else {'base': {}, 'experiments': configs}
    monkeypatch.setattr(cli, 'load_preset', lambda name: deepcopy(document))
    monkeypatch.setattr(cli, 'resolve_experiments', lambda config: [deepcopy(config)])

    def train(config, *, skip_completed):
        calls.append(('train', Path(config['output_dir']), config['device'], skip_completed))

    def command(argv):
        calls.append(('command', argv))
        if argv[0] == fail_command:
            raise RuntimeError(f'{fail_command} failed')

    monkeypatch.setattr(agfl_speech.engine, 'run_experiment', train)
    # Initial parsing uses the captured function; nested report commands use
    # this recorder, so these checks never train or reconstruct checkpoints.
    monkeypatch.setattr(cli, 'main', command)
    return launch, calls


def _diagnose(root, device):
    return ('command', ['diagnose-session', str(root.resolve()), '--device', device,
                        '--partition', 'validation', '--embedding', 'tsne'])


def _analyze(root):
    return ('command', ['analyze', str(root.resolve()), '--plots'])


@pytest.mark.parametrize('command', ['run', 'sweep'])
def test_report_routes_after_training_using_configured_device(tmp_path, monkeypatch, command, capsys):
    root = tmp_path / 'session'
    configs = [_config(root, device='cuda:1')]
    if command == 'sweep':
        configs.append(_config(root, device='cuda:1', attention='mha'))
    launch, calls = _mock_launch(monkeypatch, configs)
    launch([command, '--report', '--preset', 'mocked', '--skip-completed'])

    expected_training = [('train', root, 'cuda:1', True) for _ in configs]
    assert calls == [*expected_training, _diagnose(root, 'cuda:1'), _analyze(root)]
    output = capsys.readouterr().out
    assert f'Report generation finished. Download: {root / "report"}' in output


def test_sweep_reports_each_output_root_once_after_all_training(tmp_path, monkeypatch):
    first, second = tmp_path / 'a', tmp_path / 'b'
    configs = [_config(second, device='cuda:0'), _config(first),
               _config(second, device='cuda:0', attention='mha')]
    launch, calls = _mock_launch(monkeypatch, configs)

    launch(['sweep', '--preset', 'mocked', '--report', '--skip-completed'])

    assert calls == [
        ('train', second, 'cuda:0', True), ('train', first, 'cpu', True),
        ('train', second, 'cuda:0', True),
        _diagnose(first, 'cpu'), _analyze(first),
        _diagnose(second, 'cuda:0'), _analyze(second),
    ]


def test_mixed_device_session_explicitly_uses_its_first_device(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'mixed'
    launch, calls = _mock_launch(monkeypatch, [_config(root), _config(root, device='cuda')])

    launch(['sweep', '--preset', 'mocked', '--report'])

    assert calls[-2:] == [_diagnose(root, 'cpu'), _analyze(root)]
    assert 'using cpu from its first configuration' in capsys.readouterr().out


@pytest.mark.parametrize('command', ['run', 'sweep'])
def test_report_dry_run_does_not_train_or_generate_reports(tmp_path, monkeypatch, command, capsys):
    root = tmp_path / 'preview'
    configs = [_config(root)]
    if command == 'sweep':
        configs.append(_config(root, attention='mha'))
    launch, calls = _mock_launch(monkeypatch, configs)
    launch([command, '--report', '--dry-run', '--preset', 'mocked'])

    assert calls == []
    assert json.loads(capsys.readouterr().out)
    assert not root.exists()


@pytest.mark.parametrize('failed_command', ['diagnose-session', 'analyze'])
def test_report_failure_preserves_training_and_prints_manual_recovery(tmp_path, monkeypatch, capsys, failed_command):
    root = tmp_path / 'session with spaces'
    checkpoint = root / 'artifacts' / 'checkpoint.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'completed training checkpoint')
    launch, calls = _mock_launch(monkeypatch, [_config(root)], fail_command=failed_command)

    with pytest.raises(RuntimeError, match=f'{failed_command} failed'):
        launch(['run', '--preset', 'mocked', '--report', '--skip-completed'])

    expected = [('train', root, 'cpu', True), _diagnose(root, 'cpu'), _analyze(root)]
    assert calls == expected  # Failure never routes back into training.
    assert checkpoint.read_bytes() == b'completed training checkpoint'
    output = capsys.readouterr()
    assert 'Report generation finished' not in output.out
    assert 'Completed training artifacts are retained' in output.err
    assert 'recover the report without retraining' in output.err
    assert 'python -m agfl_speech diagnose-session' in output.err
    assert '--device cpu --partition validation --embedding tsne' in output.err
    assert 'python -m agfl_speech analyze' in output.err
    assert '--plots' in output.err
    if failed_command == 'diagnose-session':
        assert 'generating result plots from saved predictions' in output.err


def test_both_report_failures_preserve_the_original_diagnostic_error(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'failed-report'
    commands = []
    diagnostic_error = TypeError('diagnostic JSON serialization failed')

    def command(argv):
        commands.append(argv[0])
        if argv[0] == 'diagnose-session':
            raise diagnostic_error
        raise OSError('result output unavailable')

    monkeypatch.setattr(cli, 'main', command)
    with pytest.raises(TypeError) as captured:
        cli._generate_session_report(root, 'cuda')
    assert captured.value is diagnostic_error
    assert commands == ['diagnose-session', 'analyze']
    messages = capsys.readouterr()
    assert 'Result plot generation also failed: result output unavailable' in messages.err
    assert 'Report generation finished' not in messages.out


def test_report_preserves_nonzero_exit_status_from_diagnostic_command(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'invalid-report'
    launch, calls = _mock_launch(monkeypatch, [_config(root)])

    def diagnostic_exit(argv):
        calls.append(('command', argv))
        raise SystemExit(2)

    monkeypatch.setattr(cli, 'main', diagnostic_exit)
    with pytest.raises(SystemExit) as error:
        launch(['run', '--preset', 'mocked', '--report', '--skip-completed'])
    assert error.value.code == 2
    assert calls == [('train', root, 'cpu', True), _diagnose(root, 'cpu')]
    output = capsys.readouterr()
    assert 'Automatic report generation did not finish' in output.err
    assert 'Report generation finished' not in output.out


@pytest.mark.parametrize('command', ['run', 'sweep'])
def test_without_report_keeps_manual_plotting_workflow(tmp_path, monkeypatch, command, capsys):
    root = tmp_path / 'manual'
    configs = [_config(root)]
    if command == 'sweep':
        configs.append(_config(root, attention='mha'))
    launch, calls = _mock_launch(monkeypatch, configs)
    launch([command, '--output-dir', str(root), '--preset', 'mocked'])

    assert [call[0] for call in calls] == ['train'] * len(configs)
    output = capsys.readouterr().out
    assert 'python -m agfl_speech diagnose-session' in output
    assert 'python -m agfl_speech analyze' in output
    assert 'Report generation finished' not in output


@pytest.mark.parametrize('argv', [['plan', '--report'], ['analyze', 'session', '--report']])
def test_report_flag_is_not_accepted_by_other_commands(argv):
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2


def test_retired_search_command_is_rejected():
    with pytest.raises(SystemExit) as error:
        cli.main(['tune-eegnet', '--dry-run'])
    assert error.value.code == 2


@pytest.mark.parametrize('command', ['run', 'sweep'])
def test_data_dir_option_reaches_every_configuration_before_resolution(tmp_path, monkeypatch, command, capsys):
    root = tmp_path / 'preview'
    configs = [_config(root)]
    if command == 'sweep':
        configs.append(_config(root, attention='mha'))
    launch, calls = _mock_launch(monkeypatch, configs)

    launch([command, '--preset', 'mocked', '--data-dir', '../relocated-data', '--dry-run'])

    printed = json.loads(capsys.readouterr().out)
    resolved = printed if command == 'sweep' else [printed]
    assert [config['data'] for config in resolved] == [{'data_dir': '../relocated-data'}] * len(configs)
    assert calls == []
