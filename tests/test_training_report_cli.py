"""Target-machine routing checks for optional reports after training."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import agfl.cli as cli


def _config(root, *, device='cpu', attention='agfl'):
    return {'dataset': 'eeg', 'model': 'eegnet', 'attention': attention,
            'model_variant': 'eeg', 'subject_id': 'A03', 'seeds': [0],
            'device': device, 'training': {'epochs': 250}, 'output_dir': str(root)}


def _mock_launch(monkeypatch, configs, *, fail_command=None):
    import agfl.engine
    import agfl.models.eegnet.search as search

    calls = []
    launch = cli.main
    document = configs[0] if len(configs) == 1 else {'base': {}, 'experiments': configs}
    monkeypatch.setattr(cli, 'load_preset', lambda name: deepcopy(document))
    monkeypatch.setattr(cli, 'resolve_experiments', lambda config: [deepcopy(config)])

    def train(config, *, skip_completed):
        calls.append(('train', Path(config['output_dir']), config['device'], skip_completed))

    def tune(configurations, *, restart):
        calls.append(('tune', restart))

    def command(argv):
        calls.append(('command', argv))
        if argv[0] == fail_command:
            raise RuntimeError(f'{fail_command} failed')

    monkeypatch.setattr(agfl.engine, 'run_experiment', train)
    monkeypatch.setattr(search, 'search_configs', lambda *args, **kwargs: {'candidate': deepcopy(configs)})
    monkeypatch.setattr(search, 'run_search', tune)
    # Initial parsing uses the captured function; nested report commands use
    # this recorder, so these checks never train or reconstruct checkpoints.
    monkeypatch.setattr(cli, 'main', command)
    return launch, calls


def _diagnose(root, device):
    return ('command', ['diagnose-session', str(root.resolve()), '--device', device,
                        '--partition', 'validation', '--embedding', 'tsne'])


def _analyze(root):
    return ('command', ['analyze', str(root.resolve()), '--plots'])


@pytest.mark.parametrize('command', ['run', 'sweep', 'tune-eegnet'])
def test_report_routes_after_training_using_configured_device(tmp_path, monkeypatch, command, capsys):
    root = tmp_path / 'session'
    configs = [_config(root, device='cuda:1')]
    if command == 'sweep':
        configs.append(_config(root, device='cuda:1', attention='mha'))
    launch, calls = _mock_launch(monkeypatch, configs)
    args = [command, '--report']
    if command == 'tune-eegnet':
        args += ['--output-dir', str(root)]
    else:
        args += ['--preset', 'mocked', '--skip-completed']

    launch(args)

    expected_training = [('tune', False)] if command == 'tune-eegnet' else [
        ('train', root, 'cuda:1', True) for _ in configs]
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


@pytest.mark.parametrize('command', ['run', 'sweep', 'tune-eegnet'])
def test_report_dry_run_does_not_train_or_generate_reports(tmp_path, monkeypatch, command, capsys):
    root = tmp_path / 'preview'
    configs = [_config(root)]
    if command == 'sweep':
        configs.append(_config(root, attention='mha'))
    launch, calls = _mock_launch(monkeypatch, configs)
    args = [command, '--report', '--dry-run']
    if command == 'tune-eegnet':
        args += ['--output-dir', str(root)]
    else:
        args += ['--preset', 'mocked']

    launch(args)

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

    expected = [('train', root, 'cpu', True), _diagnose(root, 'cpu')]
    if failed_command == 'analyze':
        expected.append(_analyze(root))
    assert calls == expected  # Failure never routes back into training.
    assert checkpoint.read_bytes() == b'completed training checkpoint'
    output = capsys.readouterr()
    assert 'Report generation finished' not in output.out
    assert 'Completed training artifacts are retained' in output.err
    assert 'recover the report without retraining' in output.err
    assert 'python main.py diagnose-session' in output.err
    assert '--device cpu --partition validation --embedding tsne' in output.err
    assert 'python main.py analyze' in output.err
    assert '--plots' in output.err


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


@pytest.mark.parametrize('command', ['run', 'sweep', 'tune-eegnet'])
def test_without_report_keeps_manual_plotting_workflow(tmp_path, monkeypatch, command, capsys):
    root = tmp_path / 'manual'
    configs = [_config(root)]
    if command == 'sweep':
        configs.append(_config(root, attention='mha'))
    launch, calls = _mock_launch(monkeypatch, configs)
    args = [command, '--output-dir', str(root)]
    if command != 'tune-eegnet':
        args += ['--preset', 'mocked']

    launch(args)

    assert all(call[0] != 'command' for call in calls)
    output = capsys.readouterr().out
    assert 'python main.py diagnose-session' in output
    assert 'python main.py analyze' in output
    assert 'Report generation finished' not in output


@pytest.mark.parametrize('argv', [['plan', '--report'], ['analyze', 'session', '--report']])
def test_report_flag_is_not_accepted_by_other_commands(argv):
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2


def test_named_candidate_group_preserves_declared_order_in_preview(tmp_path, monkeypatch, capsys):
    import agfl.models.eegnet.search as search

    root = tmp_path / 'capacity'
    launch, calls = _mock_launch(monkeypatch, [_config(root)])
    names = ('spatial_qkv_sparse', 'spatial_qkv_capacity_focal_005',
             'spatial_qkv_capacity_focal_001', 'spatial_qkv_capacity_ce_005',
             'spatial_qkv_capacity_ce_001')
    monkeypatch.setattr(search, 'CANDIDATE_SETS', {'capacity': names})
    received = []

    def configurations(data_dir, output_dir, subjects, seeds, candidates, epochs):
        received.append((subjects, seeds, candidates, epochs))
        return {name: [{**_config(root), 'seeds': seeds}] for name in candidates}

    monkeypatch.setattr(search, 'search_configs', configurations)

    launch(['tune-eegnet', '--candidate-set', 'capacity', '--subjects', '3',
            '--seeds', '0', '1', '2', '3', '4', '--epochs', '250',
            '--output-dir', str(root), '--report', '--dry-run'])

    assert received == [([3], [0, 1, 2, 3, 4], list(names), 250)]
    assert tuple(json.loads(capsys.readouterr().out)) == names
    assert calls == []
    assert not root.exists()


@pytest.mark.parametrize('options', [
    ['--candidate', 'spatial_qkv_sparse', '--candidate-set', 'capacity'],
    ['--candidate-set', 'capacity', '--candidate', 'spatial_qkv_sparse'],
])
def test_named_group_and_individual_candidates_are_mutually_exclusive(options):
    with pytest.raises(SystemExit) as error:
        cli.main(['tune-eegnet', '--dry-run', *options])
    assert error.value.code == 2


def test_unknown_candidate_group_is_rejected_before_configuration_or_training(tmp_path, monkeypatch, capsys):
    import agfl.models.eegnet.search as search

    launch, calls = _mock_launch(monkeypatch, [_config(tmp_path / 'out')])
    monkeypatch.setattr(search, 'CANDIDATE_SETS', {'capacity': ('spatial_qkv_sparse',)})

    def forbidden(*args, **kwargs):
        raise AssertionError('Unknown candidate group reached configuration resolution')

    monkeypatch.setattr(search, 'search_configs', forbidden)
    with pytest.raises(SystemExit) as error:
        launch(['tune-eegnet', '--candidate-set', 'unknown', '--dry-run'])
    assert error.value.code == 2
    assert 'Choose from: capacity' in capsys.readouterr().err
    assert calls == []
