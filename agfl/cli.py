"""Resolve saved configurations or presets and delegate to model experiments."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from .config import apply_overrides, merge, read_config, resolve_experiments, digest
from .presets import load_preset, preset_names


def _configuration_arguments(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', help='JSON or TOML configuration')
    source.add_argument('--preset', help='Built-in preset; see the presets command')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--output-dir')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=JSON')
    parser.add_argument('--model', help='Backbone key, such as eegnet or conformer')
    parser.add_argument('--attention', help='Attention key: agfl, mha, performer, linformer, nystromformer')


def _diagnostic_arguments(parser):
    parser.add_argument('--output-dir', help='Defaults to the containing session report/diagnostics directory')
    parser.add_argument('--data-dir', help='Relocated directory containing the original recordings')
    parser.add_argument('--labels-dir', help='Relocated EEG E-session label directory')
    parser.add_argument('--data-path', help='Relocated NPZ dataset file')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--partition', choices=('validation', 'test'), default='validation')
    parser.add_argument('--max-samples', type=int, default=256)
    parser.add_argument('--embedding', choices=('pca', 'tsne', 'umap'), default='tsne')


def _session_plot_commands(root, *, device='cuda', stream=None):
    import shlex
    root = Path(root).expanduser().resolve()
    print('Generate checkpoint diagnostic plots on the experiment machine:', file=stream)
    print(f'python main.py diagnose-session {shlex.quote(str(root))} '
          f'--device {shlex.quote(str(device))} --partition validation --embedding tsne', file=stream)
    print('Generate result plots and tables:', file=stream)
    print(f'python main.py analyze {shlex.quote(str(root))} --plots', file=stream)
    print(f'Download for review: {root / "report"}', file=stream)
    print(f'Keep checkpoints and working files on the cluster: {root / "artifacts"}', file=stream)


def _generate_session_report(root, device):
    """Run existing report commands after training locks have been released."""
    root = Path(root).expanduser().resolve()
    print(f'Generating checkpoint diagnostics and result plots: {root}', flush=True)
    try:
        try:
            main(['diagnose-session', str(root), '--device', str(device),
                  '--partition', 'validation', '--embedding', 'tsne'])
        except Exception:
            # Saved predictions do not depend on checkpoint visualizations.
            # Preserve useful result plots even if one diagnostic fails, then
            # propagate the original failure so a partial report is explicit.
            print('Checkpoint diagnostics failed; generating result plots from saved predictions.',
                  file=sys.stderr)
            try:
                main(['analyze', str(root), '--plots'])
            except Exception as analysis_error:
                print(f'Result plot generation also failed: {analysis_error}', file=sys.stderr)
            raise
        main(['analyze', str(root), '--plots'])
    except BaseException:
        print('Automatic report generation did not finish. Completed training artifacts '
              'are retained; recover the report without retraining:', file=sys.stderr)
        _session_plot_commands(root, device=device, stream=sys.stderr)
        raise
    print(f'Report generation finished. Download: {root / "report"}')
    print(f'Keep checkpoints and working files on the cluster: {root / "artifacts"}')


def _finish_training_reports(configs, *, automatic):
    sessions = {}
    for config in configs:
        root = Path(config['output_dir']).expanduser().resolve()
        sessions.setdefault(root, []).append(str(config['device']))
    for root, devices in sorted(sessions.items()):
        device = devices[0]
        print('\nTraining finished.')
        if automatic:
            if len(set(devices)) > 1:
                print(f'Multiple training devices in this session; using {device} '
                      'from its first configuration for checkpoint diagnostics.')
            _generate_session_report(root, device)
        else:
            _session_plot_commands(root, device=device)


def _report_destination(parser, output, session):
    """Keep explicit in-session outputs inside its downloadable report."""
    destination = Path(output).expanduser().resolve()
    if (session is not None and destination.is_relative_to(session.root.resolve())
            and not destination.is_relative_to(session.report.resolve())):
        parser.error(f'Outputs inside this session must be under {session.report}. '
                     'Omit --output-dir for the default, or choose a directory outside the session.')
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reproducible EEG/ECG AGFL experiments')
    commands = parser.add_subparsers(dest='command')
    run = commands.add_parser('run', help='Launch one configuration; matching seed runs are overwritten')
    _configuration_arguments(run)
    run.add_argument('--dataset')
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--skip-completed', action='store_true', help='Keep matching completed seeds; restart incomplete seeds')
    run.add_argument('--report', action='store_true', help='After training, generate validation checkpoint diagnostics and result plots')
    sweep = commands.add_parser('sweep', help='Launch a comparison/ablation matrix; matching seed runs are overwritten')
    _configuration_arguments(sweep)
    sweep.add_argument('--dry-run', action='store_true')
    sweep.add_argument('--skip-completed', action='store_true', help='Keep matching completed seeds; restart incomplete seeds')
    sweep.add_argument('--report', action='store_true', help='After training, generate validation checkpoint diagnostics and result plots')
    tune = commands.add_parser('tune-eegnet', help='Validation-selected EEGNet + AGFL search; test only selected checkpoints')
    tune.add_argument('--data-dir', default='../ml', help='Directory containing A01T.gdf ... A09T.gdf')
    tune.add_argument('--output-dir', default='results/eegnet-interchannel-search-v1')
    tune.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 10)))
    tune.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    candidate_source = tune.add_mutually_exclusive_group()
    candidate_source.add_argument('--candidate', action='append', help=(
        'Select an electrode-graph candidate; repeat to choose several. '
        'Default: electrode_control, electrode_dense, electrode_top8. '
        'Old temporal EEG candidates are retired.'))
    candidate_source.add_argument('--candidate-set', help=(
        'Use a named bounded candidate group, such as electrode_sparsity; '
        'cannot be combined with --candidate.'))
    tune.add_argument('--epochs', type=int, help='Override epoch budget, e.g. 2 for a separate smoke check')
    tune.add_argument('--dry-run', action='store_true', help='Print configurations without loading recordings')
    tune.add_argument('--restart', action='store_true', help='Retrain completed candidates too; otherwise resume matching completed runs')
    tune.add_argument('--report', action='store_true', help='After selected-checkpoint evaluation, generate validation diagnostics and result plots')
    plan = commands.add_parser('plan', help='Show resolved settings and launch command without loading data')
    _configuration_arguments(plan)
    analysis = commands.add_parser('analyze')
    analysis.add_argument('results')
    analysis.add_argument('--output-dir', help='Defaults to the session report/analysis directory')
    analysis.add_argument('--plots', action='store_true', help='Also write PNG/PDF figures from saved results')
    analysis.add_argument('--diagnostics-root', help='Include sparsity/entropy versus accuracy plots from saved checkpoint diagnostics; requires --plots')
    diagnostics = commands.add_parser('diagnose', help='Plot signals, embeddings and mixer maps from a saved checkpoint')
    diagnostics.add_argument('run_dir', help='One seed directory containing config.json, split.json and checkpoint.pt')
    _diagnostic_arguments(diagnostics)
    session_diagnostics = commands.add_parser('diagnose-session', help='Generate checkpoint plots for all completed runs in one session')
    session_diagnostics.add_argument('session', help='Session directory containing artifacts/ and report/')
    _diagnostic_arguments(session_diagnostics)
    organize = commands.add_parser('organize-results', help='Separate a finished session into report/ and artifacts/ without training')
    organize.add_argument('session', help='One existing result session; stop its writers before organizing')
    audit = commands.add_parser('audit-eeg', help='Audit a saved BCI IV 2a T-session run against its raw recording; no training')
    audit.add_argument('run_dir', help='One seed directory containing config.json and split.json')
    audit.add_argument('--data-dir', help='Relocated directory containing the matching AxxT.gdf recording')
    audit.add_argument('--output-dir', required=True, help='Separate audit output directory')
    audit_plots = commands.add_parser('plot-eeg-audit', help='Generate raw-audit figures and a self-contained HTML report from saved audit.json')
    audit_plots.add_argument('audit_dir', help='Directory produced by audit-eeg')
    commands.add_parser('list')
    commands.add_parser('presets')
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        print('\nSuggested launch on the experiment machine: python main.py run --preset eeg')
        print('ECG: python main.py run --preset ecg')
        print('Preview: python main.py plan --preset eeg-comparison')
        return
    if args.command == 'presets':
        print('\n'.join(preset_names()))
        return
    if args.command == 'list':
        from .datasets import list_datasets
        from .models import available_models
        from .attention import available_attentions
        print('Models: ' + ', '.join(available_models()))
        print('Attentions: ' + ', '.join(available_attentions()))
        print('Datasets: ' + ', '.join(list_datasets()))
        return
    if args.command == 'organize-results':
        from .session import organize_session
        session = organize_session(args.session)
        print(f'Report folder ready: {session.report}')
        print(f'Checkpoints and working files: {session.artifacts}')
        print('Existing plots were preserved. This command does not train models or generate missing plots.')
        _session_plot_commands(session.root)
        return
    if args.command == 'analyze':
        if args.diagnostics_root and not args.plots:
            parser.error('--diagnostics-root requires --plots')
        from .analysis import analyze_results
        from .session import find_session, session_activity
        from .storage import run_directory
        session = find_session(args.results)
        output = (Path(args.output_dir).expanduser() if args.output_dir else
                  session.report / 'analysis' if session else Path(args.results).expanduser() / 'analysis')
        output = _report_destination(parser, output, session)
        diagnostic_root = args.diagnostics_root
        if diagnostic_root is None and session is not None:
            # Organized older sessions can retain analysis/diagnostics as well
            # as the new diagnostics/ tree. Discovery ignores non-diagnostic manifests.
            diagnostic_root = session.report
        with ExitStack() as activity:
            if session is not None and session.artifacts.is_dir():
                activity.enter_context(session_activity(session.root))
            activity.enter_context(run_directory(output))
            result = analyze_results(args.results, output)
            if args.plots:
                from .visualization.results import generate_result_plots
                generate_result_plots(result, output / 'figures', diagnostics_root=diagnostic_root)
                print(f'Plot index: {output / "figures/index.md"}')
        print(f'Result tables: {output / "report.md"}')
        return
    if args.command == 'diagnose':
        from .result_paths import resolve_artifact_run
        from .session import find_session, session_activity
        from .storage import run_directory
        directory = resolve_artifact_run(args.run_dir)
        session = find_session(directory)
        if args.output_dir:
            output = Path(args.output_dir).expanduser()
        elif session is not None:
            output = session.report / 'diagnostics' / directory.relative_to(session.artifacts.resolve())
        else:
            parser.error('For an older run outside an organized session, supply --output-dir or run organize-results first')
        output = _report_destination(parser, output, session)
        from .visualization.diagnostics import diagnose_run
        with ExitStack() as activity:
            if session is not None:
                activity.enter_context(session_activity(session.root))
            activity.enter_context(run_directory(directory))
            diagnose_run(directory, output, device=args.device, partition=args.partition,
                         max_samples=args.max_samples, embedding=args.embedding, data_dir=args.data_dir,
                         labels_dir=args.labels_dir, data_path=args.data_path)
        print(f'Diagnostic plot index: {output / "index.md"}')
        return
    if args.command == 'diagnose-session':
        from .result_paths import result_paths
        from .session import find_session, session_activity
        session = find_session(args.session)
        if session is None or not session.artifacts.is_dir():
            parser.error('diagnose-session needs the complete session with artifacts on the cluster; '
                         'use organize-results first for an older layout')
        output = Path(args.output_dir).expanduser() if args.output_dir else session.report / 'diagnostics'
        output = _report_destination(parser, output, session)
        from .storage import run_directory
        from .visualization.diagnostics import diagnose_run
        completed = 0
        with session_activity(session.root):
            for result_path in result_paths(session.artifacts):
                directory = result_path.parent
                with run_directory(directory):
                    # Re-check after locking: an interrupted replacement may
                    # have invalidated a result found during discovery.
                    if not result_path.is_file() or (directory / 'failure.json').exists():
                        continue
                    record = json.loads(result_path.read_text())
                    if record.get('status') != 'completed':
                        continue
                    destination = output / directory.relative_to(session.artifacts.resolve())
                    print(f'Checkpoint plots: {directory}', flush=True)
                    diagnose_run(directory, destination, device=args.device, partition=args.partition,
                                 max_samples=args.max_samples, embedding=args.embedding, data_dir=args.data_dir,
                                 labels_dir=args.labels_dir, data_path=args.data_path)
                    completed += 1
        if completed == 0:
            parser.error('No completed runs with checkpoints were found in this session')
        print(f'Diagnostic plots for {completed} runs: {output}')
        return
    if args.command == 'audit-eeg':
        from .eeg_audit import audit_eeg_run
        from .session import find_session, session_activity
        session = find_session(args.output_dir) or find_session(args.run_dir)
        output = _report_destination(parser, args.output_dir, session)
        with ExitStack() as activity:
            if session is not None and session.artifacts.is_dir():
                activity.enter_context(session_activity(session.root))
            result = audit_eeg_run(args.run_dir, output, data_dir=args.data_dir)
        failed = sum(check['status'] == 'fail' for check in result['checks'])
        print(f'EEG audit: {failed} failed checks. Review {output / "report.md"}')
        import shlex
        print('Generate audit plots and the single-file report:')
        print(f'python main.py plot-eeg-audit {shlex.quote(str(output))}')
        return
    if args.command == 'plot-eeg-audit':
        from .session import find_session, session_activity
        from .visualization.eeg_audit import plot_eeg_audit
        session = find_session(args.audit_dir)
        with ExitStack() as activity:
            if session is not None and session.artifacts.is_dir():
                activity.enter_context(session_activity(session.root))
            plot_eeg_audit(args.audit_dir)
        print(f'Single-file report: {args.audit_dir}/report.html')
        print(f'Audit plot index: {args.audit_dir}/figures/index.md')
        return
    if args.command == 'tune-eegnet':
        from .models.eegnet.search import CANDIDATE_SETS, search_configs, run_search
        candidates = args.candidate
        if args.candidate_set is not None:
            if args.candidate_set not in CANDIDATE_SETS:
                parser.error('Unknown candidate set. Choose from: ' + ', '.join(CANDIDATE_SETS))
            candidates = list(CANDIDATE_SETS[args.candidate_set])
        configs = search_configs(args.data_dir, args.output_dir, args.subjects, args.seeds,
                                 candidates, args.epochs)
        if args.dry_run:
            print(json.dumps(configs, indent=2))
            return
        fits = sum(len(config['seeds']) for experiments in configs.values() for config in experiments)
        print(f'EEGNet + AGFL / BCI IV 2a: {fits} validation-only fits; then test only each subject/seed winner.', flush=True)
        run_search(configs, restart=args.restart)
        root = Path(args.output_dir).expanduser().resolve()
        print(f'Overall selected accuracy and per-subject results: {root / "report/search_result.json"}')
        _finish_training_reports(
            [config for experiments in configs.values() for config in experiments],
            automatic=args.report,
        )
        return
    document = read_config(args.config) if args.config else load_preset(args.preset) if args.preset else {}
    is_sweep = 'base' in document or 'experiments' in document
    if is_sweep:
        if set(document) != {'base', 'experiments'} or not document['experiments']:
            parser.error('A sweep requires base and a nonempty experiments list')
        if args.command == 'run':
            parser.error('Use sweep for a comparison/ablation preset')
        configs = [merge(document['base'], experiment) for experiment in document['experiments']]
    else:
        if args.command == 'sweep':
            parser.error('Select a comparison/ablation preset or a sweep configuration')
        configs = [document]
    resolved_configs = []
    for config in configs:
        for key in ('model', 'attention', 'dataset', 'seeds', 'output_dir'):
            value = getattr(args, key, None)
            if value is not None:
                config[key] = value
        resolved_configs.extend(resolve_experiments(apply_overrides(config, args.set)))
    # A named backbone may already use an explicit ablation's value (e.g.
    # EEGEncoder defaults to K=3). Run each resolved configuration only once.
    resolved_configs = list({digest(config): config for config in resolved_configs}.values())
    if args.command == 'plan' or args.dry_run:
        print(json.dumps(resolved_configs if is_sweep or len(resolved_configs) > 1 else resolved_configs[0], indent=2))
        if args.command == 'plan':
            import shlex
            arguments = list(sys.argv[1:] if argv is None else argv)
            arguments[0] = 'sweep' if is_sweep else 'run'
            print('\nLaunch on the experiment machine: python main.py ' + shlex.join(arguments))
        return
    from .engine import run_experiment
    for config in resolved_configs:
        print(f"Launching {config['dataset']}/{config['model']} subject={config.get('subject_id') or 'cohort'} attention={config['attention']} ({config['model_variant']}) "
              f"graph={'electrodes' if config['model_variant'] == 'eeg' else 'time'} "
              f"seeds={config['seeds']} epochs={config['training']['epochs']} device={config['device']}")
        run_experiment(config, skip_completed=args.skip_completed)
    _finish_training_reports(resolved_configs, automatic=args.report)


if __name__ == '__main__':
    main()
