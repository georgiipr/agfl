"""Resolve saved configurations or presets and delegate to model experiments."""
import argparse
import json
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


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reproducible EEG/ECG AGFL experiments')
    commands = parser.add_subparsers(dest='command')
    run = commands.add_parser('run', help='Launch one configuration; matching seed runs are overwritten')
    _configuration_arguments(run)
    run.add_argument('--dataset')
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--skip-completed', action='store_true', help='Keep matching completed seeds; restart incomplete seeds')
    sweep = commands.add_parser('sweep', help='Launch a comparison/ablation matrix; matching seed runs are overwritten')
    _configuration_arguments(sweep)
    sweep.add_argument('--dry-run', action='store_true')
    sweep.add_argument('--skip-completed', action='store_true', help='Keep matching completed seeds; restart incomplete seeds')
    plan = commands.add_parser('plan', help='Show resolved settings and launch command without loading data')
    _configuration_arguments(plan)
    analysis = commands.add_parser('analyze')
    analysis.add_argument('results')
    analysis.add_argument('--output-dir', default='analysis')
    analysis.add_argument('--plots', action='store_true', help='Also write PNG/PDF figures from saved results')
    analysis.add_argument('--diagnostics-root', help='Include sparsity/entropy versus accuracy plots from saved checkpoint diagnostics; requires --plots')
    diagnostics = commands.add_parser('diagnose', help='Plot signals, embeddings and mixer maps from a saved checkpoint')
    diagnostics.add_argument('run_dir', help='One seed directory containing config.json, split.json and checkpoint.pt')
    diagnostics.add_argument('--output-dir', required=True)
    diagnostics.add_argument('--data-dir', help='Relocated directory containing the original recordings')
    diagnostics.add_argument('--labels-dir', help='Relocated EEG E-session label directory')
    diagnostics.add_argument('--data-path', help='Relocated NPZ dataset file')
    diagnostics.add_argument('--device', default='cpu')
    diagnostics.add_argument('--partition', choices=('validation', 'test'), default='validation')
    diagnostics.add_argument('--max-samples', type=int, default=256)
    diagnostics.add_argument('--embedding', choices=('pca', 'tsne', 'umap'), default='tsne')
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
    if args.command == 'analyze':
        if args.diagnostics_root and not args.plots:
            parser.error('--diagnostics-root requires --plots')
        from .analysis import analyze_results
        result = analyze_results(args.results, args.output_dir)
        if args.plots:
            from pathlib import Path
            from .visualization.results import generate_result_plots
            generate_result_plots(result, Path(args.output_dir) / 'figures', diagnostics_root=args.diagnostics_root)
            print(f'Plot index: {args.output_dir}/figures/index.md')
        print(f'Result tables: {args.output_dir}/report.md')
        return
    if args.command == 'diagnose':
        from .visualization.diagnostics import diagnose_run
        diagnose_run(args.run_dir, args.output_dir, device=args.device, partition=args.partition,
                     max_samples=args.max_samples, embedding=args.embedding, data_dir=args.data_dir,
                     labels_dir=args.labels_dir, data_path=args.data_path)
        print(f'Diagnostic plot index: {args.output_dir}/index.md')
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
              f"seeds={config['seeds']} epochs={config['training']['epochs']} device={config['device']}")
        run_experiment(config, skip_completed=args.skip_completed)
    import shlex
    for root in sorted({config['output_dir'] for config in resolved_configs}):
        print('\nTraining finished. Generate tables and result plots on this machine:')
        print(f'python main.py analyze {shlex.quote(root)} --output-dir {shlex.quote(root + "/analysis")} --plots')
    print('For signal, embedding and attention plots, use diagnose; the README includes a loop for every subject and seed.')


if __name__ == '__main__':
    main()
