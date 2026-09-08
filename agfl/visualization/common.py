"""Headless, standalone PNG/PDF output with an explicit artifact manifest."""
from pathlib import Path
import re

from agfl.storage import write_json


def slug(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._') or 'figure'


class FigureWriter:
    def __init__(self, output_dir):
        try:
            import matplotlib
        except ImportError as error:
            raise RuntimeError("Plotting requires the 'plots' extra: install -e '.[plots]' on the target machine") from error
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        self.plt = plt
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.figures, self.skipped = [], []

    def save(self, figure, name, caption):
        stem = self.root / name
        stem.parent.mkdir(parents=True, exist_ok=True)
        files = []
        try:
            for extension in ('png', 'pdf'):
                path = stem.parent / (stem.name + '.' + extension)
                figure.savefig(path, dpi=200, bbox_inches='tight')
                files.append(path.relative_to(self.root).as_posix())
        finally:
            self.plt.close(figure)
        self.figures.append({'caption': caption, 'files': files})

    def skip(self, name, reason):
        self.skipped.append({'name': str(name), 'reason': str(reason)})

    def finish(self, metadata):
        manifest = {**metadata, 'figures': self.figures, 'skipped': self.skipped}
        write_json(self.root / 'manifest.json', manifest)
        lines = ['# Experiment figures', '', 'PNG previews and vector PDF exports.', '']
        for entry in self.figures:
            png, pdf = entry['files']
            lines += [entry['caption'], '', f'![Figure]({png})', '', f'[PDF]({pdf})', '']
        if self.skipped:
            lines += ['## Unavailable figures', '']
            lines += [f"- {entry['name']}: {entry['reason']}" for entry in self.skipped]
        (self.root / 'index.md').write_text('\n'.join(lines) + '\n')
        return manifest
