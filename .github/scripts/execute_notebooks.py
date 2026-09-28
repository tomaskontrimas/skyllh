"""Executes Jupyter notebooks and updates their stored outputs.

A notebook file is only rewritten, when its outputs changed in a meaningful
way, i.e. differences that are only due to volatile content like timestamps,
run times, progress bars, or memory addresses are ignored. This avoids
proposing changes to the notebooks when nothing but the execution time changed.

Notebooks that fail to execute are left unchanged and are reported.

Usage::

    python execute_notebooks.py [--timeout SECONDS] [--summary FILE] NOTEBOOK [NOTEBOOK ...]

The exit code is 1, if at least one notebook failed to execute, 0 otherwise.
"""

import argparse
import base64
import copy
import hashlib
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbformat.v4.rwbase import split_lines

# The home directory of the user executing the notebooks. It is replaced by "~"
# in the outputs, e.g. in paths of the data repository.
_HOME = str(Path.home())

# Regular expressions matching volatile content of text outputs, together with
# their replacements.
_VOLATILE_PATTERNS = [
    # Timestamps, e.g. of log messages.
    (re.compile(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?'), '<timestamp>'),
    # Run times, e.g. "1.2 sec", "3.4e-05 sec/iter", "12.3s", "[00:18<00:00, 54.47it/s]".
    (re.compile(r'\d+(?:\.\d+)?(?:e[-+]?\d+)?\s*(?:sec|s|ms|us)\b(?:/iter)?'), '<time>'),
    (re.compile(r'\d+(?:\.\d+)?\s*(?:it/s|s/it)'), '<rate>'),
    (re.compile(r'\[\d{2}:\d{2}(?::\d{2})?<[^\]]*\]'), '[<eta>]'),
    # Memory addresses, e.g. "<object at 0x7f0bd7017ec0>".
    (re.compile(r'0x[0-9a-fA-F]+'), '0x<address>'),
    # Process IDs and names of worker processes in log messages.
    (re.compile(r'\b(?:pid|PID)[ =]\d+'), 'pid=<pid>'),
    (re.compile(r'(?:Fork|Spawn)Process-\d+(?::\d+)*'), '<process>'),
]

# Progress bar lines, e.g. of tqdm.
_PROGRESS_BAR_RE = re.compile(r'^\s*\d+%\|.*\|.*$', re.MULTILINE)


def sanitize_text(text: str) -> str:
    """Replaces the home directory of the user by "~" in the given text."""
    return text.replace(_HOME, '~')


def normalize_text(text: str) -> str:
    """Removes volatile content from the given text output."""
    text = sanitize_text(text)
    text = _PROGRESS_BAR_RE.sub('', text)
    for pattern, replacement in _VOLATILE_PATTERNS:
        text = pattern.sub(replacement, text)
    # Carriage returns are used by progress bars to overwrite lines.
    text = '\n'.join(line.rsplit('\r', 1)[-1] for line in text.split('\n'))
    return '\n'.join(line for line in text.split('\n') if line.strip())


def normalize_image(data: str) -> str:
    """Creates a representation of the given base64 encoded image that depends
    only on its pixels, i.e. not on metadata like the matplotlib version, which
    is embedded in PNG images.
    """
    try:
        from PIL import Image

        with Image.open(io.BytesIO(base64.b64decode(data))) as img:
            pixels = img.convert('RGBA').tobytes()
            return f'{img.size}:{hashlib.sha256(pixels).hexdigest()}'
    except (ImportError, OSError, ValueError):
        # Pillow is not available or the data is not a decodable image.
        return data


def normalize_notebook(nb: nbformat.NotebookNode) -> list:
    """Creates a representation of the given notebook that is independent of
    volatile content, for comparing the outputs of two executions.
    """
    cells = []
    for cell in nb.cells:
        outputs = []
        for output in cell.get('outputs', []):
            output_type = output.get('output_type')
            if output_type == 'stream':
                text = normalize_text(output.get('text', ''))
                if text:
                    outputs.append(('stream', output.get('name'), text))
            elif output_type in ('execute_result', 'display_data'):
                data = {}
                for mime, value in output.get('data', {}).items():
                    if mime.startswith('text/') and isinstance(value, str):
                        value = normalize_text(value)
                    elif mime.startswith('image/') and mime != 'image/svg+xml':
                        value = normalize_image(value)
                    data[mime] = value
                outputs.append((output_type, sorted(data.items())))
            elif output_type == 'error':
                outputs.append(('error', output.get('ename'), normalize_text(output.get('evalue', ''))))
        # Merge the texts of consecutive stream outputs of the same name, because
        # the splitting of streams into outputs depends on the timing.
        merged = []
        for output in outputs:
            if merged and output[0] == 'stream' and merged[-1][0] == 'stream' and merged[-1][1] == output[1]:
                merged[-1] = ('stream', output[1], merged[-1][2] + '\n' + output[2])
            else:
                merged.append(output)
        cells.append((cell.cell_type, cell.source, merged))
    return cells


def sanitize_outputs(outputs: list) -> list:
    """Replaces the home directory of the user by "~" in the given text
    outputs.
    """
    for output in outputs:
        if 'text' in output:
            output['text'] = [sanitize_text(line) for line in output['text']]
        for mime, value in output.get('data', {}).items():
            if mime.startswith('text/') and isinstance(value, list):
                output['data'][mime] = [sanitize_text(line) for line in value]
    return outputs


def write_outputs(path: Path, original_text: str, nb: nbformat.NotebookNode) -> None:
    """Writes the outputs and execution counts of the code cells of the executed
    notebook into the notebook file. Everything else, e.g. the cell sources and
    the notebook metadata, as well as the JSON formatting of the file, is kept
    as is to keep the changes minimal.
    """
    raw = json.loads(original_text)
    executed = split_lines(copy.deepcopy(nb))
    if len(raw['cells']) != len(executed.cells):
        raise RuntimeError(f'The number of cells of {path} changed during the execution!')

    for raw_cell, cell in zip(raw['cells'], executed.cells, strict=True):
        if raw_cell['cell_type'] != 'code':
            continue
        raw_cell['outputs'] = sanitize_outputs(json.loads(json.dumps(cell['outputs'])))
        raw_cell['execution_count'] = cell['execution_count']

    # Use the JSON formatting of Jupyter and keep the original file ending.
    text = json.dumps(raw, indent=1, sort_keys=True, ensure_ascii=False)
    if original_text.endswith('\n'):
        text += '\n'
    path.write_text(text)


def execute_notebook(path: Path, timeout: int, kernel_name: str) -> tuple[str, str | None, float]:
    """Executes the given notebook and updates it in-place, if its outputs
    changed in a meaningful way.

    Returns
    -------
    status
        One of ``'updated'``, ``'unchanged'``, or ``'failed'``.
    error
        The error message, if the execution failed, otherwise ``None``.
    duration
        The execution time in seconds.
    """
    original_text = path.read_text()
    original_nb = nbformat.reads(original_text, as_version=4)
    nb = copy.deepcopy(original_nb)

    client = NotebookClient(
        nb,
        timeout=timeout if timeout > 0 else None,
        kernel_name=kernel_name,
        resources={'metadata': {'path': str(path.parent)}},
    )

    t_start = time.monotonic()
    try:
        client.execute()
    except CellExecutionError as exc:
        return ('failed', str(exc), time.monotonic() - t_start)
    except Exception:  # noqa: BLE001 - Report any error, e.g. of the kernel, as failure.
        return ('failed', traceback.format_exc(), time.monotonic() - t_start)
    duration = time.monotonic() - t_start

    if normalize_notebook(nb) == normalize_notebook(original_nb):
        return ('unchanged', None, duration)

    write_outputs(path, original_text, nb)

    return ('updated', None, duration)


def create_summary(results: dict) -> str:
    """Creates a markdown summary of the execution results."""
    lines = ['| Notebook | Status | Run time |', '| --- | --- | --- |']
    icons = {'updated': ':arrows_counterclockwise: updated', 'unchanged': ':white_check_mark: unchanged'}
    for path, (status, _, duration) in results.items():
        lines.append(f'| `{path}` | {icons.get(status, ":x: failed")} | {duration / 60:.1f} min |')

    failed = {path: error for (path, (status, error, _)) in results.items() if status == 'failed'}
    if failed:
        lines += ['', '### Failed notebooks', '']
        for path, error in failed.items():
            # Keep the end of the error message, which contains the exception.
            error = (error or '').strip()
            if len(error) > 3000:
                error = '...\n' + error[-3000:]
            lines += [f'<details><summary><code>{path}</code></summary>', '', '```', error, '```', '', '</details>']

    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('notebooks', nargs='+', type=Path, help='The notebooks to execute.')
    parser.add_argument(
        '--timeout', type=int, default=-1, help='The timeout per cell in seconds. Default is no timeout.'
    )
    parser.add_argument('--kernel', default='python3', help='The name of the Jupyter kernel to use.')
    parser.add_argument('--summary', type=Path, help='The file to write the markdown summary to.')
    args = parser.parse_args()

    results = {}
    for path in args.notebooks:
        print(f'Executing {path} ...', flush=True)
        results[path] = execute_notebook(path, timeout=args.timeout, kernel_name=args.kernel)
        (status, error, duration) = results[path]
        print(f'  {status} ({duration / 60:.1f} min)', flush=True)
        if error is not None:
            print(error, file=sys.stderr, flush=True)

    summary = create_summary(results)
    print(summary)
    if args.summary is not None:
        args.summary.write_text(summary)

    return 1 if any(status == 'failed' for (status, _, _) in results.values()) else 0


if __name__ == '__main__':
    sys.exit(main())
