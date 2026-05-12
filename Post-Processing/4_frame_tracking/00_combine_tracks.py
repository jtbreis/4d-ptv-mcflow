"""
Combine per-run tracks.h5part into one Parquet table.

One row per particle per Step# frame. Values and HDF5 attributes are copied as-is;
no Track objects or recomputed kinematics.
"""

import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

base_folder = 'raw_data/low_threshold/PTV_below'
case = 'TTI_aligned_with_gravity'

runs = ['Run1', 'Run2']

# Parallel over runs (separate files; I/O-bound reads). Set to 1 to disable.
MAX_RUN_WORKERS = min(8, (os.cpu_count() or 1) * 2)

# Rows per Parquet write batch (progress updates between chunks).
WRITE_CHUNK_ROWS = 256_000

# While reading h5part, merge accumulated column chunks this often to limit
# np.concatenate/vstack fan-in (many tiny steps -> fewer larger blocks).
READ_COLLAPSE_EVERY = 256


def _attr_to_python(v):
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='replace')
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _step_frame(key):
    m = re.match(r'Step#(\d+)', key)
    return int(m.group(1)) if m else -1


def count_step_keys(path):
    if not os.path.isfile(path):
        return 0
    with h5py.File(path, 'r') as f:
        return sum(1 for k in f.keys() if k.startswith('Step#'))


def _collapse_column_chunks(chunks):
    """Merge chunk list in place to a single array (1d concat or 2d vstack)."""
    if len(chunks) < 2:
        return
    if chunks[0].ndim == 1:
        merged = np.concatenate(chunks)
    else:
        merged = np.vstack(chunks)
    chunks.clear()
    chunks.append(merged)


def _collapse_all_chunks(col_chunks):
    for chunks in col_chunks.values():
        if len(chunks) > 1:
            _collapse_column_chunks(chunks)


def _finalize_h5part_columns(col_chunks):
    """Build dict suitable for pd.DataFrame (2d data -> one .tolist() per column)."""
    out = {}
    for key, chunks in col_chunks.items():
        if not chunks:
            continue
        if len(chunks) > 1:
            _collapse_column_chunks(chunks)
        merged = chunks[0]
        if merged.ndim == 1:
            out[key] = merged
        else:
            out[key] = merged.tolist()
    return out


def dataframe_from_h5part(path, run_label, run_number, pbar=None):
    """Load one tracks.h5part into a DataFrame (all steps).

    Fast path: partial HDF5 reads ([:n] only), numpy chunk lists, one concat per
    column instead of one DataFrame per step + pd.concat.
    """
    if not os.path.isfile(path):
        if tqdm is not None and pbar is not None:
            tqdm.write(f'Warning: missing {path}, skipping')
        else:
            print(f'Warning: missing {path}, skipping')
        return pd.DataFrame()

    col_chunks = defaultdict(list)
    step_index = 0

    with h5py.File(path, 'r') as f:
        root_meta = {f'h5_{k}': _attr_to_python(f.attrs[k]) for k in f.attrs}

        step_keys = sorted(
            (k for k in f.keys() if k.startswith('Step#')),
            key=_step_frame,
        )
        for sk in step_keys:
            grp = f[sk]
            frame = _step_frame(sk)
            step_meta = {
                f'step_{k}': _attr_to_python(grp.attrs[k])
                for k in grp.attrs
            }

            dsnames = sorted(
                k for k in grp.keys() if isinstance(grp[k], h5py.Dataset)
            )
            if not dsnames:
                if pbar is not None:
                    pbar.update(1)
                continue

            ds_by_name = {name: grp[name] for name in dsnames}
            lens = []
            for name in dsnames:
                ds = ds_by_name[name]
                if ds.ndim >= 1:
                    lens.append(ds.shape[0])
            if not lens:
                if pbar is not None:
                    pbar.update(1)
                continue
            n = min(lens)
            if max(lens) != n:
                msg = (
                    f'Warning: ragged step {sk} in {path}: row lengths {lens}, using n={n}'
                )
                if tqdm is not None and pbar is not None:
                    tqdm.write(msg)
                else:
                    print(msg)

            dattr = {}
            for name in dsnames:
                ds = ds_by_name[name]
                for ak, av in ds.attrs.items():
                    dattr[f'{name}_attr_{ak}'] = _attr_to_python(av)

            col_chunks['run'].append(np.full(n, run_label, dtype=object))
            col_chunks['run_number'].append(
                np.full(n, run_number, dtype=np.int32))
            col_chunks['frame'].append(np.full(n, frame, dtype=np.int32))
            for k, v in root_meta.items():
                col_chunks[k].append(np.full(n, v, dtype=object))
            for k, v in step_meta.items():
                col_chunks[k].append(np.full(n, v, dtype=object))
            for k, v in dattr.items():
                col_chunks[k].append(np.full(n, v, dtype=object))

            for name in dsnames:
                ds = ds_by_name[name]
                if ds.ndim < 1 or ds.shape[0] < n:
                    continue
                sl = np.asarray(ds[:n])
                if sl.ndim == 1:
                    col_chunks[name].append(sl)
                else:
                    col_chunks[name].append(sl)

            step_index += 1
            if (
                READ_COLLAPSE_EVERY > 0
                and step_index % READ_COLLAPSE_EVERY == 0
            ):
                _collapse_all_chunks(col_chunks)

            if pbar is not None:
                pbar.update(1)

    if not col_chunks.get('frame'):
        return pd.DataFrame()

    _collapse_all_chunks(col_chunks)
    return pd.DataFrame(_finalize_h5part_columns(col_chunks))


def write_parquet_chunked(dfs_by_run, parquet_path):
    """
    Write one Parquet file in row chunks with tqdm (rows + current run in postfix).
    dfs_by_run: list of (run_label, DataFrame) in output order.
    """
    pairs = [(r, d) for r, d in dfs_by_run if not d.empty]
    if not pairs:
        return

    if pa is None or pq is None:
        df = pd.concat([d for _, d in pairs], ignore_index=True, copy=False)
        df.to_parquet(parquet_path, index=False)
        print(
            f'Wrote {len(df)} rows (pyarrow not installed; no write progress bar)')
        return

    tables = [
        pa.Table.from_pandas(d, preserve_index=False) for _, d in pairs
    ]
    try:
        schema = pa.unify_schemas([t.schema for t in tables])
    except Exception as e:
        msg = f'Parquet schema unify failed ({e}); falling back to single write.'
        if tqdm is not None:
            tqdm.write(msg)
        else:
            print(msg)
        df = pd.concat([d for _, d in pairs], ignore_index=True, copy=False)
        df.to_parquet(parquet_path, index=False)
        return

    total_rows = sum(len(d) for _, d in pairs)
    writer = None
    try:
        wbar = (
            tqdm(
                total=total_rows,
                desc='Writing parquet',
                unit='row',
                smoothing=0.05,
            )
            if tqdm is not None
            else None
        )
        for run_label, df_run in pairs:
            if wbar is not None:
                wbar.set_postfix_str(run_label, refresh=False)
            nrun = len(df_run)
            for start in range(0, nrun, WRITE_CHUNK_ROWS):
                sub = df_run.iloc[start: start + WRITE_CHUNK_ROWS]
                tbl = pa.Table.from_pandas(
                    sub, preserve_index=False).cast(schema)
                if writer is None:
                    writer = pq.ParquetWriter(
                        parquet_path,
                        schema,
                        compression='snappy',
                    )
                writer.write_table(tbl)
                if wbar is not None:
                    wbar.update(len(sub))
        if wbar is not None:
            wbar.close()
    finally:
        if writer is not None:
            writer.close()


def main():
    paths_meta = [
        (os.path.join(base_folder, case, run, 'tracks.h5part'), run)
        for run in runs
    ]
    total_steps = sum(count_step_keys(p) for p, _ in paths_meta)

    results = {}
    n_workers = min(MAX_RUN_WORKERS, len(runs))
    use_parallel = n_workers > 1 and len(runs) > 1

    if use_parallel:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {}
            for run in runs:
                run_dir = os.path.join(base_folder, case, run)
                h5part_path = os.path.join(run_dir, 'tracks.h5part')
                run_number = int(run.replace('Run', ''))
                fut = ex.submit(
                    dataframe_from_h5part,
                    h5part_path,
                    run,
                    run_number,
                    None,
                )
                futures[fut] = run
            run_iter = as_completed(futures)
            if tqdm is not None:
                run_iter = tqdm(
                    run_iter,
                    total=len(futures),
                    desc='Loading runs (h5part)',
                    unit='file',
                )
            for fut in run_iter:
                run = futures[fut]
                results[run] = fut.result()
    else:
        pbar = None
        if tqdm is not None and total_steps > 0:
            pbar = tqdm(
                total=total_steps,
                desc='Steps (all runs)',
                unit='step',
                smoothing=0.05,
            )
        for path, run in paths_meta:
            run_number = int(run.replace('Run', ''))
            results[run] = dataframe_from_h5part(
                path, run, run_number, pbar=pbar
            )
        if pbar is not None:
            pbar.close()

    dfs_by_run = [(run, results[run]) for run in runs]
    if all(d.empty for _, d in dfs_by_run):
        print('No rows collected; not writing Parquet.')
        return

    parquet_path = os.path.join(base_folder, case, f'{case}_tracks.parquet')
    write_parquet_chunked(dfs_by_run, parquet_path)
    n_out = sum(len(d) for _, d in dfs_by_run if not d.empty)
    print(f'Wrote {n_out} rows to {parquet_path}')


if __name__ == '__main__':
    main()
