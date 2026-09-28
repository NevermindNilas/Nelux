"""Matched sequential benchmarks; setup and correctness checks are outside timing.

Baseline/candidate samples run in interleaved order without overlapping children.
Reader rewind and internal indexed-batch decoder setup remain measured API work.
Encoder throughput includes draining/close, using preallocated synthetic inputs.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time

CASES = ('cpu', 'cpu-auto', 'numpy', 'cpu-batch', 'nvdec', 'nvdec-async',
         'nvdec-model', 'nvdec-owned', 'nvdec-owned-model', 'nvdec-batch',
         'encode-software', 'encode-nvenc')

def artifact_identity(source):
    artifacts = [path.resolve() for path in (source / 'nelux').glob('_nelux.*')
                 if path.suffix in ('.pyd', '.so')]
    if len(artifacts) != 1:
        raise RuntimeError(f'Expected exactly one native artifact in {source}, found {artifacts}')
    artifact = artifacts[0]
    return artifact, hashlib.sha256(artifact.read_bytes()).hexdigest()

def require_artifact_identity(source, expected):
    actual = artifact_identity(source)
    if actual != expected:
        raise RuntimeError(f'Native artifact changed during benchmark: expected {expected}, found {actual}')
    return actual

def process_peak_rss():
    # Lifetime peak includes imports and untimed validation; label it clearly
    # instead of implying this is an incremental sustained-workload peak.
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == 'darwin' else peak * 1024

def digest(frame):
    import numpy as np
    array = frame if isinstance(frame, np.ndarray) else frame.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()

def cli(args, source, name):
    import shutil
    executable = name + ('.exe' if os.name == 'nt' else '')
    directory = args.ffmpeg_bin or source / 'external/ffmpeg/bin'
    candidate = directory / executable
    found = str(candidate.resolve()) if candidate.is_file() else shutil.which(executable)
    if not found:
        raise RuntimeError(f'Independent {name} CLI required')
    return found

def verify_encode(args, source, path, count, width, height):
    import numpy as np
    result = subprocess.run([cli(args, source, 'ffprobe'), '-v', 'error', '-select_streams', 'v:0',
                            '-count_packets', '-show_entries', 'stream=nb_read_packets', '-of', 'json', str(path)],
                            capture_output=True, text=True, check=True, timeout=args.timeout or None)
    packets = int(json.loads(result.stdout)['streams'][0]['nb_read_packets'])
    if packets != count:
        raise RuntimeError(f'Encoder returned {packets} packets for {count} inputs')
    selected = sorted(set(range(min(16, count))) | set(range(max(0, count - 16), count)) | {count // 2})
    expression = '+'.join(f'eq(n,{index})' for index in selected).replace(',', '\\,')
    result = subprocess.run([cli(args, source, 'ffmpeg'), '-v', 'error', '-i', str(path),
                             '-vf', 'select=' + expression, '-fps_mode', 'passthrough',
                             '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                            capture_output=True, check=True, timeout=args.timeout or None)
    if len(result.stdout) != len(selected) * width * height * 3:
        raise RuntimeError('Independent encoder sample count differs')
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(len(selected), height, width, 3)
    errors = [int(np.abs(frame.astype(np.int16) - (24 + index % 16 * 12)).max())
              for frame, index in zip(pixels, selected)]
    if max(errors) > 3:
        raise RuntimeError(f'Encoded identity/order differs: maximum pixel error {max(errors)}')
    return dict(method='independent ffprobe packet count and FFmpeg solid samples', packets=packets,
                sampled_ordinals=selected, max_pixel_error=max(errors), tolerance=3, all_pixels_checked=False)

def child(args):
    source = args.source.resolve()
    artifact, artifact_sha = artifact_identity(source)
    if args.artifact_sha256 and artifact_sha != args.artifact_sha256:
        raise RuntimeError(f'Native artifact changed before child import: {artifact_sha} != {args.artifact_sha256}')
    sys.path.insert(0, str(source))
    ffbin = source / 'external/ffmpeg/bin'
    dll_handles = [os.add_dll_directory(str(ffbin))] if hasattr(os, 'add_dll_directory') and ffbin.is_dir() else []
    import torch
    import nelux
    assert Path(nelux.__file__).resolve().parent == source / 'nelux'
    if Path(nelux._nelux.__file__).resolve() != artifact:
        raise RuntimeError('Imported native artifact differs from the frozen source artifact')
    require_artifact_identity(source, (artifact, artifact_sha))
    encoding = args.case.startswith('encode-')
    batching = args.case.endswith('-batch')
    cuda = args.case.startswith('nvdec') or args.case == 'encode-nvenc'
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError('GPU benchmark requires a real CUDA device')
        torch.cuda.set_device(args.device)
    stream = torch.cuda.Stream(device=args.device) if cuda and args.nondefault_stream else None
    scope = lambda: torch.cuda.stream(stream) if stream else nullcontext()
    kwargs = dict(num_threads=1, convert_workers=0, prefetch=False)
    if args.case == 'cpu-auto':
        kwargs = dict(prefetch=False)
    elif args.case == 'numpy':
        kwargs['backend'] = 'numpy'
    elif cuda and not encoding:
        kwargs.update(decode_accelerator='nvdec', cuda_device_index=args.device,
                      copy_frames='owned' in args.case)
        if args.case == 'nvdec-async':
            kwargs['async_frames'] = True
    verification = {}
    setup = time.perf_counter()
    with scope(), nelux.VideoReader(str(args.video), **kwargs) as reader:
        width, height = reader.width, reader.height
        started = time.perf_counter()
        expected_count = reader.frame_count
        verification['index_setup_seconds'] = time.perf_counter() - started
        if expected_count <= 0:
            raise RuntimeError('Fixture has no frames')
        if not encoding:
            selected = sorted({0, min(5, expected_count - 1), expected_count // 2, expected_count - 1})
            if batching:
                batch = reader.get_batch(selected + [selected[0]])
                assert batch.shape[0] == len(selected) + 1
                pixels = [digest(batch[index]) for index in range(len(selected))]
                assert digest(batch[-1]) == pixels[0]
                del batch
            else:
                pixels, observed = [], 0
                for frame in reader:
                    if observed in selected:
                        pixels.append(digest(frame))
                    observed += 1
                assert observed == expected_count, (observed, expected_count)
                del frame
            verification.update(method='matched baseline/candidate digests outside timing',
                                frame_count=expected_count, ordinals=selected, pixel_sha256=pixels)
    verification['total_setup_seconds'] = time.perf_counter() - setup
    if batching:
        indices = ([index % expected_count for index in range(args.batch_size)] if args.batch_pattern == 'contiguous'
                   else [index * (expected_count - 1) // max(1, args.batch_size - 1) for index in range(args.batch_size)])
    if encoding:
        codec = 'h264_nvenc' if cuda else 'libx264'
        if codec not in {encoder['name'] for encoder in nelux.get_available_encoders()}:
            raise RuntimeError(f'Required encoder {codec} unavailable')
        kwargs = dict(codec=codec, width=width, height=height, fps=30.0,
                      pixel_format='nv12' if cuda else 'yuv420p', cq=1 if cuda else 18, options={'bf': '0'})
        if not cuda:
            kwargs['options'].update(preset='ultrafast', threads='1')
        dtype = torch.float32 if args.encoder_dtype == 'float32' else torch.uint8
        with scope():
            inputs = [torch.full((height, width, 3), value / 255.0 if dtype == torch.float32 else value,
                                 dtype=dtype, device=f'cuda:{args.device}' if cuda else 'cpu')
                      for value in range(24, 24 + 16 * 12, 12)]
        if cuda:
            torch.cuda.synchronize(args.device)
    samples, elapsed_samples, counts, construction, indexing, gpu_peak = [], [], [], [], [], []
    with tempfile.TemporaryDirectory(prefix='nelux-parity-encode-') as temporary:
        for repetition in range(args.rounds + args.warmups):
            if cuda:
                torch.cuda.synchronize(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
            with scope():
                started = time.perf_counter()
                output = Path(temporary) / f'sample-{repetition}.mkv'
                instance = (nelux.VideoEncoder(str(output), **kwargs) if encoding else
                            nelux.VideoReader(str(args.video), **kwargs))
                construction.append(time.perf_counter() - started)
                started = time.perf_counter()
                if batching:
                    assert instance.frame_count == expected_count
                indexing.append(time.perf_counter() - started if batching else 0.0)
                frames, calls = 0, 0
                if cuda:
                    torch.cuda.synchronize(args.device)
                started = time.perf_counter()
                try:
                    while time.perf_counter() - started < args.seconds or calls == 0:
                        if encoding:
                            for value in inputs:
                                instance.encode_frame(value)
                                frames += 1
                        elif batching:
                            batch = instance.get_batch(indices)
                            frames += batch.shape[0]
                            del batch
                        else:
                            delivered = 0
                            for frame in instance:
                                frames += 1
                                delivered += 1
                                if args.case.endswith('model'):
                                    work = frame.float().mean()
                            assert delivered == expected_count, (delivered, expected_count)
                            del frame
                        calls += 1
                    if encoding:
                        instance.close()  # Include queued work and codec flush.
                    if cuda:
                        torch.cuda.synchronize(args.device)
                    elapsed = time.perf_counter() - started
                finally:
                    instance.close()
                if frames <= 0:
                    raise RuntimeError('Benchmark returned no frames')
                if encoding:
                    checked = verify_encode(args, source, output, frames, width, height)
                    if repetition >= args.warmups:
                        verification.setdefault('encodes', []).append(checked)
                if repetition >= args.warmups:
                    samples.append(frames / elapsed)
                    elapsed_samples.append(elapsed)
                    counts.append(frames)
                    if cuda:
                        gpu_peak.append(torch.cuda.max_memory_allocated(args.device))
    ownership = ('retained-synthetic-inputs' if encoding else 'owned-discarded' if batching or not cuda or
                 'owned' in args.case else 'borrowed-discarded')
    reported_kwargs = dict(kwargs)
    if batching:
        reported_kwargs['benchmark_indices'] = indices
    require_artifact_identity(source, (artifact, artifact_sha))
    print(json.dumps(dict(source=str(source), artifact=str(artifact), sha256=artifact_sha,
                          torch=torch.__version__, python=sys.version, ffmpeg=nelux.__ffmpeg_version__, case=args.case,
                          video=str(args.video.resolve()), video_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
                          kwargs=reported_kwargs, samples_fps=samples, median_fps=statistics.median(samples),
                          cv=statistics.pstdev(samples) / statistics.mean(samples), elapsed_seconds=elapsed_samples,
                          frames_per_sample=counts, mean_frame_seconds=[1 / fps for fps in samples],
                          construction_seconds=construction[args.warmups:], indexing_seconds=indexing[args.warmups:],
                          gpu_peak_bytes=gpu_peak, ownership=ownership, verification=verification,
                          process_lifetime_peak_rss_bytes=process_peak_rss(),
                          stream='nondefault' if stream else 'default', device=args.device if cuda else None,
                          gpu_name=torch.cuda.get_device_name(args.device) if cuda else None,
                          torch_cuda_version=torch.version.cuda,
                          encoder_dtype=args.encoder_dtype if encoding else None,
                          timing_contract=('constructor excluded; accepted inputs through completed close included' if encoding else
                                           'reader/index setup excluded; internal batch decoder setup included' if batching else
                                           'reader construction excluded; repeated complete clip iteration including rewind included'))))

def terminate_child(process):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()

def run_child(command, cwd, timeout):
    options = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
               if os.name == 'nt' else {'start_new_session': True})
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **options)
    try:
        stdout, stderr = process.communicate(timeout=timeout or None)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        terminate_child(process)
        raise
    if process.returncode:
        raise RuntimeError(f'Benchmark child failed/crashed ({process.returncode})\n{stdout}\n{stderr}')
    return json.loads(stdout.strip().splitlines()[-1])

def bootstrap_median_interval(values):
    # Paired baseline/candidate rounds share nearby machine conditions. Report
    # this interval separately from the unpaired CV and observed median ratio.
    rng = random.Random(212)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(20000))
    return [medians[499], medians[19499]]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ffmpeg-bin', type=Path)
    parser.add_argument('--cases', nargs='+', choices=CASES,
                        default=['cpu', 'cpu-auto', 'numpy', 'nvdec', 'nvdec-async', 'nvdec-model'])
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=0.5)
    parser.add_argument('--timeout', type=float, default=0, help='Per-child seconds; 0 disables timeout')
    parser.add_argument('--stop-file', type=Path, default=Path('.benchmark-stop'),
                        help='Consume this marker and stop before another child loads a native artifact')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-pattern', choices=('contiguous', 'sparse'), default='contiguous')
    parser.add_argument('--encoder-dtype', choices=('uint8', 'float32'), default='uint8')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--nondefault-stream', action='store_true')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--artifact-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stop_file.exists():
        args.stop_file.unlink()
        parser.error('Benchmark stop requested before starting another child')
    if args.rounds <= 0 or args.warmups < 0 or args.seconds <= 0 or args.batch_size <= 0 or args.timeout < 0:
        parser.error('rounds/seconds/batch-size must be positive; warmups/timeout must be nonnegative')
    if args.source:
        if not args.case:
            parser.error('child requires --case')
        child(args)
        return
    if not (args.baseline and args.candidate and args.output):
        parser.error('parent requires --baseline, --candidate and --output')
    args.baseline, args.candidate, args.output = args.baseline.resolve(), args.candidate.resolve(), args.output.resolve()
    if args.baseline == args.candidate:
        parser.error('Baseline and candidate must use different source directories')
    frozen_artifacts = {name: artifact_identity(root) for name, root in
                        (('baseline', args.baseline), ('candidate', args.candidate))}
    if frozen_artifacts['baseline'][0] == frozen_artifacts['candidate'][0]:
        parser.error('Baseline and candidate must use different native artifact paths')
    if len(set(args.cases)) != len(args.cases):
        parser.error('Each benchmark case must be requested only once')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng, jobs, records = random.Random(212), [], []
    for case in args.cases:
        for repetition in range(args.rounds):
            pair = [(case, repetition, name, root) for name, root in
                    (('baseline', args.baseline), ('candidate', args.candidate))]
            rng.shuffle(pair)
            jobs.extend(pair)
    for case, repetition, name, root in jobs:
        require_artifact_identity(root, frozen_artifacts[name])
        command = [str(args.python), str(Path(__file__).resolve()), '--source', str(root), '--case', case,
                   '--video', str(args.video.resolve()), '--rounds', '1', '--warmups', str(args.warmups),
                   '--seconds', str(args.seconds), '--timeout', str(args.timeout), '--batch-size', str(args.batch_size),
                   '--batch-pattern', args.batch_pattern, '--encoder-dtype', args.encoder_dtype, '--device', str(args.device),
                   '--artifact-sha256', frozen_artifacts[name][1]]
        if args.ffmpeg_bin:
            command += ['--ffmpeg-bin', str(args.ffmpeg_bin.resolve())]
        if args.nondefault_stream:
            command.append('--nondefault-stream')
        record = run_child(command, args.output.parent, args.timeout)
        require_artifact_identity(root, frozen_artifacts[name])
        if (Path(record['artifact']).resolve(), record['sha256']) != frozen_artifacts[name]:
            raise RuntimeError(f'Child artifact identity differs from frozen {name} identity')
        record.update(route=name, repetition=repetition)
        records.append(record)
        print(f"{case}/{name}/{repetition}: {record['median_fps']:.1f} FPS", flush=True)
        args.output.write_text(json.dumps(dict(measurements=records, complete=False), indent=2))
    comparisons = []
    for case in args.cases:
        old = [record for record in records if record['case'] == case and record['route'] == 'baseline']
        new = [record for record in records if record['case'] == case and record['route'] == 'candidate']
        for record in old + new:
            for key in ('torch', 'python', 'ffmpeg', 'video_sha256', 'kwargs', 'ownership', 'stream',
                        'encoder_dtype', 'device', 'gpu_name', 'torch_cuda_version'):
                if record[key] != old[0][key]:
                    raise RuntimeError(f'Mismatched {key} invalidates {case} comparison')
            if not case.startswith('encode-'):
                for key in ('frame_count', 'ordinals', 'pixel_sha256'):
                    if record['verification'][key] != old[0]['verification'][key]:
                        raise RuntimeError(f'Output parity failed for {case}: {key}')
        old_fps, new_fps = [record['median_fps'] for record in old], [record['median_fps'] for record in new]
        old_median, new_median = statistics.median(old_fps), statistics.median(new_fps)
        old_cv, new_cv = statistics.pstdev(old_fps) / statistics.mean(old_fps), statistics.pstdev(new_fps) / statistics.mean(new_fps)
        paired_ratios = [next(record['median_fps'] for record in new if record['repetition'] == repetition) /
                         next(record['median_fps'] for record in old if record['repetition'] == repetition)
                         for repetition in range(args.rounds)]
        interval = bootstrap_median_interval(paired_ratios)
        comparisons.append(dict(case=case, baseline_median_fps=old_median, candidate_median_fps=new_median,
                                ratio=new_median / old_median, baseline_cv=old_cv, candidate_cv=new_cv,
                                paired_ratios=paired_ratios, paired_median_ratio=statistics.median(paired_ratios),
                                paired_median_ratio_ci95=interval, paired_ci_method='20000 paired median bootstrap resamples',
                                budget_supported_by_paired_ci=len(paired_ratios) >= 7 and interval[0] >= 0.99,
                                within_one_percent=new_median >= old_median * 0.99,
                                noise_below_budget=len(paired_ratios) >= 7 and max(old_cv, new_cv) < 0.01,
                                repetition_count=len(paired_ratios), output_parity_passed=True))
    args.output.write_text(json.dumps(dict(measurements=records, comparisons=comparisons, complete=True,
                                          sequential_interleaved=True), indent=2))
    print(json.dumps(comparisons, indent=2))

if __name__ == '__main__':
    main()
