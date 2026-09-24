"""Inference overlap example: prefetch frames while inference runs.

Pattern: bounded prefetch queue -> per-frame decode -> BHWC batches ->
inference on a dedicated torch.cuda.Stream. NVDEC frames are cloned before
the reader reuses its output tensor; decoder-side resize is supported.

Run:
  python examples/inference_overlap.py <video> [--nvdec] [--resize 224 224]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def _add_ffmpeg_dll_dir() -> None:
    if os.name != "nt":
        return
    for key in ("NELUX_FFMPEG_DLL_DIR", "FFMPEG_DLL_DIR"):
        d = os.environ.get(key)
        if d:
            try:
                os.add_dll_directory(d)
            except Exception:
                pass


_add_ffmpeg_dll_dir()

from nelux import VideoReader  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", nargs="?",
                    default="tests/data/test_1080p.mp4")
    ap.add_argument("--nvdec", action="store_true",
                    help="NVDEC GPU decode (frames stay on GPU; clone, don't copy)")
    ap.add_argument("--resize", nargs=2, type=int, default=None, metavar=("W", "H"),
                    help="Decoder-side resize target, e.g. --resize 224 224")
    ap.add_argument("--prefetch", type=int, default=12,
                    help="Prefetch buffer 8-16 for typical ML pipelines")
    ap.add_argument("--batch", type=int, default=8,
                    help="number of prefetched frames per inference batch")
    ap.add_argument("--iters", type=int, default=50)
    return ap.parse_args()


@torch.no_grad()
def fake_infer(bhwc_fp32_cuda: torch.Tensor) -> torch.Tensor:
    # Stand-in for model(frame): channel mean. Keep on the infer stream.
    return bhwc_fp32_cuda.mean(dim=(1, 2, 3))


def main() -> None:
    args = parse_args()
    accel = "nvdec" if (args.nvdec and torch.cuda.is_available()) else "cpu"
    resize = tuple(args.resize) if args.resize else None

    reader = VideoReader(
        args.video,
        backend="pytorch",
        decode_accelerator=accel,
        resize=resize,  # decoder-side resize: no F.interpolate/cv2 pass after
        prefetch=True,
    )
    n_total = reader.get_frame_count()
    print(f"video={args.video} accel={accel} resize={resize} "
          f"frames={n_total} shape={reader.shape}")

    # 1) Bounded prefetch queue decouples decode from inference.
    # 8-16 covers typical ML pipelines; 32+ only for highly variable latency.
    buf = max(8, min(16, args.prefetch))
    reader.start_prefetch(buffer_size=buf)
    print(f"prefetch: buffer={buf} (bounded queue)")

    use_cuda = torch.cuda.is_available()
    infer_stream = torch.cuda.Stream() if use_cuda else None

    t0 = time.perf_counter()
    done = 0
    step = args.batch
    for start in range(0, min(args.iters * step, n_total), step):
        # 2) Consume the prefetch queue. get_batch() uses a separate random
        # access path and stops prefetch, so collect streaming frames here.
        frames = []
        for _ in range(min(step, n_total - start)):
            frame = reader.read_frame()
            if frame.numel() == 0:
                break
            if accel == "nvdec":
                # NVDEC reuses one output tensor. Finish this clone before the
                # next read can overwrite it; inference still runs separately.
                frame = frame.clone()
                torch.cuda.current_stream().synchronize()
            frames.append(frame)
        if not frames:
            break
        batch = torch.stack(frames)  # [B,H,W,C] uint8, CPU or CUDA
        # Hot path stays in tensors: NO .numpy() here (that would force a
        # D2H sync + copy every batch and serialize the pipeline).

        if accel == "nvdec":
            # 3) GPU-direct: frames are already private CUDA tensors.
            work = batch.float().div_(255.0)
            # HWC -> BHWC already; models usually want BCHW:
            work = work.permute(0, 3, 1, 2)
        else:
            work = batch.float().div_(255.0).permute(0, 3, 1, 2)
            if use_cuda:
                work = work.cuda(non_blocking=True)

        if use_cuda:
            assert infer_stream is not None
            # 4) Event handoff: decode records, inference waits — streams
            # overlap instead of serializing on the default stream.
            decode_event = torch.cuda.Event()
            decode_event.record()
            with torch.cuda.stream(infer_stream):
                infer_stream.wait_event(decode_event)
                out = fake_infer(
                    work.permute(0, 2, 3, 1) if work.dim() == 4 else work)
                done_event = torch.cuda.Event()
                done_event.record()
            # Non-blocking: decode of the next batch proceeds while this
            # inference runs. Sync once per print interval, not per batch.
            if (start // step) % 10 == 0:
                infer_stream.synchronize()
                print(f"  batch {start}-{start + len(frames)} out_mean={out.mean().item():.4f} "
                      f"buffered={reader.prefetch_buffered}")
        else:
            out = fake_infer(work.permute(0, 2, 3, 1))
            print(f"  batch {start}-{start + len(frames)} out_mean={out.mean().item():.4f} "
                  f"buffered={reader.prefetch_buffered}")
        done += len(frames)
        if len(frames) < step:
            break

    if use_cuda and infer_stream is not None:
        infer_stream.synchronize()
    dt = time.perf_counter() - t0
    reader.stop_prefetch()
    print(f"Done: {done} frames in {dt:.2f}s = {done / dt:.1f} fps "
          f"(prefetched decode and inference overlapped)")


if __name__ == "__main__":
    sys.exit(main())
