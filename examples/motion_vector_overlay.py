"""Render a video with decoded motion vectors drawn over each frame.

Usage:
  python examples/motion_vector_overlay.py input.mp4 output.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

import torch  # Must be imported before nelux

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _add_ffmpeg_dll_dir() -> None:
    if os.name != "nt":
        return
    ffmpeg_dir = os.environ.get("NELUX_FFMPEG_DLL_DIR") or os.environ.get("FFMPEG_DLL_DIR")
    if not ffmpeg_dir:
        ffmpeg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "ffmpeg", "bin"))
    if os.path.isdir(ffmpeg_dir):
        os.add_dll_directory(ffmpeg_dir)


_add_ffmpeg_dll_dir()

from nelux import VideoEncoder, VideoReader  # noqa: E402


COLOR = torch.tensor([255, 40, 40], dtype=torch.uint8)
TIP = torch.tensor([255, 230, 80], dtype=torch.uint8)


def _draw_line(frame: torch.Tensor, x0: int, y0: int, x1: int, y1: int, color: torch.Tensor) -> None:
    h, w = frame.shape[:2]
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            frame[y0, x0] = color
        if x0 == x1 and y0 == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _draw_arrow(frame: torch.Tensor, x0: int, y0: int, x1: int, y1: int) -> None:
    _draw_line(frame, x0, y0, x1, y1, COLOR)
    vx = x1 - x0
    vy = y1 - y0
    if vx == 0 and vy == 0:
        return
    # ponytail: integer arrowhead is rough; switch to vector graphics if publication polish matters.
    hx = -1 if vx > 0 else 1 if vx < 0 else 0
    hy = -1 if vy > 0 else 1 if vy < 0 else 0
    _draw_line(frame, x1, y1, x1 + 3 * hx + 2 * hy, y1 + 3 * hy - 2 * hx, TIP)
    _draw_line(frame, x1, y1, x1 + 3 * hx - 2 * hy, y1 + 3 * hy + 2 * hx, TIP)


def draw_vectors(frame: torch.Tensor, vectors: list[dict], stride: int, scale: float) -> torch.Tensor:
    out = frame.cpu().contiguous().clone()
    if out.dtype != torch.uint8:
        out = (out >> 8).to(torch.uint8)
    for i, mv in enumerate(vectors):
        if stride > 1 and i % stride:
            continue
        x0 = int(mv["dst_x"])
        y0 = int(mv["dst_y"])
        motion_scale = max(1, int(mv["motion_scale"]))
        x1 = int(round(x0 + int(mv["motion_x"]) * scale / motion_scale))
        y1 = int(round(y0 + int(mv["motion_y"]) * scale / motion_scale))
        _draw_arrow(out, x0, y0, x1, y1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=2, help="draw every Nth vector")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply vector length")
    args = parser.parse_args()

    reader = VideoReader(args.input, decode_accelerator="cpu", convert_workers=0)
    encoder = VideoEncoder(
        args.output,
        codec="libx264",
        width=reader.width,
        height=reader.height,
        fps=float(reader.fps) if reader.fps > 0 else 30.0,
        pixel_format="yuv420p",
        preset="ultrafast",
    )

    frames = 0
    vectors = 0
    try:
        while frames < args.max_frames:
            frame, mvs = reader.read_frame_with_motion_vectors()
            if not frame.numel():
                break
            overlay = draw_vectors(frame, mvs, max(1, args.stride), args.scale)
            encoder.encode_frame(overlay)
            frames += 1
            vectors += len(mvs)
    finally:
        encoder.close()

    print(f"wrote {frames} frames, drew from {vectors} vectors -> {args.output}")
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
