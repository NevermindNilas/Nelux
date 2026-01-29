import sys
import subprocess
import time
import numpy as np
import cv2
import torch

from nelux import VideoReader
from utils.video_downloader import (
    get_video,
)  # utility to download open source test clips

VIDEO_PATH = get_video(
    "full"
)  # default to full video, change to "lite" for shorter video.
MAX_W, MAX_H = 1920, 1080


def ffmpeg_rgb24_pipe(path):
    """
    Launch ffmpeg to decode `path` into raw RGB24 piped frames.
    Yields numpy arrays shaped (H, W, 3), dtype=uint8.
    """
    # probe width/height
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            path,
        ],
        capture_output=True,
        text=True,
    )
    w, h = map(int, p.stdout.strip().split("x"))

    # spawn ffmpeg pipe
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-color_range",
        "pc",  # full range metadata
        "-pix_fmt",
        "rgb24",  # raw RGB triplets
        "-an",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_size = w * h * 3

    # yield each frame as an RGB numpy array
    while True:
        data = proc.stdout.read(frame_size)
        if len(data) < frame_size:
            break
        yield np.frombuffer(data, np.uint8).reshape((h, w, 3))

    proc.stdout.close()
    proc.wait()


def compare_with_ffmpeg(path):
    """
    Compare every frame from Nelux vs. an ffmpeg RGB24 pipe.
    Displays side-by-side in OpenCV and logs a sample pixel + MAD.
    """
    ce_reader = VideoReader(path)
    ff_pipe = ffmpeg_rgb24_pipe(path)

    for idx, nelux_frame in enumerate(ce_reader):
        # --- read & prep raw arrays ---
        arr_nelux = nelux_frame.numpy()

        # Handle Float32 output (likely 0.0-1.0 range)
        if arr_nelux.dtype == np.float32:
            # Simple heuristic: if max <= 1.0, assume 0-1 range and scale up
            # (Or just always scale if we know NeLux now outputs 0-1 float)
            # We'll assume 0-1 for float32 as per standard ffmpeg/swscale behavior for float formats.
            arr_nelux = (arr_nelux * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr_nelux = arr_nelux.astype(np.uint8)

        try:
            arr_ff = next(ff_pipe)  # RGB
        except StopIteration:
            print("FFmpeg pipe ended early.")
            break

        # shape check
        if arr_ff.shape != arr_nelux.shape:
            raise RuntimeError(
                f"Shape mismatch on frame {idx}: {arr_nelux.shape} vs {arr_ff.shape}"
            )

        # compute Mean Absolute Difference
        diff = np.abs(arr_nelux.astype(int) - arr_ff.astype(int))
        mad = diff.mean()

        # --- logging ---
        h0, w0, _ = arr_nelux.shape
        # pick center pixel for a quick sanity check
        yc, xc = h0 // 2, w0 // 2
        print(
            f"[Frame {idx:03d}] "
            f"Sample@({xc},{yc}) NeLux={arr_nelux[yc, xc]} "
            f"FFmpeg={arr_ff[yc, xc]}  MAD={mad:.2f}"
        )

        # --- prepare for display (OpenCV wants BGR) ---
        arr_nelux_bgr = cv2.cvtColor(arr_nelux, cv2.COLOR_RGB2BGR)
        arr_ff_bgr = cv2.cvtColor(arr_ff, cv2.COLOR_RGB2BGR)

        combined = np.hstack((arr_nelux_bgr, arr_ff_bgr))

        # scale down if too large
        h, w = combined.shape[:2]
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        display = cv2.resize(combined, (disp_w, disp_h))

        # overlay text
        cv2.putText(
            display,
            f"Frame {idx:03d}  MAD={mad:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1 * scale,
            (0, 255, 0),
            max(1, int(2 * scale)),
        )

        # show & handle keys
        cv2.imshow("NeLux (L) | FFmpeg (R)", display)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            # extra debug print if you really want
            print(f"[Frame {idx:03d}] MAD = {mad:.2f}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    start = time.time()
    compare_with_ffmpeg(VIDEO_PATH)
    print(f"Done in {time.time() - start:.1f}s")
