import random
import time
import numpy as np
import cv2
import torch
import sys

from nelux import VideoReader
from tests.utils.video_downloader import (
    get_video,
)  # utility to download open source test clips

VIDEO_PATH = get_video(
    "lite"
)  # default to full video, change to "lite" for shorter video.


def tensor_to_bgr_uint8(frame: torch.Tensor, bit_depth: int) -> np.ndarray:
    """
    Convert HxWxC torch tensor (uint8/uint16/float) to uint8 BGR for cv2.imshow.
    Assumes RGB order from VideoReader. Returns a numpy array.
    """
    if frame.numel() == 0:
        return None

    # Move to CPU and numpy (zero-copy when possible)
    arr = frame.cpu().contiguous().numpy()  # HWC, channels=3

    # Normalize to uint8
    if arr.dtype == np.uint8:
        rgb8 = arr
    elif arr.dtype == np.uint16:
        # Map 10/12/16-bit to 8-bit (bit_depth tells how much is meaningful)
        shift = max(0, bit_depth - 8)
        rgb8 = (arr >> shift).astype(np.uint8)
    elif arr.dtype in (np.float16, np.float32, np.float64):
        # Assume 0..1; clamp and scale
        rgb8 = np.clip(arr, 0.0, 1.0)
        rgb8 = (rgb8 * 255.0 + 0.5).astype(np.uint8)
    elif arr.dtype == np.uint32:
        # Very rare as video output; downscale conservatively
        rgb8 = (arr >> 24).astype(np.uint8)
    else:
        # Fallback: scale to 0..255 based on min/max
        amin, amax = arr.min(), arr.max()
        if amax > amin:
            rgb8 = ((arr - amin) * (255.0 / (amax - amin))).astype(np.uint8)
        else:
            rgb8 = np.zeros_like(arr, dtype=np.uint8)

    # Convert RGB->BGR for OpenCV
    bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    return bgr8


def main():
    path = VIDEO_PATH
    vr = VideoReader(path)
    props = vr.get_properties()
    fps = float(props["fps"]) if props["fps"] else 30.0
    duration = float(props["duration"])
    total_frames = int(props["total_frames"])
    bit_depth = int(props.get("bit_depth", 8))

    print("Video properties:", props)

    vr.reset()
    paused = False
    i = 0
    last_random_show = 0
    random_interval_frames = max(30, int(fps * 3))  # show a random frame ~every 3s

    while True:
        if not paused:
            frame = vr.read_frame()
            if frame.numel() == 0:
                print("EOF or decode fail")
                break

            img = tensor_to_bgr_uint8(frame, bit_depth)
            if img is None:
                print("Bad frame")
                break

            # Annotate and show sequential frame
            cv2.putText(
                img,
                f"Sequential frame {i}/{total_frames}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Sequential", img)
            i += 1

            # Periodically also show a random-access frame (second decoder via frame_at)
            if i - last_random_show >= random_interval_frames:
                # pick a random timestamp safely within duration
                t = random.uniform(0.0, max(0.0, duration - 0.001))
                try:
                    rnd = vr.frame_at(
                        t
                    )  # uses the secondary decoder; won't disturb sequential
                    rnd_img = tensor_to_bgr_uint8(rnd, bit_depth)
                    if rnd_img is not None:
                        cv2.putText(
                            rnd_img,
                            f"Random @ {t:.3f}s",
                            (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 200, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        # cv2.imshow("RandomAccess", rnd_img)
                    last_random_show = i
                except Exception as e:
                    print(f"Random access failed: {e}")

        # UI controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):  # pause / resume
            paused = not paused
        elif key == ord("r"):  # on-demand random access
            t = random.uniform(0.0, max(0.0, duration - 0.001))
            try:
                rnd = vr.frame_at(t)
                rnd_img = tensor_to_bgr_uint8(rnd, bit_depth)
                if rnd_img is not None:
                    cv2.putText(
                        rnd_img,
                        f"Random @ {t:.3f}s (manual)",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )
            #  cv2.imshow("RandomAccess", rnd_img)
            except Exception as e:
                print(f"Random access failed: {e}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
