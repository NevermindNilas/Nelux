import random
import numpy as np
import cv2
import torch
import sys

sys.path.append(".")

from nelux import VideoReader
from utils.video_downloader import get_video

VIDEO_PATH = get_video("lite")  # short clip for quick manual runs


def tensor_to_bgr_uint8(frame: torch.Tensor, bit_depth: int) -> np.ndarray | None:
    """Convert HxWxC tensor (uint8/uint16/float/uint32) to uint8 BGR for imshow."""
    if frame is None or (frame.numel() == 0):
        return None
    arr = frame.cpu().contiguous().numpy()  # HWC
    if arr.dtype == np.uint8:
        rgb8 = arr
    elif arr.dtype == np.uint16:
        shift = max(0, bit_depth - 8)
        rgb8 = (arr >> shift).astype(np.uint8)
    elif arr.dtype in (np.float16, np.float32, np.float64):
        rgb8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    elif arr.dtype == np.uint32:
        rgb8 = (arr >> 24).astype(np.uint8)
    else:
        amin, amax = arr.min(), arr.max()
        if amax > amin:
            rgb8 = ((arr - amin) * (255.0 / (amax - amin))).astype(np.uint8)
        else:
            rgb8 = np.zeros_like(arr, dtype=np.uint8)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)


def main():
    vr = VideoReader(VIDEO_PATH)
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
    random_interval_frames = max(30, int(fps * 3))  # ~every 3s

    while True:
        if not paused:
            # Sequential decode
            frame = vr.read_frame()
            if frame is None:
                print("EOF or decode fail")
                break
            img = tensor_to_bgr_uint8(frame, bit_depth)
            if img is None:
                print("Bad frame")
                break
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

            # Periodic random-access demo (explicit frame_at, separate decoder)
            if i - last_random_show >= random_interval_frames:
                t = random.uniform(0.0, max(0.0, duration - 0.001))
                try:
                    rnd = vr.frame_at(t)  # explicit random-access path
                    rnd_img = tensor_to_bgr_uint8(rnd, bit_depth)
                    if rnd_img is not None:
                        cv2.putText(
                            rnd_img,
                            f"Random @ {t:.3f}s (frame_at)",
                            (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 200, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("RandomAccess", rnd_img)
                    last_random_show = i
                except Exception as e:
                    print(f"Random access failed: {e}")

            # Also show the smart __getitem__ by timestamp a tiny bit ahead (near case)
            near_ts = min(duration, i / max(1.0, fps) + 1.0 / max(1.0, fps))
            try:
                smart = vr[near_ts]
                smart_img = tensor_to_bgr_uint8(smart, bit_depth)
                if smart_img is not None:
                    cv2.putText(
                        smart_img,
                        f"Smart __getitem__ ~{near_ts:.3f}s",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 200, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("SmartGetItem", smart_img)
            except Exception as e:
                print(f"Smart __getitem__ failed: {e}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("r"):
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
                    cv2.imshow("RandomAccess", rnd_img)
            except Exception as e:
                print(f"Random access failed: {e}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
