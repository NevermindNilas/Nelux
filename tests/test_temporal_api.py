"""Public frame identity and preprocessing contracts."""

import subprocess
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

import nelux


def _ffmpeg():
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    override = os.environ.get("FFMPEG_BIN") or os.environ.get("NELUX_FFMPEG_BIN")
    if override:
        path = Path(override)
        path = path / name if path.is_dir() else path
        if path.is_file():
            return path
    bundled = Path(__file__).resolve().parents[1] / "external/ffmpeg/bin" / name
    if bundled.is_file():
        return bundled
    found = shutil.which("ffmpeg")
    if not found:
        pytest.skip("ffmpeg required to generate temporal fixtures")
    return Path(found)


@pytest.fixture(scope="module")
def vfr_clip(tmp_path_factory):
    ffmpeg = _ffmpeg()
    path = tmp_path_factory.mktemp("temporal") / "vfr.mkv"
    frames = np.empty((20, 64, 64, 3), dtype=np.uint8)
    for i in range(20):
        frames[i] = [i * 10, i * 5, i * 3]
    subprocess.run(
        [str(ffmpeg), "-v", "error", "-y", "-f", "rawvideo",
         "-pixel_format", "rgb24", "-video_size", "64x64", "-framerate", "10",
         "-i", "pipe:0", "-vf",
         r"settb=1/1000,setpts=if(lt(N\,10)\,N*100\,1000+(N-10)*300)",
         "-fps_mode", "passthrough", "-frames:v", "20", "-c:v", "ffv1",
         "-pix_fmt", "bgr0", str(path)],
        input=frames.tobytes(), check=True, capture_output=True, timeout=30,
    )
    return path


def test_vfr_random_and_batch_indices_match_sequential_frames(vfr_clip):
    with nelux.VideoReader(str(vfr_clip), num_threads=1, convert_workers=0) as reader:
        reference = [f.clone() for f in reader]
        assert len(reference) == 20
        for i in (12, 15, 19, 5, 12):
            assert torch.equal(reader.frame_at(i), reference[i]), f"frame_at({i})"
            assert torch.equal(reader[i], reference[i]), f"reader[{i}]"
        indices = [19, 12, 15, 12, 0]
        assert torch.equal(reader.get_batch(indices), torch.stack([reference[i] for i in indices]))


@pytest.mark.parametrize("color_format", ["rgb", "gray", "rgba"])
@pytest.mark.parametrize("resize_filter", ["bilinear", "neighbor", "lanczos"])
def test_batch_preprocessing_matches_streaming(vfr_clip, color_format, resize_filter):
    with nelux.VideoReader(str(vfr_clip), num_threads=1, convert_workers=0,
                           resize=(32, 48), color_format=color_format,
                           resize_filter=resize_filter) as reader:
        reference = [f.clone() for f in reader]
        indices = [19, 12, 0, 12]
        batch = reader.get_batch(indices)
        assert torch.equal(batch, torch.stack([reference[i] for i in indices]))
        assert batch.shape == (4, 48, 32, {"rgb": 3, "gray": 1, "rgba": 4}[color_format])


def test_batch_does_not_move_streaming_reader(vfr_clip):
    with nelux.VideoReader(str(vfr_clip), num_threads=1, convert_workers=0) as reader:
        first = reader.read_frame().clone()
        reader.get_batch([19, 12, 0])
        second = reader.read_frame().clone()
        assert first[0, 0].tolist() == [0, 0, 0]
        assert second[0, 0].tolist() == [10, 5, 3]


def test_timing_results_and_playback_intervals(vfr_clip):
    with nelux.VideoReader(vfr_clip, num_threads=1, convert_workers=0) as reader:
        frame = reader.get_frame_at(12)
        assert isinstance(frame, nelux.Frame)
        assert frame.pts_seconds == pytest.approx(1.6)
        assert frame.duration_seconds == pytest.approx(0.3)
        played = reader.get_frame_played_at(1.75)
        assert torch.equal(played.data, frame.data)
        batch = reader.get_frames_at([19, 12, 12, 0])
        assert batch.pts_seconds.tolist() == pytest.approx([3.7, 1.6, 1.6, 0.0])
        assert torch.equal(batch.data[1], batch.data[2])
        for time in (-0.1, float("nan"), 20.0):
            with pytest.raises(ValueError):
                reader.get_frame_played_at(time)


def test_chw_layout_is_a_view_and_sampler_preserves_timing(vfr_clip):
    with nelux.VideoReader(vfr_clip, dimension_order="CHW", num_threads=1,
                           convert_workers=0) as reader:
        assert reader.read_frame().shape == (3, 64, 64)
        assert next(iter(reader)).shape == (3, 64, 64)
        assert reader[12].shape == (3, 64, 64)
        assert reader.get_batch([12, 0]).shape == (2, 3, 64, 64)
        clips = nelux.samplers.clips_at_indices(reader, [18, 0], num_frames_per_clip=3,
                                                policy="repeat_last")
        assert clips.data.shape == (2, 3, 3, 64, 64)
        assert clips.pts_seconds[0].tolist() == pytest.approx([3.4, 3.7, 3.7])
        assert torch.equal(clips.data[0, 1], clips.data[0, 2])


def test_seeded_random_sampler(vfr_clip):
    with nelux.VideoReader(vfr_clip, num_threads=1, convert_workers=0) as reader:
        a = nelux.samplers.clips_at_random_indices(reader, num_clips=4,
            num_frames_per_clip=3, generator=torch.Generator().manual_seed(42))
        b = nelux.samplers.clips_at_random_indices(reader, num_clips=4,
            num_frames_per_clip=3, generator=torch.Generator().manual_seed(42))
        assert torch.equal(a.data, b.data)
        assert torch.equal(a.pts_seconds, b.pts_seconds)


def test_reusable_index(vfr_clip, tmp_path):
    with nelux.VideoReader(vfr_clip, num_threads=1) as reader:
        mapping = reader.frame_index
        assert mapping.num_frames == 20
    with nelux.VideoReader(vfr_clip, frame_index=mapping, resize=(32, 32)) as reader:
        assert reader.get_frame_at(12).pts_seconds == pytest.approx(1.6)
    copy = tmp_path / "copy.mkv"
    copy.write_bytes(vfr_clip.read_bytes())
    with pytest.raises(ValueError, match="different or modified"):
        nelux.VideoReader(copy, frame_index=mapping)


@pytest.fixture(scope="module")
def cuda_vfr_clip(vfr_clip):
    path = vfr_clip.with_suffix(".mp4")
    ffmpeg = _ffmpeg()
    subprocess.run([str(ffmpeg), "-v", "error", "-y", "-i", str(vfr_clip),
                    "-fps_mode", "passthrough", "-c:v", "libx264", "-g", "5",
                    "-bf", "2", "-pix_fmt", "yuv420p", str(path)],
                   check=True, capture_output=True, timeout=30)
    return path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("async_frames", [False, True])
def test_cuda_sparse_batch_and_owned_outputs(cuda_vfr_clip, async_frames):
    with nelux.VideoReader(cuda_vfr_clip, decode_accelerator="nvdec",
                           copy_frames=True, resize=(32, 32), async_frames=async_frames) as reader:
        reference = [f for f in reader]
        assert len(reference) == 20
        assert len({f.data_ptr() for f in reference}) == 20
        snapshots = torch.stack(reference).clone()
        for indices in ([12, 19, 12], [15, 16], [0, 19], [19, 18, 19]):
            assert torch.equal(reader.get_batch(indices), snapshots[indices])
        assert torch.equal(torch.stack(reference), snapshots)
        assert torch.equal(reader.get_frame_at(12).data, snapshots[12])


@pytest.mark.parametrize("source_kind", ["bytes", "file", "array", "tensor"])
def test_encoded_sources_are_seekable_and_cleaned(vfr_clip, source_kind):
    import io

    payload = vfr_clip.read_bytes()
    source = {"bytes": lambda: payload, "file": lambda: io.BytesIO(payload),
              "array": lambda: np.frombuffer(payload, dtype=np.uint8),
              "tensor": lambda: torch.tensor(list(payload), dtype=torch.uint8)}[source_kind]()
    with nelux.VideoReader(source, num_threads=1, convert_workers=0) as reader:
        spool = Path(reader._source_owner.name)
        assert reader.get_frame_at(12).pts_seconds == pytest.approx(1.6)
        assert reader.metadata.num_frames == 20
    assert not spool.exists()
    if source_kind == "file":
        assert source.tell() == 0


def test_selected_video_stream(tmp_path):
    ffmpeg = _ffmpeg()
    path = tmp_path / "streams.mkv"
    subprocess.run([str(ffmpeg), "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=red:size=64x64:rate=10:duration=0.3", "-f", "lavfi", "-i",
                    "color=green:size=96x64:rate=10:duration=0.5", "-map", "0:v",
                    "-map", "1:v", "-c:v", "ffv1", str(path)],
                   check=True, capture_output=True, timeout=30)
    with nelux.VideoReader(path, stream_index=1, num_threads=1) as reader:
        assert reader.width == 96
        assert reader.frame_count == 5
        assert reader.get_frame_at(0).data[0, 0, 1] > 100
        assert torch.equal(reader.get_batch([4, 0])[1], reader.read_frame())
    for stream in (-1, 2):
        with pytest.raises((ValueError, RuntimeError), match="stream_index"):
            nelux.VideoReader(path, stream_index=stream)


def test_approximate_mode_does_not_build_exact_index(cuda_vfr_clip):
    with nelux.VideoReader(cuda_vfr_clip, seek_mode="approximate", num_threads=1) as reader:
        reader.get_batch([0, 1])
        reader.frame_at(1)
        assert reader._get_sampling_stats()[0] == 0
        # Rich results opt into exact identity even on an approximate reader.
        frame = reader.get_frame_at(12)
        assert frame.pts_seconds == pytest.approx(1.6)
        assert reader._get_sampling_stats()[0] == 20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_adjacent_batches_reuse_decoder_and_skip_conversion(cuda_vfr_clip):
    with nelux.VideoReader(cuda_vfr_clip, decode_accelerator="nvdec") as reader:
        reader.get_batch([12, 12])
        indexed, decoded, converted, opens, seeks = reader._get_sampling_stats()
        assert indexed == 20
        assert converted == 1
        assert decoded > converted
        assert opens == 1
        reader.get_batch([13, 14, 14])
        _, next_decoded, next_converted, next_opens, _ = reader._get_sampling_stats()
        assert next_decoded == decoded + 2
        assert next_converted == converted + 2
        assert next_opens == opens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_async_owned_frames_across_cuda_streams_and_reconfigure(cuda_vfr_clip):
    with nelux.VideoReader(cuda_vfr_clip, decode_accelerator="nvdec",
                           copy_frames=True) as baseline:
        expected = torch.stack([f for f in baseline])
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    with nelux.VideoReader(cuda_vfr_clip, decode_accelerator="nvdec", copy_frames=True,
                           async_frames=True) as reader:
        frames = []
        for index in range(20):
            with torch.cuda.stream(streams[index % 2]):
                frames.append(reader.read_frame())
        torch.cuda.synchronize()
        assert torch.equal(torch.stack(frames), expected)
        reader.reconfigure(cuda_vfr_clip)
        assert torch.equal(reader.read_frame(), expected[0])


def test_native_depth_batch_preprocessing(tmp_path):
    ffmpeg = _ffmpeg()
    path = tmp_path / "deep.mkv"
    ramp = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 13
    frames = np.stack([ramp, ramp + 101, ramp + 202])
    subprocess.run([str(ffmpeg), "-v", "error", "-y", "-f", "rawvideo", "-pixel_format",
                    "gray16le", "-video_size", "64x64", "-framerate", "10", "-i", "pipe:0",
                    "-frames:v", "3", "-c:v", "ffv1", str(path)], input=frames.tobytes(),
                   check=True, capture_output=True, timeout=30)
    for color in ("rgb", "gray", "rgba"):
        with nelux.VideoReader(path, color_format=color, resize=(29, 31), num_threads=1,
                               convert_workers=0) as reader:
            expected = [f.clone() for f in reader]
            batch = reader.get_batch([2, 0, 1, 2])
            assert batch.dtype == torch.uint16
            assert torch.equal(batch, torch.stack([expected[i] for i in [2, 0, 1, 2]]))

