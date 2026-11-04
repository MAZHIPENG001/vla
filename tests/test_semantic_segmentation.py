"""Exercise sampling, interval alignment and persistence without model weights."""

from io import BytesIO
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from data_process.config import resolve_camera_keys
from data_process.dataset import LeRobotReader
from data_process.semantic_segmentation import (
    EpisodeSource, SegmentationConfig, SemanticSegmenter, main,
)


class ScriptedVisionModel:
    metadata = {"backend": "scripted_test"}

    def __init__(self, invalid_once=False, inconclusive=False):
        self.calls = 0
        self.invalid_once = invalid_once
        self.inconclusive = inconclusive

    def generate(self, messages):
        self.calls += 1
        if self.invalid_once and self.calls == 1:
            return '{"segments":[{"start_sample":999,"subtask":"抓取","evidence":"test"}]}'
        content = messages[0]["content"]
        prompt = content[0]["text"]
        rows = [int(match.group(1)) for item in content if item["type"] == "text"
                for match in [re.search(r"row_index=(\d+)", item["text"])] if match]
        if '"subtasks"' in prompt:
            return json.dumps({"subtasks": ["抓取", "搬运", "放下"]})
        label = lambda row: "抓取" if row < 17 else "搬运" if row < 43 else "放下"
        if '"boundary_sample"' in prompt:
            target = "搬运" if "到“搬运”" in prompt else "放下"
            sample = next((i for i, row in enumerate(rows) if label(row) == target), None)
            return json.dumps({"boundary_sample": None if self.inconclusive else sample,
                               "evidence": "测试中的可见行为切换"})
        items = []
        for sample, row in enumerate(rows):
            if not items or items[-1]["subtask"] != label(row):
                items.append({"start_sample": sample, "subtask": label(row), "evidence": "可见行为"})
        return json.dumps({"segments": items})


class SemanticSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "data").mkdir()
        self.times = np.cumsum(np.resize([0.08, 0.12, 0.1], 60)) - 0.08
        self.features = {"timestamp": {"dtype": "float64"}, "frame_index": {"dtype": "int64"},
                         "task_index": {"dtype": "int64"}, "camera": {"dtype": "image"},
                         "wrist": {"dtype": "image"}}
        self.info = dict(codebase_version="v2.1", features=self.features, fps=10,
                         chunks_size=1000, data_path="data/episode_{episode_index:06d}.parquet",
                         video_path="videos/{video_key}/episode_{episode_index:06d}.mp4")
        self._write_info()
        (self.dataset / "meta/episodes.jsonl").write_text(json.dumps(
            {"episode_index": 0, "length": 60, "tasks": ["抓取物体并搬运放下"]}) + "\n")
        (self.dataset / "meta/tasks.jsonl").write_text(json.dumps(
            {"task_index": 0, "task": "抓取物体并搬运放下"}) + "\n")
        images = []
        for row in range(60):
            data = BytesIO()
            Image.new("RGB", (48, 32), (row, 0, 0)).save(data, format="PNG")
            images.append({"bytes": data.getvalue(), "path": None})
        self.data = dict(timestamp=self.times, frame_index=np.arange(60) * 2 + 100,
                         task_index=np.zeros(60, dtype=int), camera=images, wrist=images)
        self._write_data()
        self.config = SegmentationConfig(window_seconds=2.5, overlap_seconds=0.7,
                                         max_frames=8, overview_frames=8, refine_frames=30,
                                         refine_seconds=0.4)

    def _write_info(self):
        (self.dataset / "meta/info.json").write_text(json.dumps(self.info))

    def _write_data(self):
        pq.write_table(pa.table(self.data), self.dataset / "data/episode_000000.parquet")

    def source(self, **kwargs):
        return EpisodeSource(LeRobotReader(self.dataset), 0, **kwargs)

    def test_camera_aliases_use_real_dataset_keys_for_decoding_and_reports(self):
        for old, new in (("camera", "image"), ("wrist", "wrist_image")):
            self.features[new] = self.features.pop(old)
            self.data[new] = self.data.pop(old)
        self._write_info()
        self._write_data()
        source = self.source(cameras=["observation.images.cam_high", "observation.images.cam_wrist"])
        self.assertEqual(source.cameras, ["image", "wrist_image"])
        self.assertEqual(set(source.images([0], 32)[0]), {"image", "wrist_image"})
        report = SemanticSegmenter(ScriptedVisionModel(), self.config).segment(source)
        self.assertEqual(report["cameras"], ["image", "wrist_image"])

    def test_camera_resolution_prefers_exact_names_and_handles_prefixes_and_globs(self):
        available = ["observation.images.cam_high", "observation.images.cam_left_wrist",
                     "observation.images.cam_right_wrist", "video.primary_image"]
        self.assertEqual(resolve_camera_keys(available, ["observation.images.cam_high"]), available[:1])
        self.assertEqual(resolve_camera_keys(available, ["video.cam_high"]), available[:1])
        self.assertEqual(resolve_camera_keys(available, ["PRIMARY_IMAGE"]), ["video.primary_image"])
        self.assertEqual(resolve_camera_keys(available, ["*wrist*", "cam_left_wrist"]), available[1:3])
        self.assertEqual(resolve_camera_keys(["image", "wrist_image"], ["observation.images.*"]),
                         ["image", "wrist_image"])
        self.assertEqual(resolve_camera_keys(available), available[:1])
        self.assertEqual(resolve_camera_keys(["video.wrist_image"], ["cam_wrist"]), ["video.wrist_image"])

    def test_camera_resolution_rejects_missing_or_ambiguous_names(self):
        with self.assertRaisesRegex(ValueError, "匹配多个字段"):
            resolve_camera_keys(["video.primary_image", "observation.images.cam_high"], ["image"])
        with self.assertRaisesRegex(ValueError, "匹配多个字段"):
            resolve_camera_keys(["left.image", "right.image"], ["image"])
        with self.assertRaisesRegex(ValueError, "可用相机"):
            resolve_camera_keys(["image", "wrist_image"], ["unknown_camera"])
        with self.assertRaisesRegex(ValueError, "可用相机"):
            resolve_camera_keys(["image", "wrist_image"], ["*depth*"])
        with self.assertRaisesRegex(ValueError, "可用相机"):
            resolve_camera_keys(["cam_left_wrist", "cam_right_wrist"], ["cam_wrist"])

    def test_overlapping_windows_refine_boundaries_and_keep_original_identifiers(self):
        before = {path: path.read_bytes() for path in self.dataset.rglob("*") if path.is_file()}
        report = SemanticSegmenter(ScriptedVisionModel(), self.config).segment(
            self.source(cameras=["camera", "wrist"]))
        segments = report["segments"]
        self.assertEqual([(s["start_frame"], s["end_frame_exclusive"], s["subtask"]) for s in segments],
                         [(0, 17, "抓取"), (17, 43, "搬运"), (43, 60, "放下")])
        self.assertEqual(segments[1]["start_frame_index"], 134)
        self.assertEqual(segments[1]["start_time"], self.times[17])
        self.assertEqual(segments[1]["last_time"], self.times[42])
        for segment in segments[1:]:
            boundary = segment["boundary_before"]
            self.assertEqual(boundary["method"], "refined_samples")
            self.assertLessEqual(boundary["lower_row"], segment["start_frame"])
            self.assertGreaterEqual(boundary["upper_row"], segment["start_frame"])
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        json.dumps(report, allow_nan=False)

    def test_bad_model_output_retries_without_silently_accepting_out_of_range_rows(self):
        backend = ScriptedVisionModel(invalid_once=True)
        report = SemanticSegmenter(backend, self.config).segment(self.source(), labels=["抓取", "搬运", "放下"])
        attempts = report["inference_calls"][0]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertIn("validation_error", attempts[0])
        class InvalidModel:
            def generate(self, messages):
                return "not JSON"
        with self.assertRaisesRegex(ValueError, "校验失败"):
            SemanticSegmenter(InvalidModel(), self.config).segment(self.source())

    def test_inconclusive_refinement_keeps_coarse_boundary(self):
        report = SemanticSegmenter(ScriptedVisionModel(inconclusive=True), self.config).segment(self.source())
        self.assertEqual(report["segments"][-1]["end_frame_exclusive"], 60)
        self.assertTrue(all(s["boundary_before"]["refinement"] == "inconclusive"
                            for s in report["segments"][1:]))

    def test_local_windows_can_discover_steps_missed_by_overview_but_explicit_labels_are_fixed(self):
        class IncompleteOverview(ScriptedVisionModel):
            def generate(self, messages):
                if '"subtasks"' in messages[0]["content"][0]["text"]:
                    return '{"subtasks":["抓取"]}'
                return super().generate(messages)

        report = SemanticSegmenter(IncompleteOverview(), self.config).segment(self.source())
        self.assertIn("搬运", report["labels"])
        self.assertIn("放下", report["labels"])
        with self.assertRaisesRegex(ValueError, "指定词表"):
            SemanticSegmenter(ScriptedVisionModel(), self.config).segment(self.source(), labels=["抓取"])

    def test_keyboard_interrupt_preserves_completed_reports(self):
        class InterruptedModel(ScriptedVisionModel):
            def generate(self, messages):
                raise KeyboardInterrupt

        output = self.root / "reports"
        with patch("data_process.semantic_segmentation.TransformersVisionModel", return_value=InterruptedModel()):
            self.assertEqual(main(["--path", str(self.dataset), "--output", str(output)]), 130)
        manifest = json.loads(next(output.glob("*/manifest.json")).read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(manifest["completed"], [])

    def test_fps_fallback_does_not_replace_invalid_original_timestamps(self):
        self.data["timestamp"][12] = self.data["timestamp"][11]
        self._write_data()
        with self.assertRaisesRegex(ValueError, "严格递增"):
            self.source()
        self.features.pop("timestamp")
        self.data.pop("timestamp")
        self._write_info()
        self._write_data()
        source = self.source()
        self.assertEqual(source.timestamp_source, "estimated_from_fps")
        np.testing.assert_allclose(source.times, np.arange(60) / 10)

    def test_single_frame_and_small_sampling_budgets_still_cover_episode(self):
        for key in self.data:
            self.data[key] = self.data[key][:1]
        self._write_data()
        (self.dataset / "meta/episodes.jsonl").write_text(json.dumps(
            {"episode_index": 0, "length": 1, "tasks": ["抓取"]}) + "\n")
        report = SemanticSegmenter(ScriptedVisionModel(), self.config).segment(self.source())
        self.assertEqual(report["segments"][0]["end_frame_exclusive"], 1)
        self.assertEqual(report["segments"][0]["last_time"], self.times[0])

    def test_dry_run_never_loads_model_or_writes_reports(self):
        output = self.root / "reports"
        with patch("data_process.semantic_segmentation.TransformersVisionModel") as model:
            self.assertEqual(main(["--path", str(self.dataset), "--episodes", "0", "--dry-run",
                                   "--output", str(output)]), 0)
            model.assert_not_called()
        self.assertFalse(output.exists())
        with self.assertRaises(SystemExit):
            main(["--path", str(self.dataset), "--output", str(self.dataset / "reports"), "--dry-run"])

    def test_cli_persists_reports_and_records_failed_episode_as_partial(self):
        # A second episode exists in metadata but its Parquet file is absent.
        with (self.dataset / "meta/episodes.jsonl").open("a") as stream:
            stream.write(json.dumps({"episode_index": 2, "length": 60, "tasks": ["抓取"]}) + "\n")
        output = self.root / "reports"
        with patch("data_process.semantic_segmentation.TransformersVisionModel", return_value=ScriptedVisionModel()):
            result = main(["--path", str(self.dataset), "--episodes", "0", "2", "--output", str(output)])
        self.assertEqual(result, 1)
        manifest_path = next(output.glob("*/manifest.json"))
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(manifest["completed"], [0])
        self.assertEqual(manifest["errors"][0]["episode_index"], 2)
        self.assertTrue((manifest_path.parent / "episode_000000.json").exists())
        self.assertFalse((manifest_path.parent / "episode_000002.json").exists())

    def test_video_frames_follow_pts_and_original_row_mapping(self):
        import av
        self.features["camera"] = {"dtype": "video"}
        self._write_info()
        folder = self.dataset / "videos/camera"
        folder.mkdir(parents=True)
        with av.open(str(folder / "episode_000000.mp4"), "w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height, stream.pix_fmt = 48, 32, "yuv420p"
            for row in range(60):
                image = np.full((32, 48, 3), row * 3, dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        source = self.source()
        images = source.images([0, 20, 40, 59], 32)
        for row in images:
            image = images[row]["camera"]
            self.assertLessEqual(max(image.size), 32)
            expected = round(self.times[row] * 10) * 3
            self.assertLess(abs(np.asarray(image).mean() - expected), 8)

    def test_window_ownership_with_irregular_clocks_always_covers_each_row_once(self):
        # Include gaps greater than a whole window and zero overlap.
        source = self.source()
        source.times[30:] += 10
        config = SegmentationConfig(window_seconds=1, overlap_seconds=0, max_frames=3,
                                    overview_frames=3, refine_seconds=0)
        report = SemanticSegmenter(ScriptedVisionModel(), config).segment(source)
        covered = [row for segment in report["segments"]
                   for row in range(segment["start_frame"], segment["end_frame_exclusive"])]
        self.assertEqual(covered, list(range(60)))
        self.assertTrue(all(len(call["rows"]) <= 3 for call in report["inference_calls"]))


if __name__ == "__main__":
    unittest.main()
