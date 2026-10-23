"""局部运动异常、真实时间差分窗口、覆盖与只读报告的回归测试。"""

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from data_process.config import QualityPolicy, SignalPolicy
from data_process.dataset import BrowserModel
from data_process.processing import QualityInspector, _derivatives, save_quality_report


QUIET = SignalPolicy(static_min_frames=None, static_min_seconds=None,
                     velocity_zscore=None, acceleration_zscore=None)


class SegmentDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / 'meta').mkdir()
        (self.root / 'data').mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def dataset(self, state, times='default', action=None, *, eid=0, source_start=0):
        state = np.asarray(state, dtype=float)
        if state.ndim == 1:
            state = state[:, None]
        count = len(state)
        action = np.arange(count, dtype=float)[:, None] if action is None else np.asarray(action, dtype=float).reshape(count, -1)
        features = {'state': dict(dtype='float64', shape=[state.shape[1]]),
                    'actions': dict(dtype='float64', shape=[action.shape[1]]),
                    'frame_index': dict(dtype='int64', shape=[1])}
        values = dict(state=state.tolist(), actions=action.tolist(), frame_index=list(range(source_start, source_start + count)))
        if isinstance(times, str) and times == 'default':
            times = np.arange(count, dtype=float)
        if times is not None:
            features['timestamp'] = dict(dtype='float64', shape=[1])
            values['timestamp'] = np.asarray(times, dtype=float).tolist()
        info = dict(codebase_version='v2.1', features=features, fps=50, chunks_size=1000,
                    data_path='data/episode_{episode_index:06d}.parquet')
        (self.root / 'meta/info.json').write_text(json.dumps(info))
        (self.root / 'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=0, task='motion'))+'\n')
        metadata_path = self.root / 'meta/episodes.jsonl'
        with metadata_path.open('a') as stream:
            stream.write(json.dumps(dict(episode_index=eid, length=count, tasks=['motion']))+'\n')
        pq.write_table(pa.table(values), self.root / f'data/episode_{eid:06d}.parquet')

    def inspect(self, state_rule=QUIET, *, ids=(0,), **kwargs):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        policy = QualityPolicy(state=state_rule, action=replace(QUIET, kind='command'), inspect_images=False, **kwargs)
        report = QualityInspector(self.root).inspect(ids, policy)
        json.dumps(report, allow_nan=False)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertNotIn('rejected', report['summary'])
        self.assertTrue(all('decision' not in row for row in report['episodes']))
        return report

    def test_long_static_marks_only_interval_and_preserves_source_ids(self):
        values = np.arange(200, dtype=float)
        values[60:85] = 60
        values[85:] -= 24
        self.dataset(np.c_[values, values * 2], source_start=1000)
        report = self.inspect(replace(QUIET, static_min_frames=20))
        row = report['episodes'][0]
        segment = next(s for s in row['segments'] if s['status'] == 'suspect')
        self.assertEqual((segment['start_frame'], segment['end_frame']), (60, 84))
        self.assertEqual((segment['source_start_frame_index'], segment['source_end_frame_index']), (1060, 1084))
        self.assertEqual(segment['dimensions'], [0, 1])
        self.assertEqual(segment['duration_seconds'], 24.)
        self.assertEqual((row['normal_frames'], row['suspect_frames']), (175, 25))
        self.assertEqual([(s['start_frame'], s['end_frame']) for s in row['normal_segments']], [(0, 59), (85, 199)])
        self.assertEqual(row['frame_flags'][0]['source_frame_index'], 1060)
        self.assertEqual(report['selection_unit'], 'frame_segment')

    def test_short_wait_is_recorded_without_being_anomalous(self):
        values = np.arange(50, dtype=float)
        values[10:15] = 10
        values[15:] -= 4
        self.dataset(values)
        row = self.inspect(replace(QUIET, static_min_frames=20))['episodes'][0]
        waiting = [s for s in row['segments'] if s['type'] == 'static']
        self.assertEqual(len(waiting), 1)
        self.assertEqual(waiting[0]['status'], 'waiting')
        self.assertEqual(waiting[0]['frame_count'], 5)
        self.assertEqual(row['suspect_frames'], 0)
        self.assertEqual(row['normal_frames'], 50)

    def test_duration_threshold_uses_true_irregular_time(self):
        values = np.arange(40, dtype=float)
        values[10:15] = 10
        values[15:] -= 4
        times = np.cumsum(np.resize([.1, .2, .15], 40))
        self.dataset(values, times)
        row = self.inspect(replace(QUIET, static_min_frames=30, static_min_seconds=.5), frame_gap_factor=10)['episodes'][0]
        segment = next(s for s in row['segments'] if s['status'] == 'suspect')
        self.assertEqual((segment['start_frame'], segment['end_frame']), (10, 14))
        self.assertAlmostEqual(segment['duration_seconds'], times[14] - times[10])
        self.assertNotEqual(segment['duration_seconds'], segment['frame_count'] / 50)

    def test_derivative_midpoints_and_support_windows(self):
        times = np.array([0, .1, .35, .8, 1.4, 2.1])
        values = times[:, None] ** 3
        result = _derivatives(values, times, 'position')
        current, current_time = values, times
        for order in (1, 2, 3):
            current = np.diff(current, axis=0) / np.diff(current_time)[:, None]
            current_time = (current_time[:-1] + current_time[1:]) / 2
            np.testing.assert_allclose(result[order]['values'], current)
            np.testing.assert_allclose(result[order]['times'], current_time)
            np.testing.assert_array_equal(result[order]['starts'], np.arange(len(times) - order))
            np.testing.assert_array_equal(result[order]['ends'], np.arange(order, len(times)))
        for kind, first in [('velocity', 1), ('acceleration', 2)]:
            native = _derivatives(values, times, kind)
            np.testing.assert_allclose(native[first]['values'], values)
            np.testing.assert_array_equal(native[first + 1]['starts'], np.arange(len(times) - 1))

    def test_spike_records_dimensions_values_thresholds_and_rightmost_frames(self):
        values = np.c_[np.arange(200, dtype=float), np.arange(200, dtype=float)]
        values[120, 1] += 50
        self.dataset(values)
        report = self.inspect(replace(QUIET, velocity_zscore=6, acceleration_zscore=6))
        row = report['episodes'][0]
        velocity = next(s for s in row['segments'] if s['type'] == 'velocity')
        self.assertEqual((velocity['start_frame'], velocity['end_frame']), (120, 121))
        self.assertEqual(velocity['dimensions'], [1])
        first = velocity['samples'][0]
        self.assertEqual((first['window_start_frame'], first['frame_index']), (119, 120))
        self.assertEqual(first['values'], [51.])
        self.assertEqual(first['thresholds'], [6])
        self.assertEqual(first['timestamp'], 120.)
        self.assertEqual(first['metric_timestamp'], 119.5)
        self.assertGreater(first['zscores'][0], 6)
        self.assertTrue(any(s['type'] == 'acceleration' for s in row['segments']))
        self.assertGreater(row['normal_frames'], 190)
        self.assertGreater(row['suspect_frames'], 0)

    def test_jerk_per_original_dimension_and_unset_threshold(self):
        time = np.arange(15, dtype=float)
        self.dataset(np.c_[time, time * 2, time ** 3])
        row = self.inspect(replace(QUIET, dimensions=(0, 2), jerk_limit=(100, 999, .1)))['episodes'][0]
        jerk = next(s for s in row['segments'] if s['type'] == 'jerk')
        self.assertEqual(jerk['dimensions'], [2])
        self.assertEqual(jerk['samples'][0]['thresholds'], [.1])
        self.assertEqual(jerk['samples'][0]['values'], [6.])
        self.assertEqual(jerk['samples'][0]['window_start_frame'], 0)
        self.assertEqual(jerk['start_frame'], 3)
        unset = self.inspect()['episodes'][0]
        self.assertFalse(any(s['type'] == 'jerk' for s in unset['segments']))
        self.assertEqual(next(c for c in unset['checks'] if c['name'] == 'state.jerk')['status'], 'unverified')

    def test_nan_does_not_hide_another_dimension_or_remote_spike(self):
        values = np.c_[np.arange(200, dtype=float), np.arange(200, dtype=float)]
        values[30, 1] = np.nan
        values[150, 0] += 50
        self.dataset(values)
        row = self.inspect(replace(QUIET, velocity_zscore=6))['episodes'][0]
        self.assertIn(30, row['flagged_frames'])
        self.assertIn(150, row['flagged_frames'])
        self.assertGreater(row['normal_frames'], 190)
        flag = next(f for f in row['frame_flags'] if f['frame_index'] == 30)
        self.assertEqual(flag['dimensions']['state'], [1])
        segment = next(s for s in row['segments'] if s['type'] == 'velocity')
        self.assertEqual(segment['dimensions'], [0])
        finite = next(s for s in row['segments'] if s['check'] == 'state.finite')
        self.assertEqual(finite['dimensions'], [1])
        coverage = next(c for c in row['checks'] if c['name'] == 'state.velocity_zscore')['metrics']['coverage_per_dimension']
        self.assertTrue(any(a <= 30 <= b for a, b in coverage[0]['evaluated_frame_ranges']))
        self.assertFalse(any(a <= 30 <= b for a, b in coverage[1]['evaluated_frame_ranges']))

    def test_derivative_overflow_identifies_only_the_affected_dimension(self):
        values = np.c_[np.arange(20.), np.resize([-1e308, 1e308], 20)]
        self.dataset(values)
        row = self.inspect(replace(QUIET, velocity_zscore=6))['episodes'][0]
        segment = next(s for s in row['segments'] if s['check'] == 'state.velocity_zscore.nonfinite')
        self.assertEqual(segment['dimensions'], [1])
        self.assertEqual(segment['samples'][0]['dimensions'], [1])
        self.assertEqual(row['frame_flags'][0]['dimensions']['state'], [1])
        self.assertEqual(row['normal_frames'], 1)

    def test_zscore_is_not_disabled_by_a_large_signal_offset(self):
        values = np.full(200, 1e12)
        values[120] += 1
        self.dataset(values)
        row = self.inspect(replace(QUIET, kind='velocity', velocity_zscore=6))['episodes'][0]
        self.assertEqual(row['flagged_frames'], [120])
        self.assertEqual(row['normal_frames'], 199)

    def test_bad_time_edges_are_not_crossed_and_can_still_be_browsed(self):
        times = np.arange(100, dtype=float)
        times[10] = times[9]
        times[30] = np.nan
        times[50] = 48
        values = np.arange(100, dtype=float)
        self.dataset(values, times)
        row = self.inspect(replace(QUIET, jerk_limit=1e6), frame_gap_factor=4)['episodes'][0]
        self.assertIn(10, row['flagged_frames'])
        self.assertIn(30, row['flagged_frames'])
        self.assertIn(50, row['flagged_frames'])
        self.assertGreater(row['normal_frames'], 90)
        valid = np.isfinite(times[:-1]) & np.isfinite(times[1:]) & (np.diff(times) > 0)
        result = _derivatives(values[:, None], times, 'position', valid)
        for data in result.values():
            for index, (start, end) in enumerate(zip(data['starts'], data['ends'])):
                if not valid[start:end].all():
                    self.assertTrue(np.isnan(data['values'][index]).all())
        browser = BrowserModel(self.root).read_episode(0, validate_timestamps=False)
        self.assertTrue(np.isnan(browser.times[30]))
        self.assertEqual(browser.frame(30)['frame_index'], 30)

    def test_no_physical_differences_from_fps_when_timestamps_missing(self):
        values = np.arange(100, dtype=float)
        values[50] += 1e6
        self.dataset(values, None)
        row = self.inspect(replace(QUIET, velocity_zscore=2, acceleration_zscore=2, jerk_limit=.01))['episodes'][0]
        self.assertEqual(row['coverage']['timestamp_source'], 'missing')
        self.assertFalse(any(s['type'] in {'velocity', 'acceleration', 'jerk'} for s in row['segments']))
        self.assertIsNone(row['normal_segments'][0]['start_timestamp'])

    def test_boundaries_short_episode_and_no_cross_episode_difference(self):
        self.dataset(np.ones(1), eid=0)
        self.dataset(np.ones(1) * 1e9, eid=9)
        report = self.inspect(replace(QUIET, velocity_zscore=1, acceleration_zscore=1, jerk_limit=.001), ids=(0, 9))
        self.assertEqual(report['summary']['normal_frames'], 2)
        self.assertEqual(report['summary']['suspect_frames'], 0)
        self.assertTrue(all('state.jerk' in row['unverified_checks'] for row in report['episodes']))

    def test_broken_episode_is_unchecked_instead_of_rejecting_dataset(self):
        self.dataset(np.arange(10, dtype=float))
        (self.root / 'data/episode_000000.parquet').write_bytes(b'broken')
        row = self.inspect()['episodes'][0]
        self.assertEqual(row['status'], 'unreadable')
        self.assertEqual((row['normal_frames'], row['suspect_frames'], row['unchecked_frames']), (0, 0, 10))
        self.assertEqual(row['normal_segments'], [])

    def test_report_is_strict_json_and_outside_source(self):
        self.dataset(np.arange(20, dtype=float), np.r_[np.arange(19), np.inf])
        report = self.inspect()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'report.json'
            save_quality_report(report, output)
            saved = json.loads(output.read_text())
            self.assertEqual(saved['version'], 2)
            self.assertIsNone(saved['episodes'][0]['frame_flags'][-1]['timestamp'])
        with self.assertRaises(ValueError):
            save_quality_report(report, self.root / 'report.json')

    def test_native_velocity_and_acceleration_keep_signal_meaning(self):
        velocity = np.ones(40)
        velocity[10:16] = 0
        self.dataset(velocity)
        row = self.inspect(replace(QUIET, kind='velocity', static_min_frames=5))['episodes'][0]
        interval = next(s for s in row['segments'] if s['type'] == 'static' and s['status'] == 'suspect')
        self.assertEqual((interval['start_frame'], interval['end_frame']), (10, 15))
        self.assertEqual(interval['samples'][0]['values'], [0.])
        self.assertEqual(row['normal_frames'], 34)
        result = _derivatives((np.arange(10) * 2.)[:, None], np.arange(10.), 'acceleration')
        np.testing.assert_allclose(result[3]['values'], 2.)
        np.testing.assert_array_equal(result[3]['starts'], np.arange(9))
        np.testing.assert_array_equal(result[3]['ends'], np.arange(1, 10))

    def test_gap_is_flagged_locally_and_derivatives_do_not_bridge_it(self):
        times = np.arange(100, dtype=float)
        times[40:] += 100
        self.dataset(times, times)
        row = self.inspect(replace(QUIET, velocity_zscore=6, acceleration_zscore=6, jerk_limit=1))['episodes'][0]
        self.assertEqual(row['flagged_frames'], [40])
        self.assertEqual(row['normal_frames'], 99)
        velocity = next(c for c in row['checks'] if c['name'] == 'state.velocity_zscore')
        self.assertFalse(any(a <= 40 <= b for a, b in velocity['evaluated_frame_ranges']))
        self.assertEqual(velocity['metrics']['peak_abs_per_dim'], [1.])

    def test_native_acceleration_has_no_static_inference(self):
        self.dataset(np.arange(20, dtype=float) * 2)
        row = self.inspect(replace(QUIET, kind='acceleration', jerk_limit=1))['episodes'][0]
        self.assertIn('state.static', row['unverified_checks'])
        self.assertIn('state.velocity_zscore', row['unverified_checks'])
        jerk = next(s for s in row['segments'] if s['type'] == 'jerk')
        self.assertEqual(jerk['start_frame'], 1)
        self.assertEqual(jerk['samples'][0]['values'], [2.])

    def test_image_decode_blur_and_multiview_sync_remain_local(self):
        self.dataset(np.arange(20, dtype=float))
        info_path = self.root / 'meta/info.json'
        info = json.loads(info_path.read_text())
        for key in ('image', 'wrist_image'):
            info['features'][key] = dict(dtype='image', shape=[3, 8, 8])
        for key in ('image_time', 'wrist_time'):
            info['features'][key] = dict(dtype='float64', shape=[1])
        info_path.write_text(json.dumps(info))
        pixels = ((np.indices((8, 8)).sum(axis=0) % 2) * 255).astype(np.uint8)
        stream = BytesIO()
        Image.fromarray(pixels).convert('RGB').save(stream, format='PNG')
        sharp = stream.getvalue()
        stream = BytesIO()
        Image.new('RGB', (8, 8), (120, 80, 40)).save(stream, format='PNG')
        blurry = stream.getvalue()
        path = self.root / 'data/episode_000000.parquet'
        table = pq.read_table(path)
        first = [dict(bytes=b'broken' if i == 5 else blurry if i == 7 else sharp, path=None) for i in range(20)]
        second = [dict(bytes=sharp, path=None)] * 20
        first_time = np.arange(20, dtype=float)
        first_time[3:5] += .3
        for key, values in [('image', first), ('wrist_image', second), ('image_time', first_time), ('wrist_time', np.arange(20.))]:
            table = table.append_column(key, pa.array(values))
        pq.write_table(table, path)
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        policy = QualityPolicy(state=QUIET, action=replace(QUIET, kind='command'), image_width=8,
                               blur_variance_min=10, repeated_run=999,
                               camera_timestamp_keys={'image': 'image_time', 'wrist_image': 'wrist_time'})
        row = QualityInspector(self.root).inspect([0], policy)['episodes'][0]
        checks = {check['name']: check for check in row['checks']}
        self.assertEqual(checks['image_integrity:image']['frames'], [5])
        self.assertEqual(checks['blur:image']['frames'], [7])
        self.assertEqual(checks['video_state_sync:image']['frames'], [3, 4])
        self.assertEqual(checks['multiview_time_sync']['frames'], [3, 4])
        self.assertEqual(row['flagged_frames'], [3, 4, 5, 7])
        self.assertEqual(row['normal_frames'], 16)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
