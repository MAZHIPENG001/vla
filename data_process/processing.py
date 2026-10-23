"""Frame / Segment 级异常检测。只生成标记和报告，不筛除或改写源数据。"""

from collections import OrderedDict
from concurrent.futures import CancelledError
from dataclasses import asdict
from hashlib import blake2b
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import QUALITY_POLICY, QualityPolicy
from .dataset import BrowserModel, json_value


def _threshold(value, width, dimensions):
    if value is None:
        return None
    try:
        return np.broadcast_to(np.asarray(value, dtype=float), (width,))[dimensions]
    except ValueError as exc:
        raise ValueError(f"逐维阈值需要 1 个或 {width} 个数值") from exc


def _groups(indices):
    """相邻的原始行索引合并；不会跨缺失的帧或 Episode。"""
    ids = sorted(set(int(i) for i in indices))
    if not ids:
        return []
    result, start, end = [], ids[0], ids[0]
    for index in ids[1:]:
        if index != end + 1:
            result.append((start, end))
            start = index
        end = index
    return result + [(start, end)]


def _frame_ranges(indices):
    return [[start, end] for start, end in _groups(indices)]


def _derivatives(values, times, kind, valid_edges=None):
    """差分用真实时间和中点时刻；每个样本保留完整的原始支撑窗口。

    无效值逐维传播；无效时间边和采样缺口使该边上的窗口无效，不拼接两侧数据。
    """
    order = {"position": 0, "command": 0, "velocity": 1, "acceleration": 2}[kind]
    starts = ends = np.arange(len(times))
    result = {}
    if valid_edges is None:
        valid_edges = np.isfinite(times[:-1]) & np.isfinite(times[1:]) & (np.diff(times) > 0)
    invalid_prefix = np.r_[0, np.cumsum(~valid_edges)]
    while order <= 3 and len(values):
        if order:
            result[order] = dict(values=values, times=times, starts=starts, ends=ends)
        if len(values) < 2:
            break
        dt = np.diff(times)
        starts, ends = starts[:-1], ends[1:]
        edge_ok = (invalid_prefix[ends] - invalid_prefix[starts]) == 0
        valid = (np.isfinite(values[:-1]) & np.isfinite(values[1:])
                 & np.isfinite(dt[:, None]) & (dt[:, None] > 0) & edge_ok[:, None])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = np.divide(np.diff(values, axis=0), dt[:, None],
                               out=np.full_like(values[:-1], np.nan), where=valid)
            times = (times[:-1] + times[1:]) / 2
        order += 1
    return result


def _check_type(name):
    if name.endswith('.static'):
        return 'static'
    if name.endswith('.velocity_zscore'):
        return 'velocity'
    if name.endswith('.acceleration_zscore'):
        return 'acceleration'
    if name.endswith('.jerk'):
        return 'jerk'
    if name.startswith(('video_state_', 'multiview_', 'camera_clock:')):
        return 'sync'
    if name.startswith(('image', 'blur:', 'occlusion_', 'repeated_', 'semantic_', 'external_video')):
        return 'image'
    return 'integrity'


REPORT_TYPES = [('integrity', '完整性 / 时间戳'), ('static', '长静止'), ('velocity', '速度'),
                ('acceleration', '加速度'), ('jerk', 'Jerk'), ('image', '图像'), ('sync', '时间同步')]


def episode_summary(row, *, complete=True):
    """列表/曲线使用的小型索引；逐帧值和检查明细留在单条 Episode 文件中。"""
    result = {key: value for key, value in row.items() if key not in {'checks', 'frame_flags', 'flagged_frames', 'segments'}}
    result['segments'] = [{key: value for key, value in segment.items() if key not in {'metrics', 'samples'}}
                          for segment in row['segments']]
    result['episode_complete'] = complete
    result['type_statistics'] = []
    for kind, label in REPORT_TYPES:
        segments = [s for s in row['segments'] if s['type'] == kind and s['status'] == 'suspect']
        frames = {frame for s in segments for frame in range(s['start_frame'], s['end_frame'] + 1)}
        result['type_statistics'].append(dict(type=kind, stage=label, segments=len(segments),
                                              affected_frames=len(frames), affected_episodes=int(bool(frames))))
    return json_value(result)


def build_report(source, policy, ids, rows, complete, *, include_episodes=True):
    summary = {key: sum(row[key] for row in rows) for key in ('normal_frames', 'suspect_frames', 'unchecked_frames')}
    summary.update(checked_episodes=sum(row.get('episode_complete', True) for row in rows), total_requested=len(ids),
                   checked_frames=sum(row['length'] - row['unchecked_frames'] for row in rows),
                   total_frames=sum(row['length'] for row in rows),
                   episodes_with_findings=sum(row['suspect_frames'] > 0 for row in rows),
                   episodes_with_unverified=sum(bool(row['unverified_checks']) for row in rows),
                   anomaly_segments=sum(sum(s['status'] == 'suspect' for s in row['segments']) for row in rows),
                   waiting_segments=sum(sum(s['status'] == 'waiting' for s in row['segments']) for row in rows))
    statistics = [row.get('type_statistics') or episode_summary(row)['type_statistics'] for row in rows]
    summary['by_type'] = [dict(type=kind, stage=label,
        **{key: sum(items[index][key] for items in statistics)
           for key in ('segments', 'affected_frames', 'affected_episodes')})
        for index, (kind, label) in enumerate(REPORT_TYPES)]
    return json_value(dict(
        version=2, source=str(source), policy=asdict(policy) if isinstance(policy, QualityPolicy) else policy,
        scope_episode_ids=list(ids), complete=complete, selection_unit='frame_segment',
        source_modified=False, dataset_exported=False,
        indexing=dict(frame_index='zero_based_row_offset', source_frame_index='original frame_index column',
                      timestamp='original timestamp; null if missing/invalid',
                      derivative_anchor='rightmost source frame; window and midpoint time recorded per sample'),
        normal_definition='未被本轮已执行检查标记的可读帧，不代表未验证指标已通过', summary=summary,
        episodes=rows if include_episodes else []))


class QualityInspector:
    """独立读取实例和小型图像指标缓存；检测、区间选择、数据导出互相独立。"""

    def __init__(self, path, *, state_key=None, action_key=None):
        self.model = BrowserModel(path, state_key=state_key, action_key=action_key)
        self._image_cache = OrderedDict()
        self._episode_tasks = {}
        for task in self.model.task_stats:
            for eid in task.episode_ids:
                self._episode_tasks.setdefault(eid, []).append(task.name)

    def validate_policy(self, policy: QualityPolicy) -> None:
        clocks = list(policy.camera_timestamp_keys.values())
        if policy.state_timestamp_key in clocks or len(clocks) != len(set(clocks)):
            raise ValueError("同步检查需要各传感器独立的采集时间戳字段，不能重复映射公共 timestamp")
        for role in ("state", "action"):
            field = getattr(self.model.schema, role)
            rule = getattr(policy, role)
            dimensions = list(rule.dimensions) if rule.dimensions is not None else list(range(field.width))
            if field.key and any(i >= field.width for i in dimensions):
                raise ValueError(f"{role} 仅有 {field.width} 维，维度选择越界")
            for value in (rule.static_epsilon, rule.jerk_limit):
                if field.key:
                    _threshold(value, field.width, dimensions)
        for camera, key in policy.camera_timestamp_keys.items():
            if camera not in self.model.loader.camera_keys:
                raise ValueError(f"未知相机: {camera}")
            feature = self.model.loader.features.get(key)
            if feature is None or feature["dtype"] in {"image", "video"}:
                raise ValueError(f"相机 {camera} 的时间戳字段 {key!r} 不存在或不是数值字段")

    @staticmethod
    def _cancel(cancel):
        if cancel is not None and cancel.is_set():
            raise CancelledError()

    def inspect(self, episode_ids, policy: QualityPolicy = QUALITY_POLICY, *, progress=None, cancel=None,
                on_episode=None, on_update=None, retain_episodes=True):
        self.validate_policy(policy)
        ids = list(dict.fromkeys(int(i) for i in episode_ids))
        if not ids:
            raise ValueError("检查范围没有 Episode")
        if any(i not in self.model.loader.episodes for i in ids):
            raise ValueError("检查范围包含不存在的 Episode")
        rows, cancelled = [], False
        for index, eid in enumerate(ids):
            try:
                self._cancel(cancel)
                row = self._episode(eid, policy, cancel, on_update)
                if on_episode:
                    on_episode(json_value(row))
                rows.append(row if retain_episodes else episode_summary(row))
            except CancelledError:
                cancelled = True
                break
            if progress:
                progress(index + 1, len(ids), eid)
        return build_report(self.model.loader.path, policy, ids, rows,
                            not cancelled and len(rows) == len(ids))

    @staticmethod
    def _point(episode, frame):
        timestamp = episode.times[frame] if episode is not None else None
        source = None
        if episode is not None and 'frame_index' in episode.values:
            indices = np.asarray(episode.values['frame_index']).reshape(-1)
            if len(indices) == episode.length:
                source = indices[frame]
        point = dict(frame_index=int(frame), source_frame_index=json_value(source), timestamp=json_value(timestamp))
        if episode is not None and 'timestamp' in episode.values and not np.isfinite(timestamp):
            point['timestamp_raw'] = str(timestamp)
        return point

    def _range(self, episode, eid, start, end):
        first, last = self._point(episode, start), self._point(episode, end)
        t0, t1 = first['timestamp'], last['timestamp']
        return dict(episode_index=eid, start_frame=int(start), end_frame=int(end), frame_count=int(end - start + 1),
                    source_start_frame_index=first['source_frame_index'], source_end_frame_index=last['source_frame_index'],
                    start_timestamp=t0, end_timestamp=t1,
                    duration_seconds=float(t1 - t0) if t0 is not None and t1 is not None and t1 >= t0 else None)

    def _episode(self, eid, policy, cancel, on_update=None):
        metadata = self.model.loader.episodes[eid]
        checks, all_segments = [], []
        episode = None
        unreadable = False
        timestamp_source = 'missing'

        def add(name, status, metrics=None, frames=(), message='', *, segments=None, samples=None,
                coverage=None, dimensions=()):
            frames = sorted({int(i) for i in frames if 0 <= int(i) < metadata['length']})
            # 检查旧的图像/同步分支也转换为局部标记；不产生 Episode 拒绝决定。
            status = 'flagged' if frames and status not in {'unverified', 'disabled'} else {
                'review': 'unverified'}.get(status, status)
            kind = _check_type(name)
            if segments is None:
                segments = []
                if status == 'flagged':
                    for start, end in _groups(frames):
                        segment_samples = [s for s in (samples or []) if start <= s.get('frame_index', s.get('frame', -1)) <= end]
                        segment_dimensions = sorted({d for sample in segment_samples for d in sample.get('dimensions', [])}) if segment_samples else list(dimensions)
                        segments.append(dict(self._range(episode, eid, start, end), type=kind, status='suspect',
                                             check=name, dimensions=segment_dimensions, metrics=metrics or {}, samples=segment_samples))
            for segment in segments:
                segment['id'] = f"{eid}:{name}:{segment['start_frame']}:{segment['end_frame']}"
                segment.setdefault('check', name)
                segment.setdefault('role', name.split('.')[0] if name.startswith(('state.', 'action.')) else None)
            all_segments.extend(segments)
            if coverage is None:
                coverage = range(metadata['length']) if episode is not None and status not in {'unverified', 'disabled'} else []
            checks.append(dict(name=name, type=kind, status=status, metrics=metrics or {}, frames=frames,
                               dimensions=list(dimensions), segment_ids=[s['id'] for s in segments], message=message,
                               evaluated_frame_ranges=_frame_ranges(coverage)))
            if on_update and any(segment['status'] == 'suspect' for segment in segments):
                snapshot = json_value(finish())
                snapshot['episode_complete'] = False
                snapshot['phase'] = name
                on_update(snapshot)


        def finish():
            suspect = sorted({frame for s in all_segments if s['status'] == 'suspect'
                              for frame in range(s['start_frame'], s['end_frame'] + 1)})
            unchecked = list(range(metadata['length'])) if unreadable else []
            normal = sorted(set(range(metadata['length'])) - set(suspect) - set(unchecked))
            flags = []
            for frame in suspect:
                segments = [s for s in all_segments if s['status'] == 'suspect' and s['start_frame'] <= frame <= s['end_frame']]
                dimensions = {}
                for s in segments:
                    point = next((p for p in s.get('samples', []) if p.get('frame_index') == frame), None)
                    if s.get('role'):
                        dimensions.setdefault(s['role'], set()).update(point['dimensions'] if point else s['dimensions'])
                flags.append(dict(self._point(episode, frame), types=sorted({s['type'] for s in segments}),
                                  segment_ids=[s['id'] for s in segments],
                                  dimensions={key: sorted(value) for key, value in dimensions.items()}))
            unverified = [c['name'] for c in checks if c['status'] in {'unverified', 'disabled'}]
            return dict(episode_index=eid, tasks=self._episode_tasks.get(eid, []), length=metadata['length'],
                        status='unreadable' if unreadable else 'flagged' if suspect else 'review' if unverified else 'clear',
                        normal_frames=len(normal), suspect_frames=len(suspect), unchecked_frames=len(unchecked),
                        coverage=dict(readable=not unreadable, timestamp_source=timestamp_source,
                                      inspected_frame_ranges=[] if unreadable else [[0, metadata['length'] - 1]],
                                      note='各指标的差分边界、无效窗口和未验证项见 checks'),
                        flagged_frames=suspect, frame_flags=flags, unverified_checks=unverified,
                        segments=all_segments,
                        normal_segments=[self._range(episode, eid, a, b) for a, b in _groups(normal)],
                        suspect_segments=[self._range(episode, eid, a, b) for a, b in _groups(suspect)],
                        unchecked_segments=[self._range(episode, eid, a, b) for a, b in _groups(unchecked)], checks=checks)

        try:
            episode = self.model.read_episode(eid, validate_timestamps=False)
        except Exception as exc:
            unreadable = True
            add('data_integrity', 'unverified', message=str(exc))
            return finish()
        if 'timestamp' not in episode.values:
            episode.times = np.full(episode.length, np.nan)
            add('timestamp_source', 'unverified', message='缺少真实时间戳，不使用 FPS 推算运动导数；仍检查数值和静止帧数')
        else:
            timestamp_source = 'recorded'
        times = episode.times
        dt = np.diff(times)
        finite_time = np.isfinite(times)
        invalid = sorted(set(np.flatnonzero(~finite_time).tolist()) | set((np.flatnonzero(~np.isfinite(dt) | (dt <= 0)) + 1).tolist()))
        if invalid:
            timestamp_source = 'invalid' if 'timestamp' in episode.values else 'missing'
        if 'timestamp' in episode.values:
            add('timestamps', 'flagged' if invalid else 'passed',
                metrics=dict(expected_frames=episode.length,
                             raw_shape=list(np.asarray(episode.values['timestamp']).shape)), frames=invalid,
                samples=[dict(self._point(episode, frame),
                              previous_timestamp=json_value(times[frame - 1]) if frame else None,
                              interval_seconds=json_value(dt[frame - 1]) if frame else None) for frame in invalid],
                message='无效时间边上的差分窗口被跳过，不跨越重复、倒退或非有限时间戳')
        positive = dt[np.isfinite(dt) & (dt > 0)]
        gap_limit = float(np.median(positive) * policy.frame_gap_factor) if len(positive) else None
        gaps = np.flatnonzero(np.isfinite(dt) & (dt > gap_limit)) + 1 if gap_limit is not None else np.array([], dtype=int)
        add('frame_gaps', 'flagged' if len(gaps) else 'passed' if len(positive) else 'unverified',
            dict(gap_limit_seconds=gap_limit, reference='median_positive_recorded_interval'), gaps,
            '采样缺口单独标记；差分窗口不跨越缺口',
            samples=[dict(self._point(episode, int(frame)), interval_seconds=float(dt[frame - 1]), threshold=gap_limit) for frame in gaps])
        valid_edges = finite_time[:-1] & finite_time[1:] & (dt > 0)
        if len(gaps):
            valid_edges[gaps - 1] = False
        if 'frame_index' in episode.values:
            indices = np.asarray(episode.values['frame_index']).reshape(-1)
            if len(indices) == episode.length:
                bad = sorted(set(np.flatnonzero(~np.isfinite(indices) | (indices != np.floor(indices))).tolist())
                             | set((np.flatnonzero(np.diff(indices) != 1) + 1).tolist()))
            else:
                bad = list(range(episode.length))
            add('frame_indices', 'flagged' if len(bad) else 'passed', frames=bad,
                message='原始 frame_index 原样保留，定位使用独立的 0 起始行索引')
        for role in ('state', 'action'):
            self._cancel(cancel)
            self._signal(episode, role, getattr(policy, role), add, valid_edges, 'timestamp' in episode.values)
        self._cancel(cancel)
        if policy.inspect_images:
            self._images(episode, policy, add, cancel)
        else:
            add('images', 'unverified', message='本轮未检查图像')
        self._synchronization(episode, policy, add)
        add('semantic_occlusion', 'unverified', message='未运行语义遮挡模型；像素代理指标仅供复核')
        return finish()

    def _signal(self, episode, role, rule, add, valid_edges, recorded_time):
        field, array = getattr(episode.schema, role), getattr(episode, role)
        if array is None:
            add(f'{role}.mapping', 'unverified', message=f'请指定 {role} 字段')
            return
        dimensions = list(rule.dimensions) if rule.dimensions is not None else list(range(field.width))
        values = array[:, dimensions].astype(np.float64)
        finite = np.isfinite(values)
        invalid = np.flatnonzero(~finite.all(axis=1))
        samples = [dict(self._point(episode, int(frame)), dimensions=[dimensions[d] for d in np.flatnonzero(~finite[frame])],
                        values=json_value(values[frame, ~finite[frame]]), reason='NaN / Inf') for frame in invalid]
        add(f'{role}.finite', 'flagged' if len(invalid) else 'passed', frames=invalid, samples=samples,
            dimensions=dimensions, message='无效数值仅使包含该值的对应维度差分窗口失效')
        self._static(episode, role, rule, dimensions, values, valid_edges, recorded_time, add)
        derivatives = _derivatives(values, episode.times, rule.kind, valid_edges)
        for order, name, threshold in [(1, 'velocity_zscore', rule.velocity_zscore),
                                       (2, 'acceleration_zscore', rule.acceleration_zscore), (3, 'jerk', rule.jerk_limit)]:
            data = derivatives.get(order)
            if data is None:
                add(f'{role}.{name}', 'unverified', message='信号类型不提供此量或帧数不足', dimensions=dimensions)
                continue
            samples = data['values']
            finite = np.isfinite(samples)
            counts = finite.sum(axis=0)
            metrics = dict(kind=rule.kind, dimensions=dimensions, sample_count=len(samples), valid_samples_per_dim=counts.tolist(),
                           peak_abs_per_dim=[float(np.abs(samples[finite[:, d], d]).max()) if counts[d] else None for d in range(len(dimensions))],
                           interpretation='指令序列的时间差分，不自动解释为物理运动' if rule.kind == 'command' else '按配置信号含义解释')
            overflow = np.flatnonzero(np.isinf(samples).any(axis=1))
            if len(overflow):
                add(f'{role}.{name}.nonfinite', 'flagged', frames=data['ends'][overflow], dimensions=dimensions,
                    samples=[dict(self._point(episode, int(data['ends'][index])),
                                  dimensions=[dimensions[d] for d in np.flatnonzero(np.isinf(samples[index]))],
                                  values=json_value(samples[index, np.isinf(samples[index])]),
                                  window_start_frame=int(data['starts'][index]), window_end_frame=int(data['ends'][index]),
                                  reason='derivative_overflow') for index in overflow],
                    message='差分发生数值溢出；未参与 Z-score 参考分布')
            if threshold is None:
                add(f'{role}.{name}', 'unverified' if order == 3 else 'disabled', metrics,
                    dimensions=dimensions, coverage=[], message='未配置阈值，不标记为异常')
                continue
            scores = np.full_like(samples, np.nan)
            if order == 3:
                limits = _threshold(threshold, field.width, dimensions)
                bad = finite & (np.abs(samples) > limits)
                eligible = finite
                metrics['threshold_per_dim'] = limits.tolist()
            else:
                eligible = finite & (counts[None, :] >= 4)
                means, deviations = [], []
                for d in range(len(dimensions)):
                    if counts[d] >= 4:
                        column = samples[finite[:, d], d]
                        # 先缩放以避免极端有限值在均值/方差计算时溢出；Z-score 不受缩放影响。
                        scale = float(np.abs(column).max()) or 1.
                        column = column / scale
                        std = column.std()
                        scores[finite[:, d], d] = np.abs(column - column.mean()) / std if std > 0 else 0.
                        means.append(float(column.mean() * scale))
                        deviations.append(float(std * scale))
                    else:
                        means.append(None)
                        deviations.append(None)
                bad = eligible & (scores > threshold)
                metrics.update(threshold=threshold, reference='episode_per_dimension',
                               reference_mean_per_dim=means, reference_std_per_dim=deviations,
                               insufficient_dimensions=[dimensions[d] for d in range(len(dimensions)) if counts[d] < 4])
            coverage = data['ends'][eligible.any(axis=1)]
            metrics['skipped_frame_ranges'] = _frame_ranges(set(range(episode.length)) - set(coverage.tolist()))
            metrics['coverage_per_dimension'] = [dict(
                dimension=dimensions[d],
                evaluated_frame_ranges=_frame_ranges(data['ends'][eligible[:, d]]),
                skipped_frame_ranges=_frame_ranges(set(range(episode.length)) - set(data['ends'][eligible[:, d]].tolist()))
            ) for d in range(len(dimensions))]
            points = []
            for sample_index in np.flatnonzero(bad.any(axis=1)):
                affected = np.flatnonzero(bad[sample_index])
                start, end = int(data['starts'][sample_index]), int(data['ends'][sample_index])
                points.append(dict(self._point(episode, end), window_start_frame=start, window_end_frame=end,
                                   window_start_timestamp=json_value(episode.times[start]), window_end_timestamp=json_value(episode.times[end]),
                                   metric_timestamp=json_value(data['times'][sample_index]),
                                   dimensions=[dimensions[d] for d in affected],
                                   values=samples[sample_index, affected].tolist(),
                                   zscores=scores[sample_index, affected].tolist() if order != 3 else None,
                                   thresholds=limits[affected].tolist() if order == 3 else [threshold] * len(affected)))
            segments = []
            for start, end in _groups(p['frame_index'] for p in points):
                selected = [p for p in points if start <= p['frame_index'] <= end]
                segments.append(dict(self._range(episode, episode.index, start, end), type=_check_type(f'{role}.{name}'),
                                     status='suspect', role=role, dimensions=sorted({d for p in selected for d in p['dimensions']}),
                                     window_start_frame=min(p['window_start_frame'] for p in selected), window_end_frame=end,
                                     metrics=metrics, samples=selected))
            incomplete = order != 3 and any(counts < 4)
            add(f'{role}.{name}', 'flagged' if points else 'passed' if len(coverage) and not incomplete else 'unverified', metrics,
                [p['frame_index'] for p in points], segments=segments, dimensions=dimensions, coverage=coverage,
                message='异常以差分窗口最右帧定位；边界和无效窗口不参与参考分布。局部异常不拒绝 Episode')

    def _static(self, episode, role, rule, dimensions, values, valid_edges, recorded_time, add):
        epsilon = _threshold(rule.static_epsilon, getattr(episode.schema, role).width, dimensions)
        valid = np.isfinite(values).all(axis=1)
        if rule.kind in {'position', 'command'}:
            with np.errstate(over='ignore', invalid='ignore'):
                delta = np.abs(np.diff(values, axis=0))
            allowed_edges = valid_edges if recorded_time else np.ones(max(0, episode.length - 1), dtype=bool)
            mask = valid[:-1] & valid[1:] & allowed_edges & np.all(delta <= epsilon, axis=1)
            runs = [(a - 1, b) for a, b in _groups(np.flatnonzero(mask) + 1)]
            coverage = np.flatnonzero(valid[:-1] & valid[1:] & allowed_edges) + 1
        elif rule.kind == 'velocity':
            delta = np.abs(values)
            mask = valid & np.all(delta <= epsilon, axis=1)
            if recorded_time:
                mask &= np.isfinite(episode.times)
            runs = []
            for a, b in _groups(np.flatnonzero(mask)):
                breaks = np.flatnonzero(~valid_edges[a:b]) + a + 1 if recorded_time else []
                starts, ends = [a, *breaks], [*(i - 1 for i in breaks), b]
                runs.extend(zip(starts, ends))
            coverage = np.flatnonzero(valid)
        else:
            add(f'{role}.static', 'unverified', message='加速度本身不能推出是否静止', dimensions=dimensions)
            return
        segments, flagged = [], []
        for start, end in runs:
            segment = self._range(episode, episode.index, start, end)
            duration = segment['duration_seconds'] if recorded_time else None
            long_frames = rule.static_min_frames is not None and segment['frame_count'] >= rule.static_min_frames
            long_time = rule.static_min_seconds is not None and duration is not None and duration >= rule.static_min_seconds
            suspect = long_frames or long_time
            selected = delta[start:end] if rule.kind in {'position', 'command'} else delta[start:end + 1]
            segment.update(type='static', status='suspect' if suspect else 'waiting', role=role, dimensions=dimensions,
                           metrics=dict(kind=rule.kind, epsilon_per_dim=epsilon.tolist(),
                                        max_abs_change_per_dim=np.max(selected, axis=0).tolist() if len(selected) else [],
                                        static_min_frames=rule.static_min_frames, static_min_seconds=rule.static_min_seconds,
                                        trigger='frames_or_duration', duration_verified=recorded_time and duration is not None), samples=[])
            if suspect:
                is_delta = rule.kind in {'position', 'command'}
                for frame in range(start + 1 if is_delta else start, end + 1):
                    segment['samples'].append(dict(
                        self._point(episode, frame), dimensions=dimensions,
                        metric='absolute_delta' if is_delta else 'velocity',
                        values=delta[frame - 1].tolist() if is_delta else values[frame].tolist(),
                        thresholds=epsilon.tolist(), window_start_frame=frame - 1 if is_delta else frame,
                        window_end_frame=frame))
            segments.append(segment)
            if suspect:
                flagged.extend(range(start, end + 1))
        calibrated = rule.static_min_frames is not None or (rule.static_min_seconds is not None and recorded_time and valid_edges.any())
        add(f'{role}.static', 'flagged' if flagged else 'passed' if calibrated and len(coverage) else 'unverified',
            dict(dimensions=dimensions, kind=rule.kind, epsilon_per_dim=epsilon.tolist(),
                 waiting_intervals=sum(s['status'] == 'waiting' for s in segments),
                 long_static_intervals=sum(s['status'] == 'suspect' for s in segments),
                 static_min_frames=rule.static_min_frames, static_min_seconds=rule.static_min_seconds),
            flagged, segments=segments, coverage=coverage, dimensions=dimensions,
            message='整组选中维度静止；帧数或真实持续时间达到阈值时标记长静止，其余作为正常等待记录')

    def _image_measurements(self, episode, policy, cancel):
        cache_key = (episode.index, policy.image_stride, policy.image_width)
        if cache_key in self._image_cache:
            self._image_cache.move_to_end(cache_key)
            return self._image_cache[cache_key]
        cameras = self.model.camera_keys
        samples = sorted(set(range(0, episode.length, policy.image_stride)) | {episode.length - 1})
        result = {camera: [] for camera in cameras}
        for index in samples:
            self._cancel(cancel)
            images, errors = self.model.loader.read_image_frame(episode.index, index, cameras)
            for camera in cameras:
                entry = {"frame": index}
                try:
                    if camera in errors:
                        raise ValueError(errors[camera])
                    array = images[camera]
                    image = Image.fromarray(array.transpose(1, 2, 0)).convert("L")
                    image = image.resize((policy.image_width, max(8, round(image.height * policy.image_width / image.width))),
                                         Image.Resampling.BILINEAR)
                    gray = np.asarray(image, dtype=float)
                    laplace = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
                    entry.update(blur_variance=float(laplace.var()), std=float(gray.std()),
                                 saturation=max(float((gray < 5).mean()), float((gray > 250).mean())),
                                 fingerprint=blake2b(image.tobytes(), digest_size=8).hexdigest())
                except Exception as exc:
                    entry["error"] = str(exc)
                result[camera].append(entry)
        self._image_cache[cache_key] = result
        if len(self._image_cache) > 16:
            self._image_cache.popitem(last=False)
        return result

    def _images(self, episode, policy, add, cancel):
        if self.model.video_keys:
            add("external_video", "unverified", message="当前读取器尚未支持外部视频解码")
        if not self.model.camera_keys:
            add("images", "unverified", message="没有可检查的内嵌图像")
            return
        try:
            measures = self._image_measurements(episode, policy, cancel)
        except CancelledError:
            raise
        except Exception as exc:
            add("image_integrity", "unverified", message=str(exc))
            return
        if policy.image_stride > 1:
            add("image_coverage", "unverified", {"stride": policy.image_stride}, message="抽样检查不能覆盖未采样的帧")
        state = episode.state
        movement = np.zeros(episode.length, dtype=bool)
        if state is not None and np.isfinite(state).all():
            dims = list(policy.state.dimensions) if policy.state.dimensions is not None else list(range(state.shape[1]))
            epsilon = _threshold(policy.state.static_epsilon, state.shape[1], dims)
            if policy.state.kind == "velocity":
                movement = np.any(np.abs(state[:, dims]) > epsilon, axis=1)
            elif policy.state.kind in {"position", "command"}:
                movement[1:] = np.any(np.abs(np.diff(state[:, dims], axis=0)) > epsilon, axis=1)
        for camera, samples in measures.items():
            errors = [s["frame"] for s in samples if "error" in s]
            valid = [s for s in samples if "error" not in s]
            add(f"image_integrity:{camera}", "flagged" if errors else "passed",
                {"sampled": len(samples), "total": episode.length,
                 "errors": [s for s in samples if "error" in s]}, errors,
                samples=[dict(self._point(episode, s["frame"]), error=s["error"]) for s in samples if "error" in s],
                coverage=[s["frame"] for s in samples])
            if not valid:
                continue
            scores = np.array([s["blur_variance"] for s in valid])
            metrics = {"min": float(scores.min()), "median": float(np.median(scores)),
                       "resize_width": policy.image_width, "sampled": len(valid)}
            if policy.blur_variance_min is None:
                add(f"blur:{camera}", "unverified", metrics, message="请先根据场景与固定缩放尺寸校准清晰度阈值")
            else:
                bad = [s["frame"] for s in valid if s["blur_variance"] < policy.blur_variance_min]
                ratio = len(bad) / len(valid)
                metrics.update(bad_ratio=ratio, threshold=policy.blur_variance_min, ratio_reference=policy.max_bad_image_ratio)
                add(f"blur:{camera}", "flagged" if bad else "passed", metrics, bad,
                    samples=[dict(self._point(episode, s["frame"]), blur_variance=s["blur_variance"], threshold=policy.blur_variance_min) for s in valid if s["frame"] in bad],
                    coverage=[s["frame"] for s in valid])
            uniform = [s["frame"] for s in valid if s["std"] < policy.uniform_std_max and s["saturation"] >= policy.saturation_fraction]
            add(f"occlusion_proxy:{camera}", "review" if uniform else "passed",
                {"count": len(uniform), "std_limit": policy.uniform_std_max, "saturation_limit": policy.saturation_fraction}, uniform,
                "近乎纯黑/纯白低纹理帧，只作为遮挡或曝光异常线索",
                samples=[dict(self._point(episode, s["frame"]), std=s["std"], saturation=s["saturation"]) for s in valid if s["frame"] in uniform],
                coverage=[s["frame"] for s in valid])
            repeated = []
            run = []
            for sample in samples + [{"frame": episode.length}]:
                if run and ("fingerprint" not in sample or sample["fingerprint"] != run[-1]["fingerprint"]):
                    if len(run) >= policy.repeated_run and movement[run[0]["frame"]:run[-1]["frame"]+1].any():
                        repeated.extend(s["frame"] for s in run)
                    run = []
                if "fingerprint" in sample:
                    run.append(sample)
            add(f"repeated_image_proxy:{camera}", "review" if repeated else "passed",
                {"minimum_run": policy.repeated_run}, repeated,
                "机器人有变化时画面重复，仅提示疑似冻结；不能等同于掉帧", coverage=[s["frame"] for s in samples])

    def _synchronization(self, episode, policy, add):
        clocks = {}
        state_time = episode.values.get(policy.state_timestamp_key)
        if state_time is not None:
            state_time = np.asarray(state_time).reshape(-1)
        for camera in self.model.loader.camera_keys:
            key = policy.camera_timestamp_keys.get(camera)
            if not key:
                add(f"video_state_sync:{camera}", "unverified", message="没有独立的相机采集时间戳，同行存储不构成同步证据")
                continue
            times = np.asarray(episode.values[key], dtype=float).reshape(-1)
            if len(times) != episode.length or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                bad = sorted(set(np.flatnonzero(~np.isfinite(times)).tolist()) | set((np.flatnonzero(~np.isfinite(np.diff(times)) | (np.diff(times) <= 0)) + 1).tolist())) if len(times) == episode.length else list(range(episode.length))
                add(f"camera_clock:{camera}", "flagged", frames=bad, message="相机时间戳数量、有限性或递增性无效")
                continue
            clocks[camera] = times
            if state_time is None or len(state_time) != episode.length or not np.isfinite(state_time).all() or np.any(np.diff(state_time) <= 0):
                add(f"video_state_sync:{camera}", "unverified", message="缺少有效的状态采集时间戳")
            else:
                delta = np.abs(times - state_time)
                bad = np.flatnonzero(delta > policy.sync_tolerance_seconds)
                add(f"video_state_sync:{camera}", "flagged" if len(bad) else "passed",
                    {"max_skew_seconds": float(delta.max()), "limit_seconds": policy.sync_tolerance_seconds}, bad,
                    "要求配置的时间戳处于同一时钟域",
                    samples=[dict(self._point(episode, int(i)), skew_seconds=float(delta[i]), threshold=policy.sync_tolerance_seconds) for i in bad])
        if len(clocks) >= 2:
            stack = np.array(list(clocks.values()))
            skew = stack.max(axis=0) - stack.min(axis=0)
            bad = np.flatnonzero(skew > policy.sync_tolerance_seconds)
            add("multiview_time_sync", "flagged" if len(bad) else "passed",
                {"max_skew_seconds": float(skew.max()), "cameras": list(clocks), "threshold": policy.sync_tolerance_seconds}, bad,
                samples=[dict(self._point(episode, int(i)), skew_seconds=float(skew[i]), threshold=policy.sync_tolerance_seconds) for i in bad])
        if len(self.model.loader.camera_keys) > 1 and len(clocks) != len(self.model.loader.camera_keys):
            add("multiview_time_coverage", "unverified", message="没有覆盖全部相机的独立采集时间戳")
        if len(self.model.loader.camera_keys) > 1:
            add("multiview_geometry", "unverified", message="未提供外参/共同视觉事件，未验证空间一致性")


def save_quality_report(report, path):
    """保存完整阈值快照、逐帧异常索引和检查覆盖；不写入原始数据集。"""
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(Path(report["source"]).resolve()):
        raise ValueError("请将质量报告保存到源数据集目录之外")
    path.write_text(json.dumps(json_value(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
