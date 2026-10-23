"""按数据集 / 检测轮次 / Episode 保存报告，明细压缩并按需读取。"""

from collections import OrderedDict
from datetime import datetime
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import secrets
from zoneinfo import ZoneInfo

from .config import quality_policy_from_dict
from .dataset import json_value
from .processing import REPORT_TYPES, build_report, episode_summary

DEFAULT_REPORT_ROOT = Path(__file__).parent / 'output'
FORMAT = 'lerobot_frame_segment_bundle'


def _write_json(path, value):
    temporary = path.with_name(path.name + '.tmp-' + secrets.token_hex(4))
    try:
        temporary.write_text(json.dumps(json_value(value), ensure_ascii=False, allow_nan=False,
                                        separators=(',', ':')), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def metadata_digest(model):
    digest = sha256()
    for name in ('info.json', 'episodes.jsonl', 'tasks.jsonl'):
        digest.update((model.loader.path / 'meta' / name).read_bytes())
    digest.update(json.dumps({'state': model.schema.state.key, 'action': model.schema.action.key}, sort_keys=True).encode())
    return digest.hexdigest()


def dataset_key(model):
    name = re.sub(r'[^\w.-]+', '_', model.loader.path.name)[:70] or 'dataset'
    return name + '-' + sha256(str(model.loader.path).encode()).hexdigest()[:10]


def _input_stat(model, eid):
    relative = model.loader.info['data_path'].format(episode_chunk=eid // model.loader.info['chunks_size'], episode_index=eid)
    path = model.loader.path / relative
    try:
        stat = path.stat()
        return dict(path=relative, exists=True, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    except FileNotFoundError:
        return dict(path=relative, exists=False)


def validate_episode(row, model):
    if not isinstance(row, dict):
        raise ValueError('Episode 报告必须为对象')
    eid = row.get('episode_index')
    if type(eid) is not int or eid not in model.loader.episodes:
        raise ValueError(f'报告包含不存在的 Episode：{eid}')
    length = model.loader.episodes[eid]['length']
    if row.get('length') != length:
        raise ValueError(f'Episode {eid} 帧数与当前数据集不一致')
    counts = [row.get(key) for key in ('normal_frames', 'suspect_frames', 'unchecked_frames')]
    if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != length:
        raise ValueError(f'Episode {eid} 的帧级计数无效')
    if row.get('status') not in {'unreadable', 'flagged', 'review', 'clear'}:
        raise ValueError(f'Episode {eid} 的状态无效')
    if any(not isinstance(row.get(key), list) or any(not isinstance(v, str) for v in row[key])
           for key in ('tasks', 'unverified_checks')):
        raise ValueError(f'Episode {eid} 的任务或未验证项目无效')
    partition, suspect, marked, segment_ids = set(), set(), set(), set()
    for key in ('normal_segments', 'suspect_segments', 'unchecked_segments', 'segments'):
        if not isinstance(row.get(key), list):
            raise ValueError(f'Episode {eid} 缺少区间字段 {key}')
        covered = set()
        for segment in row[key]:
            if not isinstance(segment, dict):
                raise ValueError('报告的区间必须为对象')
            a, b = segment.get('start_frame'), segment.get('end_frame')
            if type(a) is not int or type(b) is not int or not 0 <= a <= b < length:
                raise ValueError(f'Episode {eid} 区间索引越界')
            frames = set(range(a, b + 1))
            if segment.get('episode_index') != eid:
                raise ValueError('报告的区间与 Episode 映射不一致')
            if key == 'segments':
                if segment.get('episode_index') != eid or segment.get('type') not in dict(REPORT_TYPES):
                    raise ValueError('报告的区间类型或 Episode 映射无效')
                if segment.get('status') not in {'suspect', 'waiting'} or not isinstance(segment.get('id'), str):
                    raise ValueError('报告的区间状态或标识无效')
                if segment['id'] in segment_ids:
                    raise ValueError('报告的区间标识重复')
                segment_ids.add(segment['id'])
                if segment['status'] == 'suspect':
                    marked.update(frames)
                if not isinstance(segment.get('dimensions'), list):
                    raise ValueError('报告的异常维度必须为数组')
                role = segment.get('role')
                if role in {'state', 'action'}:
                    width = getattr(model.schema, role).width
                    if any(type(d) is not int or not 0 <= d < width for d in segment.get('dimensions', [])):
                        raise ValueError(f'{role} 异常维度与当前数据集不一致')
            else:
                if frames & (covered | partition):
                    raise ValueError(f'Episode {eid} 的帧级区间重叠')
                covered.update(frames)
        if key != 'segments':
            if len(covered) != row[key.replace('_segments', '_frames')]:
                raise ValueError(f'Episode {eid} 的区间与帧数统计不一致')
            partition.update(covered)
            if key == 'suspect_segments':
                suspect = covered
    if suspect != marked:
        raise ValueError(f'Episode {eid} 的异常区间与疑似异常帧不一致')
    return eid


class ReportBundle:
    def __init__(self, root, model, manifest, rows=None):
        self.root, self.model, self.manifest = root, model, manifest
        self.rows = rows or {}
        self.cache = OrderedDict()

    @classmethod
    def create(cls, model, policy, ids, report_root=DEFAULT_REPORT_ROOT):
        output = Path(report_root).expanduser().resolve()
        if output.is_relative_to(model.loader.path):
            raise ValueError('检测结果目录必须位于原始数据集之外')
        run_id = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(4)
        root = output / dataset_key(model) / run_id
        (root / 'episodes').mkdir(parents=True)
        header = build_report(model.loader.path, policy, ids, [], False)
        header.pop('episodes')
        manifest = dict(header, format=FORMAT, bundle_version=1, created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                        metadata_digest=metadata_digest(model), index='index.jsonl', compression='gzip')
        bundle = cls(root, model, manifest)
        (root / 'index.jsonl').touch()
        bundle.finish(False)
        return bundle

    def _episode_path(self, eid):
        path = self.root / 'episodes' / f'episode_{eid:06d}.json.gz'
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('报告中的 Episode 文件不能指向报告目录之外')
        return path

    def append(self, row):
        eid = validate_episode(row, self.model)
        if eid not in self.manifest['scope_episode_ids'] or eid in self.rows:
            raise ValueError('报告的 Episode 重复或不在检测范围内')
        path = self._episode_path(eid)
        temporary = path.with_suffix('.tmp')
        try:
            with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=6) as stream:
                json.dump(json_value(row), stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        index = episode_summary(row)
        index['input_file'] = _input_stat(self.model, eid)
        with (self.root / 'index.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(index, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
            stream.flush()
        self.rows[eid] = index
        return index

    def report(self):
        result = build_report(self.manifest['source'], self.manifest['policy'], self.manifest['scope_episode_ids'],
                              list(self.rows.values()), self.manifest['complete'])
        result.update(report_path=str(self.root), created_at=self.manifest['created_at'])
        return result

    def finish(self, complete):
        result = build_report(self.manifest['source'], self.manifest['policy'], self.manifest['scope_episode_ids'],
                              list(self.rows.values()), complete)
        self.manifest.update(complete=complete, summary=result['summary'])
        _write_json(self.root / 'manifest.json', self.manifest)

    def get_episode(self, eid):
        if eid not in self.rows:
            raise KeyError(f'Episode {eid} 尚未完成检测')
        if eid in self.cache:
            self.cache.move_to_end(eid)
            return self.cache[eid]
        with gzip.open(self._episode_path(eid), 'rt', encoding='utf-8') as stream:
            row = json.load(stream)
        if validate_episode(row, self.model) != eid:
            raise ValueError('Episode 明细与索引不匹配')
        self.cache[eid] = row
        if len(self.cache) > 3:
            self.cache.popitem(last=False)
        return row

    @classmethod
    def open(cls, path, model):
        root = Path(path).expanduser().resolve()
        if root.is_file():
            if root.name != 'manifest.json':
                raise ValueError('请选择检测结果目录或 manifest.json')
            root = root.parent
        if root.is_relative_to(model.loader.path):
            raise ValueError('检测结果目录必须位于原始数据集之外')
        with (root / 'manifest.json').open(encoding='utf-8') as stream:
            manifest = json.load(stream)
        if manifest.get('format') != FORMAT or manifest.get('bundle_version') != 1:
            raise ValueError('不支持的报告目录格式')
        if type(manifest.get('complete')) is not bool:
            raise ValueError('报告的完成状态无效')
        if Path(manifest['source']).resolve() != model.loader.path or manifest.get('metadata_digest') != metadata_digest(model):
            raise ValueError('历史报告与当前数据集路径、元数据或 State / Action 映射不一致')
        quality_policy_from_dict(manifest['policy'])
        ids = manifest.get('scope_episode_ids')
        if not isinstance(ids, list) or any(type(eid) is not int or eid not in model.loader.episodes for eid in ids) or len(set(ids)) != len(ids):
            raise ValueError('报告检测范围与当前数据集不一致')
        bundle = cls(root, model, manifest)
        index_path = root / 'index.jsonl'
        if not index_path.resolve().is_relative_to(root):
            raise ValueError('报告索引不能指向目录之外')
        with index_path.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith('\n') and not manifest['complete']:
                        break  # 进程中断留下的最后一条未完整索引不冒充已完成数据。
                    raise ValueError('报告索引损坏') from None
                eid = validate_episode(row, model)
                if eid in bundle.rows or eid not in ids or not row.get('episode_complete'):
                    raise ValueError('报告索引包含重复、越界或未完成 Episode')
                if row.get('input_file') != _input_stat(model, eid):
                    raise ValueError(f'Episode {eid} 的源文件已改变，请重新检测')
                if not bundle._episode_path(eid).is_file():
                    raise ValueError(f'Episode {eid} 的检测明细文件缺失')
                bundle.rows[eid] = row
        if manifest['complete'] and set(bundle.rows) != set(ids):
            raise ValueError('完整报告缺少部分 Episode，不能应用区间预览')
        return bundle


def report_history(model, report_root=DEFAULT_REPORT_ROOT):
    root = Path(report_root).expanduser().resolve() / dataset_key(model)
    result = []
    if root.is_dir():
        for path in sorted(root.glob('*/manifest.json'), reverse=True):
            try:
                header = json.loads(path.read_text(encoding='utf-8'))
                if header.get('format') != FORMAT:
                    continue
                result.append(dict(path=str(path.parent), created_at=header['created_at'], complete=header['complete'],
                                   checked_episodes=header['summary']['checked_episodes'], summary=header['summary']))
            except (OSError, ValueError, KeyError):
                continue
    return result


def import_report(path, model, report_root=DEFAULT_REPORT_ROOT):
    """新目录直接打开；旧版 v2 单 JSON 验证后转换为分目录文件，原文件保留。"""
    source = Path(path).expanduser().resolve()
    if source.is_dir() or source.name == 'manifest.json':
        return ReportBundle.open(source, model)
    with source.open(encoding='utf-8') as stream:
        report = json.load(stream)
    if report.get('version') != 2 or report.get('selection_unit') != 'frame_segment':
        raise ValueError('只支持帧 / 区间 v2 报告；旧 Episode 拒绝制报告需要重新检测')
    if Path(report.get('source', '')).resolve() != model.loader.path:
        raise ValueError('历史报告不属于当前数据集，请先打开对应数据集')
    policy = quality_policy_from_dict(report['policy'])
    ids = report.get('scope_episode_ids')
    if not isinstance(ids, list) or len(set(ids)) != len(ids) or any(type(eid) is not int or eid not in model.loader.episodes for eid in ids):
        raise ValueError('历史报告的检测范围无效')
    rows = report.get('episodes')
    if not isinstance(rows, list):
        raise ValueError('历史报告缺少 Episode 数据')
    seen = set()
    for row in rows:
        eid = validate_episode(row, model)
        if eid in seen or eid not in ids:
            raise ValueError('历史报告的 Episode 重复或超出范围')
        seen.add(eid)
    if report.get('complete') and seen != set(ids):
        raise ValueError('历史报告不完整，不能作为完整检测导入')
    bundle = ReportBundle.create(model, policy, ids, report_root)
    for row in rows:
        bundle.append(row)
    bundle.manifest['imported_from'] = str(source)
    bundle.finish(bool(report.get('complete')))
    return bundle
