"""浏览器数据适配及后台检查回归测试；小型临时 LeRobot 数据集。"""

from dataclasses import asdict
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path
import tempfile
import shutil
import time
import unittest
from threading import Event

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from data_process.__main__ import build_parser
from data_process.config import QUALITY_POLICY, quality_policy_from_dict
from data_process.dataset import BrowserModel
from data_process.application import BrowserApplication, clean_json


class BrowserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.results = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / 'meta').mkdir()
        (root / 'data').mkdir()
        self.root = root
        features = {'state': {'dtype': 'float32', 'shape': [2]},
                    'actions': {'dtype': 'float32', 'shape': [2]},
                    'timestamp': {'dtype': 'float32', 'shape': [1]},
                    'frame_index': {'dtype': 'int64', 'shape': [1]},
                    'image': {'dtype': 'image', 'shape': [3, 8, 8]},
                    'wrist_image': {'dtype': 'image', 'shape': [3, 8, 8]}}
        info = dict(codebase_version='v2.1', features=features, fps=10, chunks_size=1000,
                    data_path='data/episode_{episode_index:06d}.parquet')
        (root / 'meta/info.json').write_text(json.dumps(info))
        (root / 'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=0, task='测试任务'))+'\n')
        (root / 'meta/episodes.jsonl').write_text(''.join(json.dumps(dict(episode_index=i, tasks=['测试任务'], length=8))+'\n' for i in [0,2]))
        stream = BytesIO()
        Image.new('RGB', (8,8), (120,80,40)).save(stream, format='PNG')
        for eid in [0,2]:
            table = pa.table(dict(state=[[float(i),float(i+1)] for i in range(8)],
                                  actions=[[float(i+2),float(i+3)] for i in range(8)],
                                  timestamp=np.arange(8)/10, frame_index=list(range(8)),
                                  image=[dict(bytes=stream.getvalue(),path=None)]*8,
                                  wrist_image=[dict(bytes=stream.getvalue(),path=None)]*7
                                      + [dict(bytes=b'broken',path=None)]))
            pq.write_table(table,root/f'data/episode_{eid:06d}.parquet')
        self.before = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
        self.app = BrowserApplication(BrowserModel(root),2, report_root=self.results.name)

    def tearDown(self):
        self.app.close()
        self.assertEqual(self.before,{str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.temp.cleanup()
        self.results.cleanup()

    def wait_job(self, job):
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            result=self.app.quality_status(job['id'])
            if result['status']!='running':
                return result
            time.sleep(.01)
        self.fail('检查超时')

    def test_info_and_episode_preserve_schema(self):
        info=self.app.info()
        self.assertEqual(info['initial_episode'],2)
        self.assertEqual(info['tasks'][0]['episode_ids'],(0,2))
        episode=self.app.episode_data(2)
        self.assertEqual(episode['state'][3],[3.,4.])
        self.assertEqual(episode['action'][3],[5.,6.])
        self.assertEqual(episode['values']['frame_index'],list(range(8)))

    def test_synchronized_frame_and_original_image(self):
        frame=self.app.frame_data(0,4)
        self.assertEqual((frame['episode'],frame['frame']),(0,4))
        self.assertEqual(frame['values']['state'],[4.,5.])
        self.assertTrue(frame['images']['image'].startswith('data:image/jpeg;base64,'))
        with Image.open(BytesIO(self.app.image_data(0,4,'image'))) as image:
            self.assertEqual(image.size,(8,8))
            self.assertEqual(image.getpixel((0,0)),(120,80,40))

    def test_broken_camera_does_not_hide_other_images(self):
        frame = self.app.frame_data(0, 7)
        self.assertIn('image', frame['images'])
        self.assertNotIn('wrist_image', frame['images'])
        self.assertIn('wrist_image', frame['errors'])
        from data_process.processing import QualityInspector
        report = QualityInspector(self.root).inspect([0])
        check = next(item for item in report['episodes'][0]['checks']
                     if item['name'] == 'image_integrity:wrist_image')
        self.assertEqual(check['status'], 'flagged')
        self.assertEqual(check['frames'], [7])

    def test_invalid_episode_and_frame(self):
        with self.assertRaises((IndexError,KeyError)):
            self.app.episode_data(1)
        for frame in [-1,8]:
            with self.assertRaises(IndexError):
                self.app.frame_data(0,frame)
        with self.assertRaises(ValueError):
            self.app.image_data(0,0,'missing')

    def test_policy_round_trip_and_validation(self):
        self.assertEqual(quality_policy_from_dict(asdict(QUALITY_POLICY)),QUALITY_POLICY)
        with self.assertRaises(ValueError):
            quality_policy_from_dict({'image_stride':1.5})
        with self.assertRaises(ValueError):
            quality_policy_from_dict({'camera_timestamp_keys':[]})
        with self.assertRaises(ValueError):
            self.app.start_quality({'episode_ids':[0],'policy':{'state':{'dimensions':[9]}}})

    def test_quality_report_and_scope(self):
        job=self.app.start_quality({'episode_ids':[2],'policy':{'inspect_images':False}})
        result=self.wait_job(job)
        self.assertEqual(result['status'],'done')
        report=result['report']
        self.assertTrue(report['complete'])
        self.assertFalse(report['source_modified'])
        self.assertEqual(report['scope_episode_ids'],[2])
        self.assertEqual(report['summary']['checked_frames'],8)
        self.assertEqual(len(report['summary']['by_type']),7)
        self.assertEqual(report['selection_unit'], 'frame_segment')
        json.dumps(report,allow_nan=False)

    def test_quality_cancel_partial_report(self):
        self.app.inspector.inspect=lambda ids,policy,**kw: self.cancelled_inspect(ids,policy,**kw)
        job=self.app.start_quality({'episode_ids':[0,2]})
        self.app.stop_quality(job['id'])
        result=self.wait_job(job)
        self.assertFalse(result['report']['complete'])
        self.assertEqual(result['report']['summary']['checked_episodes'],0)

    def cancelled_inspect(self,ids,policy,**kw):
        self.assertTrue(kw['cancel'].wait(3))
        from data_process.processing import QualityInspector
        return QualityInspector(self.root).inspect(ids,policy,**kw)

    def test_json_nonfinite(self):
        self.assertEqual(clean_json({'data':np.array([1,np.nan,np.inf])}),{'data':[1.,None,None]})

    @contextmanager
    def alternate_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(self.root, root, dirs_exist_ok=True)
            info_path = root / 'meta/info.json'
            info = json.loads(info_path.read_text())
            info['features'].pop('wrist_image')
            for old, new in [('state', 'observation.state'), ('actions', 'action')]:
                info['features'][new] = info['features'].pop(old)
                info['features'][new]['shape'] = [1]
            info_path.write_text(json.dumps(info))
            for path in (root / 'data').glob('*.parquet'):
                original = pq.read_table(path).to_pydict()
                table = pa.table({
                    'observation.state': [[42.]] * 8,
                    'action': [[43.]] * 8,
                    'timestamp': original['timestamp'],
                    'frame_index': original['frame_index'],
                    'image': original['image'],
                })
                pq.write_table(table, path)
            yield root

    def test_switch_dataset_resets_cached_data_and_report(self):
        self.app.episode_data(0)
        job = self.app.start_quality({'episode_ids': [0], 'policy': {'inspect_images': False}})
        self.wait_job(job)
        revision = self.app.dataset_id
        with self.alternate_dataset() as root:
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            info = self.app.open_dataset(str(root))
            self.assertEqual(info['path'], str(root))
            self.assertNotEqual(info['dataset_id'], revision)
            self.assertIsNone(info['initial_episode'])
            self.assertIsNone(self.app.job)
            self.assertEqual(info['schema']['state']['key'], 'observation.state')
            frame = self.app.frame_data(0, 4)
            self.assertEqual(frame['values']['observation.state'], [42.])
            self.assertEqual(set(frame['images']), {'image'})
            self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes()
                                      for p in root.rglob('*') if p.is_file()})

    def test_invalid_dataset_keeps_current_source(self):
        model, revision = self.app.model, self.app.dataset_id
        for path in ['', None, str(self.root / 'missing'), str(self.root / 'data')]:
            with self.assertRaises(ValueError):
                self.app.open_dataset(path)
            self.assertIs(self.app.model, model)
            self.assertEqual(self.app.dataset_id, revision)
        with self.alternate_dataset() as root:
            (root / 'data/episode_000000.parquet').write_bytes(b'broken')
            with self.assertRaises(ValueError):
                self.app.open_dataset(str(root))
            self.assertIs(self.app.model, model)
        self.assertEqual(self.app.episode_data(0)['state'][0], [0., 1.])

    def test_running_quality_blocks_dataset_switch(self):
        self.app.inspector.inspect = self.cancelled_inspect
        job = self.app.start_quality({'episode_ids': [0, 2]})
        with self.assertRaisesRegex(ValueError, '先.*停止'):
            self.app.open_dataset(str(self.root))
        self.app.stop_quality(job['id'])
        self.assertFalse(self.wait_job(job)['report']['complete'])

    def test_browse_service_directories(self):
        result = self.app.directories()
        self.assertEqual(result['path'], str(self.root))
        self.assertTrue(result['is_dataset'])
        self.assertEqual([row['name'] for row in result['directories']], ['data', 'meta'])
        result = self.app.directories(str(self.root / 'meta'))
        self.assertEqual(result['parent'], str(self.root))
        self.assertFalse(result['is_dataset'])
        self.assertEqual(result['directories'], [])
        with self.assertRaises(ValueError):
            self.app.directories(str(self.root / 'missing'))

    def test_cli_browser_options(self):
        args=build_parser().parse_args(['view','--no-browser','--port','9000'])
        self.assertEqual(args.host,'127.0.0.1')
        self.assertTrue(args.no_browser)
        self.assertEqual(args.port,9000)

    def test_report_folder_round_trip_and_lazy_details(self):
        from data_process.report_store import ReportBundle
        job = self.app.start_quality({'episode_ids': [0, 2]})
        result = self.wait_job(job)
        self.assertEqual(result['status'], 'done', result.get('error'))
        folder = Path(result['report_path'])
        self.assertEqual(len(list((folder / 'episodes').glob('*.json.gz'))), 2)
        index = result['report']['episodes'][0]
        self.assertNotIn('frame_flags', index)
        self.assertNotIn('checks', index)
        self.assertTrue(all('samples' not in s for s in index['segments']))
        raw = self.app.quality_episode(job['id'], 0)
        self.assertTrue(raw['frame_flags'])
        opened = ReportBundle.open(folder / 'manifest.json', self.app.model)
        self.assertFalse(opened.cache)
        self.assertEqual(opened.get_episode(0), raw)
        self.assertEqual(opened.report()['summary'], result['report']['summary'])
        saved = self.app.save_quality(job['id'])
        self.assertEqual(saved['path'], str(folder))
        self.assertEqual(saved['episodes'], 2)
        again = self.app.import_quality(str(folder))
        self.assertNotEqual(again['id'], job['id'])
        self.assertEqual(self.app.quality_episode(again['id'], 0), raw)

    def test_realtime_partial_episode_and_incremental_poll(self):
        entered, resume = Event(), Event()
        original = self.app.inspector._synchronization
        def delayed(episode, policy, add):
            entered.set()
            self.assertTrue(resume.wait(4))
            original(episode, policy, add)
        self.app.inspector._synchronization = delayed
        job = self.app.start_quality({'episode_ids': [0]})
        try:
            self.assertTrue(entered.wait(3))
            status = self.app.quality_status(job['id'], -1)
            self.assertEqual(status['status'], 'running')
            self.assertEqual(status['completed'], 0)
            self.assertGreater(status['report']['summary']['suspect_frames'], 0)
            self.assertFalse(status['report']['episodes'][0]['episode_complete'])
            self.assertTrue(self.app.quality_episode(job['id'], 0)['frame_flags'])
            delta = self.app.quality_status(job['id'], status['revision'])
            self.assertFalse(delta['reset'])
            self.assertEqual(delta['report']['episodes'], [])
            self.assertEqual(delta['report']['summary'], status['report']['summary'])
            with self.assertRaisesRegex(ValueError, '自动保存'):
                self.app.save_quality(job['id'])
            with self.assertRaisesRegex(ValueError, '停止'):
                self.app.import_quality(status['report_path'])
        finally:
            resume.set()
        complete = self.wait_job(job)
        self.assertTrue(complete['report']['complete'])
        final_delta = self.app.quality_status(job['id'], status['revision'])
        self.assertTrue(final_delta['report']['episodes'][0]['episode_complete'])

    def test_cancel_discards_only_unfinished_live_episode(self):
        from concurrent.futures import CancelledError
        from data_process.report_store import ReportBundle
        entered, resume = Event(), Event()
        def delayed(episode, policy, add):
            entered.set()
            resume.wait(4)
            raise CancelledError()
        self.app.inspector._synchronization = delayed
        job = self.app.start_quality({'episode_ids': [0, 2]})
        try:
            self.assertTrue(entered.wait(3))
            live = self.app.quality_status(job['id'])
            self.assertEqual(live['episode_ids'], [0])
            self.app.stop_quality(job['id'])
        finally:
            resume.set()
        stopped = self.wait_job(job)
        self.assertFalse(stopped['report']['complete'])
        self.assertEqual(stopped['episode_ids'], [])
        delta = self.app.quality_status(job['id'], live['revision'])
        self.assertEqual(delta['episode_ids'], [])
        self.assertFalse(ReportBundle.open(stopped['report_path'], self.app.model).rows)

    def test_legacy_json_import_converts_and_preserves_original(self):
        from data_process.processing import QualityInspector
        report = QualityInspector(self.root).inspect([0, 2])
        path = Path(self.results.name) / 'old.json'
        path.write_text(json.dumps(report), encoding='utf-8')
        before = path.read_bytes()
        activated = self.app.import_quality(str(path))
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue((Path(activated['report_path']) / 'manifest.json').is_file())
        self.assertEqual(self.app.quality_episode(activated['id'], 2), report['episodes'][1])
        report['episodes'][0]['segments'][0]['end_frame'] = 99999
        path.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '越界'):
            self.app.import_quality(str(path))
        self.assertEqual(self.app.quality_active()['id'], activated['id'])

    def test_report_restore_after_application_restart(self):
        job = self.app.start_quality({'episode_ids': [2]})
        result = self.wait_job(job)
        restored = BrowserApplication(BrowserModel(self.root), report_root=self.results.name)
        try:
            active = restored.quality_active()
            self.assertTrue(active['id'])
            self.assertEqual(active['report_path'], result['report_path'])
            report = restored.quality_status(active['id'])['report']
            self.assertTrue(report['complete'])
            self.assertEqual(report['summary'], result['report']['summary'])
        finally:
            restored.close()

    def test_import_validates_dataset_and_changed_source_fingerprint(self):
        from data_process.report_store import ReportBundle
        job = self.app.start_quality({'episode_ids': [0]})
        result = self.wait_job(job)
        folder = Path(result['report_path'])
        with self.alternate_dataset() as other:
            with self.assertRaisesRegex(ValueError, '不一致'):
                ReportBundle.open(folder, BrowserModel(other))
        index_path = folder / 'index.jsonl'
        index = json.loads(index_path.read_text())
        index['input_file']['mtime_ns'] += 1
        index_path.write_text(json.dumps(index) + '\n')
        with self.assertRaisesRegex(ValueError, '已改变'):
            ReportBundle.open(folder, self.app.model)

    def test_report_output_must_be_outside_source_dataset(self):
        with self.assertRaisesRegex(ValueError, '之外'):
            self.app.start_quality({'episode_ids': [0], 'output_root': str(self.root / 'output')})
        self.assertIsNone(self.app.job)
        self.assertFalse((self.root / 'output').exists())


if __name__=='__main__':
    unittest.main()
