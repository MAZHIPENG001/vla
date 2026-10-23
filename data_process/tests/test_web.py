"""浏览器数据适配及后台检查回归测试；小型临时 LeRobot 数据集。"""

from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
import unittest

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
        self.app = BrowserApplication(BrowserModel(root),2)

    def tearDown(self):
        self.app.close()
        self.assertEqual(self.before,{str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.temp.cleanup()

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
        self.assertEqual(check['status'], 'rejected')
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
        self.assertEqual(len(report['summary']['funnel']),7)
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

    def test_cli_browser_options(self):
        args=build_parser().parse_args(['view','--no-browser','--port','9000'])
        self.assertEqual(args.host,'127.0.0.1')
        self.assertTrue(args.no_browser)
        self.assertEqual(args.port,9000)


if __name__=='__main__':
    unittest.main()
