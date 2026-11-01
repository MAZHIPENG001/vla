"""Exercise LIBERO GPU selection without loading CUDA, a model, or LIBERO."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from examples.LIBERO.eval_files import eval_libero as evaluation


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "examples/LIBERO/eval_files"
SCRIPTS = ("run_policy_server.sh", "eval_libero.sh")
LAUNCH_MARKER = "GPU_TEST_LAUNCH "
HAS_TYRO = importlib.util.find_spec("tyro") is not None


class GPUShellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vla gpu test ")
        self.addCleanup(self.temp.cleanup)
        self.shim = Path(self.temp.name) / "python shim"
        self.shim.write_text(
            "#!" + sys.executable + "\n"
            "import json,os,sys\n"
            "print('" + LAUNCH_MARKER + "'+json.dumps({"
            "'argv':sys.argv[1:],"
            "'cuda':os.environ.get('CUDA_VISIBLE_DEVICES'),"
            "'egl':os.environ.get('MUJOCO_EGL_DEVICE_ID')}))\n"
        )
        self.shim.chmod(0o700)
        self.environment = os.environ.copy()
        for key in list(self.environment):
            if key.startswith("RTC_") or key in (
                "GPU_ID", "CUDA_VISIBLE_DEVICES", "REALTIME", "CONTROL_FREQ",
                "INFERENCE_TIMEOUT", "MUJOCO_EGL_DEVICE_ID", "EGL_DEVICE_ID",
            ):
                self.environment.pop(key)
        self.checkpoint = str(Path(self.temp.name) / "model/checkpoints/step 100.pt")
        self.environment.update({
            "VLA": str(ROOT), "VLA_DIR": str(ROOT),
            "LIBERO_HOME": self.temp.name,
            "VLA_PYTHON": str(self.shim), "LIBERO_PYTHON": str(self.shim),
            "CKPT": self.checkpoint,
        })

    def run_shell(self, script, args=(), **settings):
        return subprocess.run(
            ["bash", str(SCRIPT_DIR / script), *args],
            cwd=ROOT, env=dict(self.environment, **settings),
            capture_output=True, text=True, timeout=10,
        )

    def assert_gpu_launch(self, script, expected, args=(), **settings):
        result = self.run_shell(script, args, **settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        launches = [line.removeprefix(LAUNCH_MARKER)
                    for line in result.stdout.splitlines() if line.startswith(LAUNCH_MARKER)]
        self.assertEqual(len(launches), 1, result.stdout)
        launched = json.loads(launches[0])
        self.assertEqual(launched["cuda"], str(expected))
        self.assertIn(self.checkpoint, launched["argv"])
        if script == "eval_libero.sh":
            self.assertEqual(launched["egl"], str(expected))
            render_option = launched["argv"].index("--args.render-gpu-device-id")
            self.assertEqual(launched["argv"][render_option + 1], str(expected))

    def test_default_and_inherited_visible_gpu(self):
        for script in SCRIPTS:
            with self.subTest(script=script, source="default"):
                self.assert_gpu_launch(script, 0)
            with self.subTest(script=script, source="CUDA_VISIBLE_DEVICES"):
                self.assert_gpu_launch(script, 3, CUDA_VISIBLE_DEVICES="3,5")

    def test_explicit_environment_overrides_inherited_gpu(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                self.assert_gpu_launch(script, 6, GPU_ID="6", CUDA_VISIBLE_DEVICES="3,5")
            with self.subTest(script=script, source="explicit zero"):
                self.assert_gpu_launch(script, 0, GPU_ID="0", CUDA_VISIBLE_DEVICES="3,5")

    def test_invalid_inherited_selection_requires_explicit_gpu(self):
        # A disabled or unsupported inherited selection must not silently use
        # GPU 0, which might belong to another job on a shared server.
        for script in SCRIPTS:
            for visible in ("", "-1", "GPU-123", ",3"):
                with self.subTest(script=script, visible=visible):
                    result = self.run_shell(script, CUDA_VISIBLE_DEVICES=visible)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn(LAUNCH_MARKER, result.stdout)
                    self.assert_gpu_launch(script, 2, ("--gpu-id", "2"),
                                           CUDA_VISIBLE_DEVICES=visible)

    def test_command_line_overrides_environment_for_both_spellings(self):
        for script in SCRIPTS:
            for args in (("--gpu-id", "4"), ("--gpu-id=4",)):
                with self.subTest(script=script, args=args):
                    self.assert_gpu_launch(script, 4, args, GPU_ID="6", CUDA_VISIBLE_DEVICES="3,5")

    def test_evaluation_replaces_stale_egl_selection(self):
        self.assert_gpu_launch("eval_libero.sh", 2, ("--gpu-id", "2"),
                               GPU_ID="6", CUDA_VISIBLE_DEVICES="3,5", MUJOCO_EGL_DEVICE_ID="7")

    def test_invalid_gpu_ids_fail_before_python_launch(self):
        for script in SCRIPTS:
            for gpu in ("", "-1", "01", "2,3", "1.5", "GPU-123", "abc"):
                for source in ("environment", "command line"):
                    with self.subTest(script=script, gpu=gpu, source=source):
                        args = ("--gpu-id", gpu) if source == "command line" else ()
                        settings = {"GPU_ID": gpu} if source == "environment" else {}
                        result = self.run_shell(script, args, **settings)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertNotIn(LAUNCH_MARKER, result.stdout)
                        self.assertIn("GPU", result.stderr.upper())

    def test_missing_gpu_argument_and_unknown_options_fail_before_launch(self):
        for script in SCRIPTS:
            for args in (("--gpu-id",), ("--unknown",), ("2",)):
                with self.subTest(script=script, args=args):
                    result = self.run_shell(script, args)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn(LAUNCH_MARKER, result.stdout)

    def test_help_does_not_require_simulator_or_python(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result = self.run_shell(script, ("--help",), LIBERO_HOME="",
                                        VLA_PYTHON="/missing/python", LIBERO_PYTHON="/missing/python")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--gpu-id", result.stdout)
                self.assertNotIn(LAUNCH_MARKER, result.stdout)


class RenderGPUTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TYRO, "tyro is an evaluation CLI dependency")
    def test_render_gpu_argument_parses_with_real_tyro(self):
        import tyro

        def parse(args: evaluation.Args):
            return args

        self.assertEqual(tyro.cli(parse, args=[]).render_gpu_device_id, -1)
        parsed = tyro.cli(parse, args=["--args.render-gpu-device-id", "4"])
        self.assertEqual(parsed.render_gpu_device_id, 4)

    def test_render_gpu_is_forwarded_to_libero_environment(self):
        libero = ModuleType("libero")
        package = ModuleType("libero.libero")
        package.get_libero_path = Mock(return_value="/tmp/libero/bddl_files")
        envs = ModuleType("libero.libero.envs")
        env = Mock()
        envs.OffScreenRenderEnv = Mock(return_value=env)
        task = SimpleNamespace(language="test task", problem_folder="suite", bddl_file="task.bddl")
        with patch.dict(sys.modules, {"libero": libero, "libero.libero": package,
                                     "libero.libero.envs": envs}):
            actual, description = evaluation._get_libero_env(
                task, 256, 7, control_freq=30, render_gpu_device_id=4,
            )
        self.assertIs(actual, env)
        self.assertEqual(description, "test task")
        envs.OffScreenRenderEnv.assert_called_once_with(
            bddl_file_name=Path("/tmp/libero/bddl_files/suite/task.bddl"),
            camera_heights=256, camera_widths=256, control_freq=30,
            render_gpu_device_id=4,
        )
        env.seed.assert_called_once_with(7)


if __name__ == "__main__":
    unittest.main()
