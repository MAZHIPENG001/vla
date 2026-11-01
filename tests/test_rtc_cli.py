"""Check automatic/manual RTC settings across the shell and real tyro parser."""

import importlib.util
import dataclasses
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from examples.LIBERO.eval_files.eval_libero import Args


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/LIBERO/eval_files/eval_libero.sh"
HAS_TYRO = importlib.util.find_spec("tyro") is not None


class RTCArgumentTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TYRO, "tyro is an evaluation CLI dependency")
    def test_python_defaults_and_auto_manual_combinations_parse(self):
        import tyro

        def parse(args: Args):
            return args

        defaults = tyro.cli(parse, args=[])
        self.assertEqual((defaults.rtc_stride, defaults.rtc_delay), ("auto", "auto"))
        self.assertEqual(defaults.rtc_beta, 5.0)
        for stride, delay in (("auto", "auto"), (4, 2), (4, "auto"), ("auto", 0)):
            with self.subTest(stride=stride, delay=delay):
                parsed = tyro.cli(parse, args=["--args.rtc-stride", str(stride), "--args.rtc-delay", str(delay)])
                self.assertEqual((parsed.rtc_stride, parsed.rtc_delay), (stride, delay))

    @unittest.skipUnless(HAS_TYRO, "tyro is an evaluation CLI dependency")
    def test_only_paper_beta_is_exposed_and_removed_switches_are_rejected(self):
        import tyro

        fields = {field.name for field in dataclasses.fields(Args)}
        self.assertNotIn("rtc_guidance_beta", fields)
        self.assertNotIn("rtc_inpainting", fields)
        self.assertNotIn("rtc_timeout", fields)
        self.assertIn("inference_timeout", fields)

        def parse(args: Args):
            return args

        self.assertEqual(tyro.cli(parse, args=["--args.rtc-beta", "7"]).rtc_beta, 7)
        for removed in ("--args.rtc-guidance-beta", "--args.rtc-inpainting", "--args.rtc-timeout"):
            with self.subTest(removed=removed), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tyro.cli(parse, args=[removed])


class RTCShellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vla-rtc-shell-")
        self.addCleanup(self.temp.cleanup)
        self.environment = os.environ.copy()
        for key in list(self.environment):
            if key.startswith("RTC_") or key in ("REALTIME", "INFERENCE_TIMEOUT"):
                self.environment.pop(key)
        self.environment.update({"VLA": str(ROOT), "LIBERO_HOME": self.temp.name,
                                 "LIBERO_PYTHON": sys.executable, "CONTROL_FREQ": "20"})

    def run_shell(self, **settings):
        env = dict(self.environment, **settings)
        return subprocess.run(["bash", str(SCRIPT)], env=env, cwd=ROOT, capture_output=True, text=True, timeout=10)

    def test_invalid_auto_or_integer_settings_fail_before_launch(self):
        invalid = (("RTC_STRIDE", "0"), ("RTC_STRIDE", "-1"), ("RTC_STRIDE", "1.5"),
                   ("RTC_STRIDE", "automatic"), ("RTC_DELAY", "-1"), ("RTC_DELAY", "2.5"),
                   ("RTC_DELAY", "AUTO"))
        for setting, value in invalid:
            with self.subTest(setting=setting, value=value):
                result = self.run_shell(**{setting: value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(setting + " must be auto", result.stderr)
                self.assertNotIn("LIBERO evaluation:", result.stdout)

    def test_removed_environment_options_require_explicit_migration(self):
        for setting, value in (("RTC_GUIDANCE_BETA", "5"), ("RTC_INPAINTING", "true"), ("RTC_INPAINTING", "false")):
            with self.subTest(setting=setting, value=value):
                result = self.run_shell(**{setting: value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(setting + " has been removed", result.stderr)
                self.assertNotIn("LIBERO evaluation:", result.stdout)
        result = self.run_shell(RTC_TIMEOUT="10")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RTC_TIMEOUT has been renamed to INFERENCE_TIMEOUT", result.stderr)
        self.assertNotIn("LIBERO evaluation:", result.stdout)

    @unittest.skipUnless(HAS_TYRO, "tyro is an evaluation CLI dependency")
    def test_shell_passes_default_auto_and_explicit_manual_settings_to_tyro(self):
        # Parse the actual shell-produced argv with the real Args dataclass;
        # never import LIBERO or open a model connection during the smoke test.
        shim = Path(self.temp.name) / "parse-eval"
        shim.write_text(
            "#!" + sys.executable + "\n"
            "import dataclasses,json,sys,tyro\n"
            "from examples.LIBERO.eval_files.eval_libero import Args\n"
            "def parse(args: Args): return args\n"
            "parsed=tyro.cli(parse,args=sys.argv[2:])\n"
            "print(json.dumps(dataclasses.asdict(parsed)))\n"
        )
        shim.chmod(0o700)
        self.environment["LIBERO_PYTHON"] = str(shim)
        expected_settings = (({}, "auto", "auto"),
                             ({"RTC_STRIDE": "4", "RTC_DELAY": "2"}, 4, 2),
                             ({"RTC_STRIDE": "4", "RTC_DELAY": "auto"}, 4, "auto"),
                             ({"RTC_STRIDE": "auto", "RTC_DELAY": "0"}, "auto", 0))
        for settings, stride, delay in expected_settings:
            with self.subTest(settings=settings):
                result = self.run_shell(**settings)
                self.assertEqual(result.returncode, 0, result.stderr)
                parsed = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual((parsed["rtc_stride"], parsed["rtc_delay"]), (stride, delay))
                self.assertTrue(parsed["rtc_enabled"])  # Preserve the existing shell default.
                self.assertEqual(parsed["rtc_beta"], 5.0)
                self.assertNotIn("rtc_guidance_beta", parsed)
                self.assertNotIn("rtc_inpainting", parsed)
                self.assertNotIn("rtc_timeout", parsed)
                self.assertEqual(parsed["inference_timeout"], 10.0)


if __name__ == "__main__":
    unittest.main()
