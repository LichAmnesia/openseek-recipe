"""Offline contracts for the manual experiment and published measured evidence."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "experiments/deepseek-v41-flash-1017"
spec = importlib.util.spec_from_file_location("dsv41_prepare", BUNDLE / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class DeepSeekExperimentTests(unittest.TestCase):
    def test_locked_sources_and_checkpoint_are_complete(self):
        lock = json.loads((BUNDLE / "runtime-lock.json").read_text())
        prepare.validate(lock)
        self.assertEqual(len(lock["source_files"]), 152)
        model = json.loads((BUNDLE / "model-files.json").read_text())
        self.assertEqual(model["sha"], lock["model_revision"])
        self.assertEqual(len(model["siblings"]), 55)
        self.assertEqual(sum(x["size"] for x in model["siblings"]), 405050217567)
        self.assertEqual(sum(x["rfilename"].endswith(".safetensors") for x in model["siblings"]), 48)

    def test_unsafe_paths_and_floating_source_url_rejected(self):
        lock = json.loads((BUNDLE / "runtime-lock.json").read_text())
        for key, value in (("path", "../escape"), ("url", "https://example.org/main/file")):
            bad = copy.deepcopy(lock)
            bad["source_files"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare.validate(bad)

    def test_hash_failure_and_existing_edit_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.py"
            data = b"locked source\n"
            digest = hashlib.sha256(data).hexdigest()
            with self.assertRaises(ValueError):
                prepare.checked_write(path, b"corrupt", digest)
            self.assertFalse(path.exists())
            prepare.checked_write(path, data, digest)
            prepare.checked_write(path, data, digest)
            path.write_bytes(b"user edit")
            with self.assertRaises(ValueError):
                prepare.checked_write(path, data, digest)
            self.assertEqual(path.read_bytes(), b"user edit")

    def test_prepare_dry_run_does_not_write_or_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "not-created"
            with patch.object(sys, "argv", ["prepare.py", "--root", str(root), "--dry-run"]), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
                prepare.main()
            self.assertFalse(root.exists())

    def test_helpers_parse_and_shell_dry_run_is_offline(self):
        for path in (BUNDLE / "scripts").glob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
        for path in [BUNDLE / "serve.sh", *(BUNDLE / "scripts").glob("*.sh")]:
            subprocess.run(["bash", "-n", str(path)], check=True)
        out = subprocess.check_output(["bash", str(BUNDLE / "serve.sh"), "--dry-run"], text=True)
        self.assertIn("TP1/eager", out)
        self.assertIn("memory=100g swap=0", out)

    def test_published_metrics_match_raw_resource_and_api_data(self):
        result = ROOT / "results/2026-10-26-deepseek-2.7bpw-eager"
        b = json.loads((result / "benchmark.json").read_text())
        m = json.loads((result / "measurements.json").read_text())
        self.assertTrue(b["ok"])
        self.assertTrue(all(x["ok"] for x in b["smoke"].values()))
        self.assertEqual(len(b["speed"]), 4)
        for row in b["speed"]:
            self.assertTrue(row["valid_speed_sample"])
            self.assertEqual(row["completion_tokens"], 256)
            self.assertEqual(row["finish_reason"], "length")
        c1 = b["speed"][:3]
        self.assertEqual(statistics.median(x["observed_decode_tok_s"] for x in c1), m["c1_median_observed_decode_tok_s"])
        self.assertEqual(statistics.median(x["ttft_s"] for x in c1), m["c1_median_ttft_s"])
        rows = [json.loads(x) for x in (result / "resources.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1074)
        self.assertEqual(max(x["gpu_memory_used_mib"] for x in rows), 97280)
        self.assertTrue(all(x.get("container_swap_bytes", 0) == 0 for x in rows))
        self.assertTrue(all(x.get("cgroup_memory_events", {}).get("oom_kill", 0) == 0 for x in rows))
