"""Regression gates for config isolation, locks, downloads, and CLI dispatch."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from recipe_lib import artifacts, cli
from recipe_lib.config import ROOT, ConfigError, load_lock, resolve


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="openseek recipe ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for folder in ("recipes", "profiles", "locks"):
            shutil.copytree(ROOT / folder, self.root / folder)
        shutil.copy(ROOT / "VERSION", self.root / "VERSION")

    def cfg(self, args=None, env=None):
        return resolve(args or {}, self.root, {} if env is None else env)

    def config(self, data):
        path = self.root / ".recipe/config.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def capture(self, argv):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(output):
            code = cli.run(argv, self.root)
        return code, output.getvalue()


class ConfigurationTests(Fixture):
    def test_default_is_offline_and_does_not_write(self):
        cfg = self.cfg()
        self.assertEqual(cfg["model"], "qwen38-flash-next")
        self.assertEqual(cfg["env"]["MAX_BATCHED_TOKENS"], "6399")
        self.assertFalse((self.root / ".recipe").exists())

    def test_flags_process_env_legacy_and_config_precedence(self):
        self.config({"env": {"PORT": "8200"}})
        (self.root / ".env").write_text("PORT=8201\n")
        self.assertEqual(self.cfg()["env"]["PORT"], "8201")
        self.assertEqual(self.cfg(env={"PORT": "8202"})["env"]["PORT"], "8202")
        self.assertEqual(self.cfg({"port": "8203"}, {"PORT": "8202"})["env"]["PORT"], "8203")

    def test_simple_legacy_env_expands_without_execution(self):
        (self.root / ".env").write_text('HF_CACHE="$HOME/cache with spaces"\nRECIPE_DIR=./vendor/pinned\nMAX_BATCHED_TOKENS=\n')
        cfg = self.cfg(env={"HOME": str(self.root)})
        self.assertEqual(cfg["env"]["HF_CACHE"], str(self.root / "cache with spaces"))
        self.assertEqual(cfg["env"]["MAX_BATCHED_TOKENS"], "6399")

    def test_no_shell_substitution(self):
        marker = self.root / "should-not-exist"
        (self.root / ".env").write_text(f"HF_CACHE=$(touch '{marker}')\n")
        with self.assertRaisesRegex(ConfigError, "shell execution"):
            self.cfg()
        self.assertFalse(marker.exists())

    def test_version_settings_cannot_drift(self):
        for key in ("RECIPE_COMMIT", "BASE_IMAGE", "LMCACHE_VERSION", "KV_IMAGE"):
            with self.subTest(key=key), self.assertRaisesRegex(ConfigError, "version-locked"):
                self.cfg(env={key: "different"})

    def test_incompatible_profile_and_unknown_model(self):
        with self.assertRaises(ConfigError):
            self.cfg({"profile": "small-gpu"})
        with self.assertRaises(ConfigError):
            self.cfg({"model": "missing"})

    def test_invalid_inputs_fail_before_dispatch(self):
        for env in ({"PORT": "no"}, {"PORT": "6555"}, {"MAX_BATCHED_TOKENS": "1"},
                    {"GPU_MEMORY_UTILIZATION": "nan"}, {"GPU": "-1"}, {"KV_BLOCK_TOKENS": "6400"},
                    {"QUANT": "nvfp4"}, {"MTP_TOKENS": "2"}, {"MAX_MODEL_LEN": "1000000"},
                    {"BASE_URL": "https://user:secret@example.com"}):
            with self.subTest(env=env), self.assertRaises(ConfigError):
                self.cfg(env=env)

    def test_existing_runtime_names_are_preserved(self):
        cfg = self.cfg()
        self.assertEqual(cfg["recipe"]["served_model"], "qwen38-exl3-ple8")
        self.assertTrue(cfg["env"]["KV_IMAGE"].endswith(cfg["lock_sha256"][:12]))

    def test_secrets_not_in_config_output(self):
        cfg = self.cfg(env={"HF_TOKEN": "secret", "OPENAI_API_KEY": "other-secret"})
        self.assertNotIn("secret", json.dumps(cfg))

    def test_explicit_missing_config_is_an_error(self):
        with self.assertRaisesRegex(ConfigError, "Config not found"):
            self.cfg({"command": "start", "config": str(self.root / "absent.json")})

    def test_lock_rejects_floating_revisions_and_unsafe_files(self):
        path = self.root / "locks/qwen38-exl3-ple8.lock.json"
        original = json.loads(path.read_text())
        for change in ("revision", "image", "path", "duplicate"):
            bad = copy.deepcopy(original)
            if change == "revision": bad["model"]["revision"] = "main"
            if change == "image": bad["runtime"]["base_image"] = "runtime:latest"
            if change == "path": bad["model"]["files"][0]["path"] = "../escape"
            if change == "duplicate": bad["model"]["files"].append(bad["model"]["files"][0])
            path.write_text(json.dumps(bad))
            with self.subTest(change=change), self.assertRaises(ConfigError):
                load_lock(path)


class CliTests(Fixture):
    def test_dry_run_install_does_not_spawn_or_write(self):
        with patch("subprocess.run", side_effect=AssertionError("must not spawn")):
            code, output = self.capture(["install", "--dry-run", "--json"])
        self.assertEqual(code, 0)
        plan = json.loads(output)
        self.assertEqual([Path(c[1]).name for c in plan["commands"]], ["doctor.sh", "setup.sh"])
        self.assertFalse((self.root / ".recipe").exists())

    def test_global_flags_work_on_either_side_of_subcommand(self):
        for argv in (["--port", "8999", "connection"], ["connection", "--port", "8999"]):
            _, output = self.capture(argv)
            self.assertEqual(json.loads(output)["base_url"], "http://127.0.0.1:8999/v1")

    def test_help_ignores_invalid_arguments(self):
        code, output = self.capture(["install", "--nonsense", "--help"])
        self.assertEqual(code, 0)
        self.assertIn("install", output)

    def test_chat_is_external(self):
        with self.assertRaisesRegex(ConfigError, "OpenSeek TUI"):
            cli.run(["-"], self.root)

    def test_model_selection_writes_only_local_config(self):
        _, output = self.capture(["models", "select", "qwen38-flash-next"])
        self.assertEqual(json.loads(output)["selection"]["model"], "qwen38-flash-next")
        self.assertTrue((self.root / ".recipe/config.json").exists())

    def test_config_init_does_not_overwrite(self):
        self.capture(["config", "init"])
        with self.assertRaisesRegex(ConfigError, "already exists"):
            self.capture(["config", "init"])

    def test_client_smoke_passes_selected_model_and_url(self):
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.capture(["smoke", "--base-url", "http://localhost:8222/v1"])
        self.assertEqual(run.call_args.args[0][-4:], ["--base-url", "http://localhost:8222", "--model", "qwen38-exl3-ple8"])

    def test_failed_install_stops_and_writes_no_receipt(self):
        with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="x86_64"), \
             patch("subprocess.run", return_value=subprocess.CompletedProcess([], 17)) as run:
            code, _ = self.capture(["install"])
        self.assertEqual(code, 17)
        self.assertEqual(run.call_count, 1)
        self.assertFalse((self.root / ".recipe/installed-lock.json").exists())

    def test_install_passes_resolved_env_and_records_success(self):
        (self.root / ".env").write_text("PORT=8220\n")
        with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="x86_64"), \
             patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            code, _ = self.capture(["install", "--port", "8221"])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.kwargs["env"]["PORT"], "8221")
        self.assertEqual(run.call_args.kwargs["env"]["OPENSEEK_RESOLVED"], "1")
        self.assertTrue((self.root / ".recipe/installed-lock.json").exists())

    def test_json_not_mixed_with_script_logs(self):
        with self.assertRaisesRegex(ConfigError, "streams"):
            self.capture(["install", "--json"])


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name)
        self.payloads = {"config.json": b"{}", "model-00001.safetensors": b"model-payload"}
        self.lock = {"model": {"repo_id": "owner/model", "revision": "a" * 40,
            "files": [{"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                      for name, data in self.payloads.items()]}}
        self.snapshot = artifacts.snapshot_path(self.lock, self.cache)
        self.snapshot.mkdir(parents=True)

    def fill(self, **kwargs):
        for name, data in self.payloads.items():
            (self.snapshot / name).write_bytes(data)
        self.download_args = kwargs

    def test_config_alone_does_not_mean_complete_download(self):
        (self.snapshot / "config.json").write_bytes(b"{}")
        result = artifacts.download(self.lock, self.cache, self.fill)
        self.assertTrue(result["ok"])
        self.assertEqual(self.download_args["revision"], "a" * 40)
        self.assertEqual(set(self.download_args["allow_patterns"]), set(self.payloads))

    def test_complete_verified_snapshot_reused_without_network(self):
        self.fill()
        with patch("urllib.request.urlopen", side_effect=AssertionError("offline")):
            result = artifacts.download(self.lock, self.cache, lambda **_: self.fail("redownload"))
        self.assertTrue(result["ok"])

    def test_same_size_corruption_is_detected_by_sha256(self):
        self.fill()
        (self.snapshot / "config.json").write_bytes(b"[]")
        self.assertTrue(artifacts.verify(self.lock, self.cache, full=False)["ok"])
        self.assertFalse(artifacts.verify(self.lock, self.cache, full=True)["ok"])
        with self.assertRaisesRegex(ConfigError, "integrity failed"):
            artifacts.download(self.lock, self.cache, lambda **_: self.fail("must not repair silently"))

    def test_interrupted_download_can_be_retried(self):
        def interrupted(**kwargs):
            (self.snapshot / "config.json").write_bytes(b"{}")
            raise OSError("connection lost")
        with self.assertRaises(OSError):
            artifacts.download(self.lock, self.cache, interrupted)
        self.assertTrue(artifacts.download(self.lock, self.cache, self.fill)["ok"])

    def test_partial_or_broken_file_fails(self):
        self.fill()
        (self.snapshot / "model-00001.safetensors").write_bytes(b"short")
        with self.assertRaisesRegex(ConfigError, "wrong sizes"):
            artifacts.download(self.lock, self.cache, self.fill)


if __name__ == "__main__":
    unittest.main()
