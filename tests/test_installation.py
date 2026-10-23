"""Run the real installer with a local Git fixture and a Docker boundary fake.

No GPU/image/weight downloads. Exercises shell orchestration and real file hashes.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from recipe_lib.config import ROOT, resolve


DOCKER = r'''#!/usr/bin/env python3
import json,os,pathlib,sys
from recipe_lib.artifacts import download
from recipe_lib.config import load_lock
a=sys.argv[1:]
state=pathlib.Path(os.environ['FAKE_STATE'])
with (state/'calls.jsonl').open('a') as f: f.write(json.dumps(a)+'\n')
if a[:2]==['image','inspect']:
    if a[-1]==os.environ['KV_IMAGE'] and not (state/'built').exists():sys.exit(1)
    if '-f' in a:
        print(('wrong' if os.environ.get('FAKE_LABEL_BAD') else os.environ['LOCK_SHA256']) if 'Labels' in a[a.index('-f')+1] else 'sha256:fixture')
elif a and a[0]=='build':
    dockerfile=pathlib.Path(a[-1])/'Dockerfile'
    assert os.environ['BASE_IMAGE'] in dockerfile.read_text()
    assert os.environ['LOCK_SHA256'] in dockerfile.read_text()
    (state/'built').touch()
elif a and a[0]=='run':
    mounts={}
    for i,x in enumerate(a):
        if x=='-v':
            host,inside,*_=a[i+1].split(':');mounts[inside]=pathlib.Path(host)
    if 'download' in a and 'pip' in a:
        assert '--require-hashes' in a
        req=(mounts['/out']/'requirements.txt').read_text()
        assert os.environ['LMCACHE_WHEEL_SHA256'] in req
        (mounts['/out']/os.environ['LMCACHE_WHEEL']).write_bytes(b'wheel-boundary-fixture')
    elif 'recipe_lib.artifacts' in a:
        lock=load_lock(pathlib.Path(os.environ['MODEL_LOCK']))
        cache=mounts['/hf']
        def fetch(**kwargs):
            target=cache/'hub'/('models--'+kwargs['repo_id'].replace('/','--'))/'snapshots'/kwargs['revision']
            target.mkdir(parents=True,exist_ok=True)
            (target/'config.json').write_bytes(b'{}')
            if os.environ.get('FAIL_DOWNLOAD')=='1':raise OSError('interrupted download fixture')
            (target/'weights.bin').write_bytes(b'weights')
            (state/'downloaded').touch()
        print(json.dumps(download(lock,cache,fetch)))
    else:print('fixture: import check / cleanup boundary')
'''


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="recipe install ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "scripts", self.root / "scripts")
        self.vendor = self.root / "vendor/runtime"
        self.vendor.mkdir(parents=True)
        (self.vendor / "start.sh").write_text("#!/bin/bash\n")
        (self.vendor / "start.sh").chmod(0o755)
        (self.vendor / "model-profiles.sh").write_text("MODEL_REPO=fixture/model\nMODEL_REVISION=" + "a" * 40 + "\n")
        for command in (["git", "init", "-q"], ["git", "add", "start.sh", "model-profiles.sh"],
                        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.com", "commit", "-qm", "fixture"]):
            subprocess.run(command, cwd=self.vendor, check=True, capture_output=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.vendor, text=True).strip()
        cfg = resolve({}, ROOT, {})
        self.lock = cfg["lock"]
        self.lock["model"] = {"repo_id": "fixture/model", "revision": "a" * 40, "files": [
            {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in {"config.json": b"{}", "weights.bin": b"weights"}.items()]}
        (self.root / "locks").mkdir()
        self.lockpath = self.root / "locks/fixture.lock.json"
        self.lockpath.write_text(json.dumps(self.lock))
        self.state = self.root / "state"
        self.state.mkdir()
        fakebin = self.root / "bin"
        fakebin.mkdir()
        (fakebin / "docker").write_text(DOCKER)
        (fakebin / "docker").chmod(0o755)
        self.env = {**os.environ, **cfg["env"], "RECIPE_DIR": str(self.vendor), "RECIPE_COMMIT": commit,
                    "LOCKED_MODEL_REPO": "fixture/model", "LOCKED_MODEL_REVISION": "a" * 40,
                    "HF_CACHE": str(self.root / "hf cache"), "MODEL_LOCK": str(self.lockpath),
                    "LOCK_SHA256": hashlib.sha256(self.lockpath.read_bytes()).hexdigest(),
                    "PATH": str(fakebin) + os.pathsep + os.environ["PATH"], "PYTHONPATH": str(ROOT),
                    "FAKE_STATE": str(self.state)}

    def install(self, **env):
        return subprocess.run(["bash", str(self.root / "scripts/setup.sh")], env={**self.env, **env},
                              cwd=self.root, text=True, capture_output=True)

    def test_install_retry_and_cached_repeat(self):
        first = self.install(FAIL_DOWNLOAD="1")
        self.assertNotEqual(first.returncode, 0)
        self.assertIn("interrupted download", first.stderr)
        second = self.install()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertTrue((self.state / "downloaded").exists())
        (self.state / "downloaded").unlink()
        third = self.install()
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertFalse((self.state / "downloaded").exists(), "complete cache should skip SDK download")
        calls = [json.loads(s) for s in (self.state / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(sum(c[0] == "build" for c in calls), 1)
        self.assertFalse(any(c[0] in ("stop", "restart", "rm") for c in calls))

    def test_wrong_size_and_same_size_corruption_are_not_accepted(self):
        first = self.install()
        self.assertEqual(first.returncode, 0, first.stderr)
        snapshot = Path(self.env["HF_CACHE"]) / "hub/models--fixture--model/snapshots" / ("a" * 40)
        for corrupted, message in ((b"x", "wrong sizes"), (b"badbyte", "integrity failed")):
            (snapshot / "weights.bin").write_bytes(corrupted)
            result = self.install()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)

    def test_tracked_vendor_changes_are_preserved(self):
        (self.vendor / "start.sh").write_text("user changes\n")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tracked changes", result.stderr)
        self.assertEqual((self.vendor / "start.sh").read_text(), "user changes\n")
        self.assertFalse((self.state / "calls.jsonl").exists())

    def test_lock_validation_happens_before_container_mutation(self):
        self.assertEqual(self.install().returncode, 0)
        for command in ("start", "restart", "recreate"):
            (self.state / "calls.jsonl").write_text("")
            result = subprocess.run(["bash", str(self.root / "scripts/ctl.sh"), command],
                                    env={**self.env, "FAKE_LABEL_BAD": "1"}, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Image lock does not match", result.stderr)
            calls = [json.loads(s) for s in (self.state / "calls.jsonl").read_text().splitlines()]
            self.assertFalse(any(c[0] in ("stop", "restart", "rm", "start", "run") for c in calls))

    def test_incomplete_checkpoint_does_not_remove_an_existing_service(self):
        self.assertEqual(self.install().returncode, 0)
        missing_cache = str(self.root / "empty cache")
        (self.state / "calls.jsonl").write_text("")
        result = subprocess.run(["bash", str(self.root / "scripts/ctl.sh"), "recreate"],
                                env={**self.env, "HF_CACHE": missing_cache}, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("before changing containers", result.stderr)
        calls = [json.loads(s) for s in (self.state / "calls.jsonl").read_text().splitlines()]
        self.assertFalse(any(c[0] in ("stop", "restart", "rm", "start", "run") for c in calls))


if __name__ == "__main__":
    unittest.main()
