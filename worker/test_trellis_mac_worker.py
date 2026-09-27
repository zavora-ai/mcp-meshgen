"""Tests for the parts of the worker that don't need the model. Run with a Python that has trimesh and Pillow
(trellis-mac's venv has both):  <venv python> -m unittest worker/test_trellis_mac_worker.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = HERE / "trellis_mac_worker.py"


def run_worker(lines, extra_code=None):
    """Run the worker as a subprocess (or a snippet that imports it) and return (protocol events, stderr)."""
    if extra_code is None:
        cmd = [sys.executable, "-u", str(WORKER), str(HERE)]
    else:
        cmd = [sys.executable, "-u", "-c", extra_code]
    proc = subprocess.run(cmd, input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": str(HERE)})
    events = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
    return events, proc.stderr, proc.returncode


class Protocol(unittest.TestCase):
    def test_handshake_ping_errors_and_unload(self):
        events, _, code = run_worker(['{"op":"ping","id":"a"}', "nope", '{"op":"x","id":"b"}', '{"op":"unload"}'])
        self.assertEqual(code, 0)
        self.assertEqual([e["event"] for e in events], ["hello", "pong", "error", "error", "unloaded"])
        self.assertEqual(events[0]["protocol"], 1)
        self.assertEqual(events[2]["kind"], "bad_request")

    def test_eof_ends_the_worker(self):
        events, _, code = run_worker([])
        self.assertEqual((code, [e["event"] for e in events]), (0, ["hello"]))

    def test_library_prints_never_reach_the_protocol_stream(self):
        code = (
            "import trellis_mac_worker as w, os\n"
            "print('chatter from a library')\n"
            "os.write(1, b'C-level chatter\\n')\n"
            "w.emit('probe', ok=True)\n"
        )
        events, stderr, _ = run_worker([], extra_code=code)
        self.assertEqual(events, [{"event": "probe", "ok": True}])
        self.assertIn("chatter from a library", stderr)
        self.assertIn("C-level chatter", stderr)


class Progress(unittest.TestCase):
    def test_tqdm_stand_in_reports_steps_and_time(self):
        code = (
            "import trellis_mac_worker as w\n"
            "job = w.Job('j'); w.Job.current = job\n"
            "items = list(w.progress_tqdm([1, 2, 3], desc='Sampling shape SLat', disable=True))\n"
            "assert items == [1, 2, 3]\n"
            "assert 'shape' in job.timings\n"
        )
        events, stderr, code_ = run_worker([], extra_code=code)
        self.assertEqual(code_, 0, stderr)
        self.assertEqual([(e["stage"], e["step"], e["total"]) for e in events],
                         [("shape", 0, 3), ("shape", 1, 3), ("shape", 2, 3), ("shape", 3, 3)])


class ObjExport(unittest.TestCase):
    def test_textured_obj_files_are_named_after_the_mesh_and_written_as_partials(self):
        with tempfile.TemporaryDirectory() as d:
            code = (
                "import numpy as np, trimesh, trellis_mac_worker as w\n"
                "from PIL import Image\n"
                "m = trimesh.creation.box()\n"
                "uv = np.random.default_rng(0).random((len(m.vertices), 2))\n"
                "tex = Image.fromarray((np.ones((8, 8, 3)) * 200).astype('uint8'))\n"
                "mat = trimesh.visual.material.PBRMaterial(baseColorTexture=tex)\n"
                "m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)\n"
                f"extras = w.export_obj(m, r'{d}/brazier-42.obj.partial')\n"
                "w.emit('extras', files=extras)\n"
            )
            events, stderr, rc = run_worker([], extra_code=code)
            self.assertEqual(rc, 0, stderr)
            names = sorted(os.listdir(d))
            self.assertIn("brazier-42.obj.partial", names)
            self.assertIn("brazier-42.mtl.partial", names)
            textures = [n for n in names if n.startswith("brazier-42_") and n.endswith(".png.partial")]
            self.assertEqual(len(textures), 1, names)
            self.assertTrue(all(n.startswith("brazier-42") and n.endswith(".partial") for n in names), names)
            mtl = Path(d, "brazier-42.mtl.partial").read_text()
            self.assertIn(textures[0][: -len(".partial")], mtl)
            obj = Path(d, "brazier-42.obj.partial").read_text()
            self.assertIn("mtllib brazier-42.mtl", obj)
            self.assertEqual(sorted(Path(p).name for p in events[0]["files"]),
                             sorted(n[: -len(".partial")] for n in names if not n.startswith("brazier-42.obj")))


if __name__ == "__main__":
    unittest.main()
