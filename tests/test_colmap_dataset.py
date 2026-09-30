"""COLMAP input for the static-scene trainer: cameras of more than one size.

A SphereSfM dataset built with a hi-res initial pano holds two cameras: the WAN frames'
cube faces and the pano's larger cube faces, the same 90 degree frustum at two sizes. Add
HiRes Views adds perspective views with a lens of their own. The trainer renders through
one shared pinhole, so the first kind must arrive resized and the second must be left out.

Needs numpy, torch and Pillow; no CUDA.

    python tests/test_colmap_dataset.py
"""
import struct
import sys
import tempfile
import json
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("splatkit_test")
package.__path__ = [str(ROOT)]
sys.modules.setdefault(package.__name__, package)
from splatkit_test.core.splatting.training import colmap, dataset  # noqa: E402
# Package stubs so the node modules import without nodes/__init__.py pulling in ComfyUI.
for _name, _path in (("splatkit_test.nodes", ROOT / "nodes"),
                     ("splatkit_test.nodes.splatting", ROOT / "nodes" / "splatting")):
    _pkg = types.ModuleType(_name)
    _pkg.__path__ = [str(_path)]
    sys.modules.setdefault(_name, _pkg)
from splatkit_test.core.splatting import backend, runtime  # noqa: E402
from splatkit_test.nodes.splatting import scene, sequence as train  # noqa: E402

MODEL_IDS = {"SIMPLE_PINHOLE": 0, "PINHOLE": 1, "RADIAL": 3, "SPHERE": 11}


def write_model(root: Path, cameras: dict, images: list[tuple[str, int]], points: int = 8):
    """cameras: {id: (model, width, height, params)}; images: [(name, camera_id)]."""
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    (root / "images").mkdir()
    with open(sparse / "cameras.bin", "wb") as fh:
        fh.write(struct.pack("<Q", len(cameras)))
        for cid, (model, w, h, params) in cameras.items():
            fh.write(struct.pack("<iiQQ", cid, MODEL_IDS[model], w, h))
            fh.write(struct.pack("<" + "d" * len(params), *params))
    with open(sparse / "images.bin", "wb") as fh:
        fh.write(struct.pack("<Q", len(images)))
        for i, (name, cid) in enumerate(images, start=1):
            # identity rotation, cameras spread along x so the poses normalise cleanly
            fh.write(struct.pack("<idddddddi", i, 1, 0, 0, 0, float(i), 0, 0, cid))
            fh.write(name.encode() + b"\x00")
            fh.write(struct.pack("<Q", 0))
            _, w, h, _ = cameras[cid]
            Image.new("RGB", (w, h), (i * 20, 0, 0)).save(root / "images" / name)
    with open(sparse / "points3D.bin", "wb") as fh:
        fh.write(struct.pack("<Q", points))
        for p in range(points):
            fh.write(struct.pack("<QdddBBBd", p, p * 0.1, 0.5, 3.0, 128, 128, 128, 0.5))
            fh.write(struct.pack("<Q", 0))


class ColmapCameraTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_same_frustum_at_two_sizes_is_resized_to_the_reference(self):
        write_model(self.root, {
            1: ("SIMPLE_PINHOLE", 64, 64, (32.0, 32.0, 32.0)),      # WAN cube faces
            2: ("SIMPLE_PINHOLE", 128, 128, (64.0, 64.0, 64.0)),    # hi-res pano faces
        }, [(f"f{i}.png", 1) for i in range(4)] + [("pano_0.png", 2), ("pano_1.png", 2)])
        model = colmap.load(self.root / "sparse")
        self.assertEqual(len(model["names"]), 6)
        self.assertEqual(model["dropped"], [])
        self.assertEqual((model["width"], model["height"], model["fx"]), (64, 64, 32.0))

        ds = dataset.ColmapDataset(self.root)
        images = ds.images(device="cpu")
        self.assertEqual(len(images), ds.num_cameras)
        for i in range(len(images)):
            self.assertEqual(tuple(images[i].shape), (64, 64, 3))
        self.assertEqual(len(ds.cameras(device="cpu")), 6)

    def test_a_different_lens_is_dropped_and_named(self):
        write_model(self.root, {
            1: ("SIMPLE_PINHOLE", 64, 64, (32.0, 32.0, 32.0)),
            2: ("PINHOLE", 64, 48, (80.0, 80.0, 32.0, 24.0)),       # a HiRes perspective view
        }, [(f"f{i}.png", 1) for i in range(3)] + [("hires_000.png", 2)])
        model = colmap.load(self.root / "sparse")
        self.assertEqual(model["names"], ["f0.png", "f1.png", "f2.png"])
        self.assertEqual(model["dropped"], ["hires_000.png"])
        self.assertEqual(model["c2w"].shape, (3, 4, 4))

    def test_the_reference_is_the_camera_most_images_use(self):
        write_model(self.root, {
            1: ("PINHOLE", 64, 48, (80.0, 80.0, 32.0, 24.0)),
            2: ("SIMPLE_PINHOLE", 64, 64, (32.0, 32.0, 32.0)),
        }, [("a.png", 1)] + [(f"f{i}.png", 2) for i in range(3)])
        model = colmap.load(self.root / "sparse")
        self.assertEqual((model["width"], model["height"]), (64, 64))
        self.assertEqual(model["dropped"], ["a.png"])

    def test_a_sphere_camera_parses_and_is_refused(self):
        # SPHERE has 3 params; reading it with the wrong count would misparse every camera
        # after it instead of giving this clear error.
        write_model(self.root, {
            1: ("SPHERE", 128, 64, (1.0, 64.0, 32.0)),
            2: ("SIMPLE_PINHOLE", 64, 64, (32.0, 32.0, 32.0)),
        }, [("equirect.png", 1), ("f0.png", 2)])
        with self.assertRaisesRegex(ValueError, "SPHERE"):
            colmap.load(self.root / "sparse")


class TrainSceneNodeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = self.root / "dataset"
        self.data.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def cameras(self):
        return {1: ("SIMPLE_PINHOLE", 64, 64, (32.0, 32.0, 32.0))}

    def test_read_scene_fingerprints_the_model_and_its_images(self):
        write_model(self.data, self.cameras(), [(f"f{i}.png", 1) for i in range(4)])
        first = scene.read_scene(self.data)
        self.assertEqual((first["views"], first["points"], first["size"]), (4, 8, (64, 64)))
        Image.new("RGB", (64, 64), (1, 2, 3)).save(self.data / "images" / "f2.png")
        self.assertNotEqual(scene.read_scene(self.data)["fingerprint"], first["fingerprint"])

    def test_read_scene_refuses_what_the_trainer_cannot_use(self):
        with self.assertRaisesRegex(backend.BackendError, "no sparse/"):
            scene.read_scene(self.data)
        write_model(self.data, self.cameras(), [(f"f{i}.png", 1) for i in range(4)], points=0)
        with self.assertRaisesRegex(backend.BackendError, "no points3D"):
            scene.read_scene(self.data)

    def test_read_scene_refuses_lens_distortion(self):
        write_model(self.data, {1: ("RADIAL", 64, 64, (32.0, 32.0, 32.0, 0.1, 0.01))},
                    [(f"f{i}.png", 1) for i in range(4)])
        with self.assertRaisesRegex(backend.BackendError, "RADIAL"):
            scene.read_scene(self.data)

    def test_radial_reads_one_focal_length(self):
        write_model(self.data, {1: ("RADIAL", 64, 48, (40.0, 32.0, 24.0, 0.1, 0.01))},
                    [(f"f{i}.png", 1) for i in range(3)])
        model = colmap.load(self.data / "sparse")
        self.assertEqual((model["fx"], model["fy"], model["cx"], model["cy"]), (40.0, 40.0, 32.0, 24.0))

    def test_read_scene_names_a_missing_image(self):
        write_model(self.data, self.cameras(), [(f"f{i}.png", 1) for i in range(4)])
        (self.data / "images" / "f3.png").unlink()
        with self.assertRaisesRegex(backend.BackendError, "missing"):
            scene.read_scene(self.data)

    def test_a_dataset_name_resolves_under_output(self):
        with patch.object(scene, "output_root", return_value=self.root), \
             patch.object(scene, "check_path", side_effect=lambda p, _: Path(p).resolve()):
            self.assertEqual(scene.resolve_dataset("dataset"), self.data.resolve())
            with self.assertRaisesRegex(backend.BackendError, "empty"):
                scene.resolve_dataset("  ")

    def test_training_runs_the_worker_on_the_dataset_without_advection(self):
        write_model(self.data, self.cameras(), [(f"f{i}.png", 1) for i in range(4)])
        output = self.root / "out"
        output.mkdir()
        weights = self.root / "vgg.safetensors"
        weights.write_bytes(b"weights")
        seq = {"dir": str(output), "preview": ["p.png"], "ply": [str(output / "ply" / "frame_00000.ply")]}
        mm = types.ModuleType("comfy.model_management")
        mm.unload_all_models = Mock()
        mm.soft_empty_cache = Mock()
        comfy = types.ModuleType("comfy")
        comfy.model_management = mm
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": mm}), \
             patch.object(scene, "load_config", return_value={"python": sys.executable, "runtime_id": "t"}), \
             patch.object(scene, "check_path", side_effect=lambda p, _: Path(p).resolve()), \
             patch.object(train, "check_path", side_effect=lambda p, _: Path(p)), \
             patch.object(train, "_finished_sequence", return_value=None), \
             patch.object(train, "sequence_dir", return_value=output), \
             patch.object(train, "run") as run, \
             patch.object(train, "read_sequence", return_value=seq), \
             patch.object(train, "load_images", return_value="images"):
            result = scene.SplatKitTrainScene().train(str(self.data), "draft", "room",
                                                      perceptual_model=str(weights), clean=False)
        self.assertEqual(result, (seq, "images", seq["ply"][0]))
        args = run.call_args.args[0]
        self.assertEqual(args[1:4], [str(runtime.WORKER), "train", str(self.data.resolve())])
        self.assertIn("--no-advect", args)
        self.assertIn("--no-clean", args)
        self.assertNotIn("--frames", args)
        cached = json.loads((output / "splatkit_cache.json").read_text())
        self.assertEqual((cached["kind"], cached["frames"], cached["clean"]), ("colmap", [0], False))


if __name__ == "__main__":
    unittest.main()
