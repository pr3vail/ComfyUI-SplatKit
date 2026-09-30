"""A static scene: the COLMAP dataset the panorama branch builds, trained into one splat.

The backend trainer already reads COLMAP (`core/splatting/training/colmap.py`); this node
is the ComfyUI end of it, so pano -> dataset -> splat runs in one graph. The result is a
one-frame sequence, which is what every sequence node (Sequence Frame, Sequence Player,
Sequence Info, Load Sequence) already takes.
"""

from __future__ import annotations

from pathlib import Path

from ...core.splatting.constants import CATEGORY, NODE_PREFIX, TYPE_SEQUENCE
from ...core.splatting.backend import BackendError, load_config
from ...core.splatting.cache import fingerprint, file_hash
from ...core.splatting.paths import output_root
from ...core.splatting import runtime
from ...core.splatting.security import check_path
from .sequence import QUALITY, TYPE_PERCEPTUAL, _check_perceptual, _train


def resolve_dataset(dataset_dir: str) -> Path:
    """A path, or a dataset name under ComfyUI/output, holding images/ and sparse/."""
    text = (dataset_dir or "").strip().strip('"')
    if not text:
        raise BackendError("dataset_dir is empty. Connect the model_dir output of a SphereSfM "
                           "Dataset node, or type a COLMAP dataset folder.")
    root = Path(text).expanduser()
    if not root.is_dir():
        root = output_root() / text
    root = check_path(root, "dataset_dir")
    if not root.is_dir():
        raise BackendError(f"Dataset folder does not exist: {root}")
    return root


def read_scene(root: Path) -> dict:
    """Validate a COLMAP dataset the way the trainer will read it, before the backend
    starts, and fingerprint exactly the files it will train from."""
    from ...core.splatting.training import colmap
    sparse = root / "sparse"
    if not sparse.is_dir():
        raise BackendError(
            f"{root} has no sparse/ folder. SphereSfM Dataset writes one with "
            "mode=colmap_now; a panorama_only run has to be solved first.")
    try:
        model = colmap.load(sparse)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise BackendError(f"Cannot read the COLMAP model in {sparse}: {exc}") from exc
    distorted = sorted(set(model["models"]) - colmap.UNDISTORTED)
    if distorted:
        # The trainer drops distortion terms; on a real lens that puts every gaussian a
        # little in the wrong place. SphereSfM and Add HiRes Views write pure pinholes.
        raise BackendError(
            f"{root} uses camera model(s) {distorted}, which carry lens distortion the trainer "
            "does not model. Undistort the dataset first (colmap image_undistorter) so it "
            "holds PINHOLE or SIMPLE_PINHOLE cameras.")
    if len(model["points"]) == 0:
        raise BackendError(f"{sparse} has no points3D, so there is nothing to initialise "
                           "the gaussians from. Re-run the SfM with triangulation on.")
    if len(model["names"]) < 3:
        raise BackendError(f"Only {len(model['names'])} usable view(s) in {root}; a scene "
                           "needs many more.")
    images = [(root / "images" / n).resolve() for n in model["names"]]
    outside = [p for p in images if not p.is_relative_to(root.resolve())]
    if outside:
        raise BackendError(f"Image path escapes the dataset folder: {outside[0]}")
    missing = [p for p in images if not p.is_file()]
    if missing:
        raise BackendError(f"{len(missing)} image(s) of the reconstruction are missing from "
                           f"{root / 'images'}, e.g. {missing[0].name}")
    model_dir = sparse / "0" if (sparse / "0" / "cameras.bin").is_file() else sparse
    files = [p for p in model_dir.glob("*.bin")] + images
    return {"views": len(images), "dropped": model["dropped"],
            "points": int(len(model["points"])),
            "size": (model["width"], model["height"]),
            "fingerprint": fingerprint([p.resolve() for p in files], root.resolve())}


class SplatKitTrainScene:
    """COLMAP dataset in, one gaussian splat of the static scene out."""

    CATEGORY = CATEGORY
    FUNCTION = "train"
    RETURN_TYPES = (TYPE_SEQUENCE, "IMAGE", "STRING")
    RETURN_NAMES = ("sequence", "flythrough", "ply_path")
    DESCRIPTION = ("Trains one 3D gaussian splat from a COLMAP dataset: the model_dir of "
                   "SphereSfM Dataset, Add HiRes Views or any other SfM tool (images/ + "
                   "sparse/0). Seeds from the SfM point cloud. 'flythrough' replays the "
                   "captured cameras through the trained splat; 'ply_path' is the finished "
                   "splat for SuperSplat, Blender or Save 3D Model. Needs the splat backend.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Connect SphereSfM Dataset's model_dir (or Add HiRes Views' "
                               "dataset_dir), or type a dataset folder or a dataset name "
                               "under ComfyUI/output."}),
                "perceptual_model": (TYPE_PERCEPTUAL, {"tooltip": "From Splat Perceptual Model Loader."}),
                "quality": (QUALITY, {
                    "default": "standard",
                    "tooltip": "draft: 3k iterations, proves the dataset trains. standard and "
                               "best: 30k iterations, the usual 3DGS budget."}),
                "name": ("STRING", {"default": "my_scene",
                                    "tooltip": "Output folder under ComfyUI/output/. Never "
                                               "overwritten: a repeat gets a numbered suffix."}),
            },
            "optional": {
                "retrain": ("BOOLEAN", {"default": False, "tooltip": "Train into a new folder even when a matching result exists."}),
                "clean": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "After training, remove gaussians that are near-invisible, "
                               "needle-shaped or isolated specks. Off keeps everything the "
                               "optimiser produced."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Recheck the disk cache even when ComfyUI still has this graph in memory.
        return float("nan")

    def train(self, dataset_dir, quality, name, perceptual_model=None, retrain=False, clean=True):
        weights = _check_perceptual(perceptual_model)
        root = resolve_dataset(dataset_dir)
        scene = read_scene(root)
        if scene["dropped"]:
            print(f"[SplatKit] {len(scene['dropped'])} view(s) with a different lens than the "
                  f"other {scene['views']} are left out of training, e.g. {scene['dropped'][0]}")
        config = load_config(require_generator=False)
        signature = {"schema": 1, "kind": "colmap", "runtime": config["runtime_id"],
                     "pack": runtime.pack_version(), "trainer": runtime.trainer_id(),
                     "perceptual_model": file_hash(weights),
                     "source": str(root), "content": scene["fingerprint"],
                     "frames": [0], "quality": quality, "clean": bool(clean)}
        print(f"[SplatKit] scene {root.name}: {scene['views']} views at "
              f"{scene['size'][0]}x{scene['size'][1]}, {scene['points']} seed points")
        extra = ["--no-advect"] + ([] if clean else ["--no-clean"])
        seq, flythrough = _train(config, root, signature, name, quality, weights, extra, 1, retrain)
        ply = seq["ply"][0] if seq["ply"] else ""
        return (seq, flythrough, ply)


NODE_CLASS_MAPPINGS = {NODE_PREFIX + "TrainScene": SplatKitTrainScene}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_PREFIX + "TrainScene": "Train Splat (COLMAP Dataset)"}
