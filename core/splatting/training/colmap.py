"""Reading a COLMAP reconstruction.

The second input shape SplatKit accepts. A frameset is a moving subject cut out of its
background; a COLMAP dataset is a static scene with real SfM poses and a sparse cloud, and
it is what almost every other splat tool in the world produces.

Only the binary format is read (`cameras.bin`, `images.bin`, `points3D.bin`), because that
is what COLMAP writes by default and what SplatKit ships.

Two conventions have to be got right or the result is fog:

  * COLMAP stores **world-to-camera**, as a quaternion and translation. The trainer wants
    camera-to-world, so each pose is inverted.
  * COLMAP is **OpenCV** (+x right, +y down, +z forward); this project is OpenGL (+x right,
    +y up, +z back). The y and z columns of the rotation are negated.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# COLMAP camera models, by the id stored in cameras.bin. Only the ones that can be
# expressed as a pinhole are accepted: distortion is not modelled here, and silently
# ignoring it would put every gaussian slightly in the wrong place.
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),      # f, cx, cy
    1: ("PINHOLE", 4),             # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("SPHERE", 3),             # SphereSfM's equirect camera; parsed, never trained
}
PINHOLE_LIKE = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"}
UNDISTORTED = {"SIMPLE_PINHOLE", "PINHOLE"}


def _read(fh, fmt: str):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, fh.read(size))


def read_cameras(path: Path) -> dict[int, dict]:
    cameras = {}
    with open(path, "rb") as fh:
        for _ in range(_read(fh, "<Q")[0]):
            cam_id, model_id, width, height = _read(fh, "<iiQQ")
            name, n_params = CAMERA_MODELS.get(model_id, (f"UNKNOWN_{model_id}", 4))
            params = _read(fh, "<" + "d" * n_params)
            cameras[cam_id] = {"model": name, "width": width, "height": height,
                               "params": np.asarray(params)}
    return cameras


def read_images(path: Path) -> dict[int, dict]:
    images = {}
    with open(path, "rb") as fh:
        for _ in range(_read(fh, "<Q")[0]):
            image_id = _read(fh, "<i")[0]
            qvec = np.asarray(_read(fh, "<dddd"))          # w, x, y, z, world-to-camera
            tvec = np.asarray(_read(fh, "<ddd"))
            cam_id = _read(fh, "<i")[0]
            name = b""
            while True:
                char = fh.read(1)
                if char == b"\x00":
                    break
                name += char
            n_points = _read(fh, "<Q")[0]
            fh.read(24 * n_points)                          # 2D observations, not needed
            images[image_id] = {"qvec": qvec, "tvec": tvec, "camera_id": cam_id,
                                "name": name.decode()}
    return images


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    with open(path, "rb") as fh:
        for _ in range(_read(fh, "<Q")[0]):
            _read(fh, "<Q")                                 # point id
            xyz.append(_read(fh, "<ddd"))
            rgb.append(_read(fh, "<BBB"))
            _read(fh, "<d")                                 # reprojection error
            track = _read(fh, "<Q")[0]
            fh.read(8 * track)
    return np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.uint8)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """wxyz to a 3x3 rotation."""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def intrinsics(camera: dict) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy). Extra distortion terms are dropped, see the module docstring."""
    p = camera["params"]
    if camera["model"] == "SIMPLE_PINHOLE":
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    if camera["model"] in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        # one focal length: f, cx, cy, then the distortion terms
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    return float(p[0]), float(p[1]), float(p[2]), float(p[3])


def load(sparse_dir: str | Path) -> dict:
    """Read a `sparse/0` folder into poses, intrinsics and seed points.

    Returns camera-to-world matrices in the OpenGL convention this project uses, in the
    order the image names sort, so a render sequence matches the folder listing.
    """
    sparse = Path(sparse_dir)
    if not (sparse / "cameras.bin").is_file():
        inner = sparse / "0"
        if (inner / "cameras.bin").is_file():
            sparse = inner
        else:
            raise FileNotFoundError(f"No cameras.bin in {sparse} or {sparse / '0'}")

    cameras = read_cameras(sparse / "cameras.bin")
    images = read_images(sparse / "images.bin")
    points_path = sparse / "points3D.bin"
    xyz, rgb = read_points(points_path) if points_path.is_file() else (
        np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))

    models = {cameras[i["camera_id"]]["model"] for i in images.values()}
    unsupported = models - PINHOLE_LIKE
    if unsupported:
        raise ValueError(
            f"Camera model(s) {sorted(unsupported)} are not pinhole-like. This trainer does "
            "not undistort, so those poses would be wrong. Undistort the dataset first "
            "(colmap image_undistorter).")

    ordered = sorted(images.values(), key=lambda i: i["name"])
    ref_id, keep, dropped = _select_cameras(cameras, ordered)
    c2w, names = [], []
    for image in keep:
        rot = quat_to_matrix(image["qvec"])
        w2c = np.eye(4)
        w2c[:3, :3] = rot
        w2c[:3, 3] = image["tvec"]
        pose = np.linalg.inv(w2c)
        pose[0:3, 1:3] *= -1                                # OpenCV -> OpenGL
        c2w.append(pose)
        names.append(image["name"])

    ref = cameras[ref_id]
    fx, fy, cx, cy = intrinsics(ref)
    return {
        "c2w": np.stack(c2w).astype(np.float32),
        "names": names,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": int(ref["width"]), "height": int(ref["height"]),
        "points": xyz, "colors": rgb,
        "shared_intrinsics": len({i["camera_id"] for i in keep}) == 1,
        "dropped": dropped,
        "models": sorted({cameras[i["camera_id"]]["model"] for i in keep}),
    }


def _normalised(camera: dict) -> np.ndarray:
    """Intrinsics as fractions of the image size: equal for two cameras that see the same
    frustum at different resolutions."""
    fx, fy, cx, cy = intrinsics(camera)
    w, h = float(camera["width"]), float(camera["height"])
    return np.array([fx / w, fy / h, cx / w, cy / h])


def _select_cameras(cameras: dict, ordered: list[dict]) -> tuple[int, list[dict], list[str]]:
    """The reference camera and the images the trainer can use with it.

    The trainer renders every view with one shared pinhole. A SphereSfM dataset can hold
    more than one camera: the hi-res initial pano is its own SPHERE camera, so its cube
    faces are the same 90 degree frustum at a larger size, and Add HiRes Views registers
    perspective views with a field of view of their own. The first kind is kept and
    resized to the reference size on load, which is exact. The second would be rendered
    through the wrong lens, so it is dropped and named rather than trained wrong.

    The reference is the camera most images use; a tie goes to the larger image.
    """
    counts: dict[int, int] = {}
    for image in ordered:
        counts[image["camera_id"]] = counts.get(image["camera_id"], 0) + 1
    ref_id = max(counts, key=lambda c: (counts[c], cameras[c]["width"] * cameras[c]["height"]))
    ref = _normalised(cameras[ref_id])
    keep, dropped = [], []
    for image in ordered:
        same = np.allclose(_normalised(cameras[image["camera_id"]]), ref, rtol=2e-3, atol=2e-3)
        (keep if same else dropped).append(image if same else image["name"])
    return ref_id, keep, dropped
