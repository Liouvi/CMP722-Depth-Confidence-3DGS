import sys
from pathlib import Path

import pytest

from cwdepth.data.views import split_views

REPO_ROOT = Path(__file__).resolve().parents[1]
LLFF_ROOT = REPO_ROOT / "data" / "llff"


def test_split_is_disjoint_and_complete():
    names = [f"img_{i:03d}" for i in range(20)]
    train, test = split_views(names, n_views=0)
    assert test == names[::8]
    assert sorted(train + test) == names


def test_three_views_are_first_middle_last_of_the_training_pool():
    names = [f"img_{i:03d}" for i in range(20)]
    pool, _ = split_views(names, n_views=0)
    train, _ = split_views(names, n_views=3)
    assert train[0] == pool[0] and train[-1] == pool[-1] and len(train) == 3


def test_input_order_does_not_matter():
    names = [f"img_{i:03d}" for i in range(33)]
    assert split_views(names[::-1], 3) == split_views(names, 3)


def test_too_many_views_is_rejected():
    with pytest.raises(ValueError):
        split_views(["a", "b", "c"], n_views=5)


@pytest.mark.parametrize("n_views", [3, 6, 9])
def test_matches_baseline_loader(n_views):
    """The restated rule must pick the same images as the baseline's scene reader."""
    scene_dir = LLFF_ROOT / "fern"
    if not (scene_dir / f"{n_views}_views" / "dense" / "fused.ply").exists():
        pytest.skip("LLFF data not prepared")
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "DropGaussian"))
    reader = pytest.importorskip("scene.dataset_readers")
    info = reader.readColmapSceneInfo(str(scene_dir), "images", True, n_views)
    names = sorted(p.stem for p in (scene_dir / "images").iterdir())
    train, test = split_views(names, n_views)
    assert sorted(c.image_name for c in info.train_cameras) == train
    assert sorted(c.image_name for c in info.test_cameras) == test
