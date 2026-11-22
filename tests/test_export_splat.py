import numpy as np

from cwdepth.export_splat import quat_multiply, quat_to_matrix, rotation_to_y, scene_frame


def random_quats(n, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_quaternion_product_composes_rotations():
    for a, b in zip(random_quats(20, 1), random_quats(20, 2)):
        assert np.allclose(quat_to_matrix(quat_multiply(a, b)), quat_to_matrix(a) @ quat_to_matrix(b))


def test_rotation_to_y_takes_up_to_plus_y():
    for up in [*np.random.default_rng(3).normal(size=(20, 3)), np.array([0, -1.0, 0]), np.array([0, 2.0, 0])]:
        r = quat_to_matrix(rotation_to_y(up))
        assert np.allclose(r @ (up / np.linalg.norm(up)), [0, 1, 0])
        assert np.isclose(np.linalg.det(r), 1.0)


def look_at(position, target, up):
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return {"position": position.tolist(), "rotation": np.stack([right, down, forward], axis=1).tolist()}


def test_scene_frame_finds_orbit_centre_and_up():
    target, up = np.array([1.0, -2.0, 3.0]), np.array([0.3, -1.0, 0.2])
    up /= np.linalg.norm(up)
    side = np.cross(up, [1.0, 0, 0])
    side /= np.linalg.norm(side)
    other = np.cross(up, side)
    cams = [look_at(target + 4 * (np.cos(t) * side + np.sin(t) * other) + 0.5 * up, target, up)
            for t in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    q, centre = scene_frame(cams)
    assert np.allclose(centre, target, atol=1e-6)
    # Cameras look slightly down, so their mean up is tilted from the true up by a few degrees only.
    assert quat_to_matrix(q) @ up @ [0, 1, 0] > 0.98
