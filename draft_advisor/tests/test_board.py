import numpy as np

from draft_advisor import board as B


def test_build_board_shapes_and_wiring():
    bd = B.build_board(ladder="triangular", n_sims=1000, seed=1)
    assert bd.points.shape == (1000, 48)
    assert bd.stages.shape == (1000, 48)
    assert (bd.team_ev >= 0).all()
    assert bd.index_of("Spain") == 28  # config row order
    top8 = set(np.argsort(bd.team_ev)[::-1][:8])
    assert bd.index_of("Spain") in top8  # the clear favourite rates highly


def test_same_seed_different_ladder_changes_ev():
    a = B.build_board(ladder="linear", n_sims=500, seed=3)
    b = B.build_board(ladder="geometric", n_sims=500, seed=3)
    assert not np.allclose(a.team_ev, b.team_ev)


def test_board_roundtrip_and_repro_log(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(B, "_REPRO_DIR", tmp_path)
    bd = B.build_board(ladder="linear", n_sims=300, seed=2)
    path = B.save_board(bd)
    bd2 = B.load_board(path)
    assert np.array_equal(bd.points, bd2.points)
    assert bd.board_sha256 == bd2.board_sha256 and bd.ladder == bd2.ladder
    assert any(f.name.startswith("repro_board_") for f in tmp_path.iterdir())


def test_repro_log_has_no_absolute_path(tmp_path, monkeypatch):
    # identity hygiene: the committed repro record must never embed a home dir / OS username
    monkeypatch.setattr(B, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(B, "_REPRO_DIR", tmp_path)
    B.save_board(B.build_board(ladder="linear", n_sims=200, seed=9))
    txt = next(tmp_path.glob("repro_board_*.json")).read_text()
    assert "\\Users\\" not in txt and "/home/" not in txt and ":\\" not in txt


def test_load_or_build_rebuilds_on_truncated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(B, "_REPRO_DIR", tmp_path)
    bd = B.load_or_build(ladder="linear", n_sims=300, seed=44)  # builds + caches
    cache = next(tmp_path.glob("board_*.npz"))
    cache.write_bytes(cache.read_bytes()[: 1024])  # truncate -> zipfile.BadZipFile on load
    bd2 = B.load_or_build(ladder="linear", n_sims=300, seed=44)  # must rebuild, not raise
    assert bd2.board_sha256 == bd.board_sha256
