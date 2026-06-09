import json

import numpy as np
import pytest

from draft_advisor import advisor as A
from draft_advisor import board as B
from wcpool import metrics as M
from wcpool.config import load_field
from wcpool.ladders import get_ladder
from wcpool.scoring import GROUP_POINT_SCHEMES, ScoringScheme, team_points
from wcpool.strength import StrengthModel
from wcpool.tournament import simulate_tournament


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


# --- group-stage scoring scheme (milestone 6, plan section 9f) --------------------------


def _independent_sim(seed: int, n_sims: int):
    """Reproduce build_board's exact tournament batch (same field/seed/official draw)."""
    field = load_field(B._DEFAULT_FIELD)
    model = StrengthModel(field.elo.copy())
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    stages, group = simulate_tournament(model, field.fixed_groups, n_sims, rng,
                                        return_group_results=True)
    return stages, group, field


def test_default_board_uses_recommended_scheme():
    # No ladder/scheme args -> the published recommendation (triangular 3:1 @ gamma_match, mix=0.1).
    bd = B.build_board(n_sims=500, seed=7)
    assert bd.scheme == B.RECOMMENDED_SCHEME
    assert (bd.knockout_ladder, bd.w_pts, bd.d_pts, bd.mix) == ("triangular", 3.0, 1.0, 0.1)
    # mix > 0 -> the group layer is live, so a team's points are NOT a bare ladder lookup.
    assert not np.array_equal(bd.points, get_ladder("triangular")[bd.stages])


def test_board_points_equal_team_points_under_scheme():
    # The board's points must be EXACTLY scoring.team_points(stages, grp, scheme) for a fixed seed.
    scheme = B.RECOMMENDED_SCHEME
    bd = B.build_board(n_sims=800, seed=20260609, scheme=scheme)
    stages, group, _ = _independent_sim(20260609, 800)
    assert np.array_equal(bd.stages, stages)  # rng stream unaffected by return_group_results
    assert np.array_equal(bd.points, team_points(stages, group, scheme))
    assert np.array_equal(bd.team_ev, team_points(stages, group, scheme).mean(axis=0))


def test_ladder_alone_is_legacy_terminal_scoring():
    # `--ladder X` with no group flags maps to mix=0: the bare terminal ladder, bit-for-bit.
    bd = B.build_board(ladder="geometric", n_sims=400, seed=5)
    assert bd.mix == 0.0 and bd.knockout_ladder == "geometric"
    assert np.array_equal(bd.points, get_ladder("geometric")[bd.stages])


def test_cache_invalidated_when_scheme_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(B, "_REPRO_DIR", tmp_path)
    scheme_a = ScoringScheme("triangular", 1.0, 0.0, 0.0)  # legacy terminal
    scheme_b = ScoringScheme("triangular", 3.0, 1.0, 0.1)  # recommended blend
    bd_a = B.load_or_build(n_sims=300, seed=44, scheme=scheme_a)  # builds + caches under A
    bd_b = B.load_or_build(n_sims=300, seed=44, scheme=scheme_b)  # must NOT reuse A's cache
    assert bd_b.scheme == scheme_b
    # Same seed/draw -> identical stages, but different scoring -> a different board content hash.
    assert np.array_equal(bd_a.stages, bd_b.stages)
    assert bd_b.board_sha256 != bd_a.board_sha256
    assert not np.array_equal(bd_a.points, bd_b.points)
    # Each scheme persisted to its own cache file (scheme is part of the cache key).
    assert len(list(tmp_path.glob("board_*.npz"))) == 2
    # Re-requesting A returns A's cached board unchanged (the right file is keyed back).
    assert B.load_or_build(n_sims=300, seed=44, scheme=scheme_a).board_sha256 == bd_a.board_sha256


def test_mix0_schemes_collide_and_cache_hits_across_wd(tmp_path, monkeypatch):
    """Locks the mix==0 cache-fixpoint invariant (the cache-thrash remediation).

    At mix==0 the group layer is inert (``team_points`` short-circuits to bare ``ko_vec[stages]``
    so ``(w_pts, d_pts)`` never touch the board), and ``_scheme_tag`` folds every mix==0 scheme onto
    the bare-ladder filename. Two mix==0 schemes that differ ONLY in w_pts/d_pts therefore (a) share
    a cache path and (b) build a byte-identical board -- a benign collision. After canonicalising
    the store/guard boundary, (c) requesting the second is a TRUE cache hit: load_or_build must NOT
    rebuild (no second build, no second repro-log) -- the fix that stops the unbounded repro-log
    growth on a ``--ladder linear --w-pts 5 --mix 0`` <-> legacy alternation.
    """
    monkeypatch.setattr(B, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(B, "_REPRO_DIR", tmp_path)
    # Two mix==0 schemes on the same ladder, differing only in the (inert) group rates. The second
    # is exactly the non-canonical scheme reachable from `--ladder linear --w-pts 5 --mix 0`.
    s1 = ScoringScheme("linear", 1.0, 0.0, 0.0)
    s2 = ScoringScheme("linear", 5.0, 2.0, 0.0)

    # (a) same cache path -- the collision the fix must make benign rather than thrash.
    assert B._cache_path(s1, 300, 77, True) == B._cache_path(s2, 300, 77, True)

    # Count board builds directly: load_or_build must re-enter build_board exactly once across the
    # pair (the second request is served from cache), the definitive "no rebuild" instrument.
    real_build = B.build_board
    calls = {"n": 0}

    def _counting_build(*a, **k):
        calls["n"] += 1
        return real_build(*a, **k)

    monkeypatch.setattr(B, "build_board", _counting_build)

    bd1 = B.load_or_build(n_sims=300, seed=77, scheme=s1)  # builds + caches (build #1)
    n_logs_after_first = len(list(tmp_path.glob("repro_board_*.json")))
    npz = next(tmp_path.glob("board_*.npz"))
    mtime_after_first = npz.stat().st_mtime_ns

    bd2 = B.load_or_build(n_sims=300, seed=77, scheme=s2)  # MUST hit cache (no build #2)

    # (b) the collision is benign: identical points (and content hash) regardless of w/d.
    assert np.array_equal(bd1.points, bd2.points)
    assert bd1.board_sha256 == bd2.board_sha256
    # (c) cache hit: build_board entered once, the .npz untouched, and no extra repro-log written.
    assert calls["n"] == 1
    assert npz.stat().st_mtime_ns == mtime_after_first
    assert len(list(tmp_path.glob("repro_board_*.json"))) == n_logs_after_first
    assert len(list(tmp_path.glob("board_*.npz"))) == 1  # one shared cache file, not two


def test_negative_group_points_rejected():
    # Item 3: a fat-fingered negative group rate (e.g. `--w-pts -3`) is rejected at construction,
    # before any garbage board is built. mix is irrelevant -- the rule holds even where w/d inert.
    with pytest.raises(ValueError, match="w_pts must be >= 0"):
        ScoringScheme("triangular", -3.0, 1.0, 0.1)
    with pytest.raises(ValueError, match="d_pts must be >= 0"):
        ScoringScheme("triangular", 3.0, -1.0, 0.1)
    # Every legitimate scheme is non-negative: the recommended (3, 1) and all GROUP_POINT_SCHEMES.
    ScoringScheme("triangular", 3.0, 1.0, 0.1)  # recommended -- constructs fine
    for w, d in GROUP_POINT_SCHEMES.values():
        ScoringScheme("triangular", w, d, 0.1)  # 3/1, 2/1, 1/0 -- all valid


def test_stale_scheme_meta_forces_rebuild(tmp_path, monkeypatch):
    # Defensive: if a cache FILE for scheme B somehow holds a board scored under scheme A
    # (hand-edited / hash-collided meta), the scheme-equality guard must rebuild, not serve it.
    monkeypatch.setattr(B, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(B, "_REPRO_DIR", tmp_path)
    scheme_a = ScoringScheme("linear", 1.0, 0.0, 0.0)
    scheme_b = ScoringScheme("triangular", 3.0, 1.0, 0.1)
    bd_a = B.load_or_build(n_sims=250, seed=8, scheme=scheme_a)
    a_file = next(tmp_path.glob("board_*.npz"))
    # Move A's content onto B's cache filename: B will find a file whose stored scheme is A.
    b_file = B._cache_path(scheme_b, 250, 8, True)
    a_file.replace(b_file)
    bd_b = B.load_or_build(n_sims=250, seed=8, scheme=scheme_b)  # guard: scheme mismatch -> rebuild
    assert bd_b.scheme == scheme_b
    assert bd_b.board_sha256 != bd_a.board_sha256


def test_load_or_build_rebuilds_on_prerefactor_cache(tmp_path, monkeypatch):
    # A cache written BEFORE this milestone stored the bare `ladder` key (no scheme fields). Its
    # `Board(**meta)` reload now raises TypeError; load_or_build must treat that as stale and
    # rebuild (the old terminal scheme), not crash -- the plan's "old-scheme cache -> rebuild".
    monkeypatch.setattr(B, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(B, "_REPRO_DIR", tmp_path)
    scheme = ScoringScheme("linear", 1.0, 0.0, 0.0)  # mix=0 -> filename matches the legacy tag
    bd = B.build_board(n_sims=200, seed=3, scheme=scheme)
    cache_path = B._cache_path(scheme, 200, 3, True)
    # Hand-write a PRE-REFACTOR .npz: meta carries `ladder`, not knockout_ladder/w_pts/d_pts/mix.
    old_meta = {
        "ladder": "linear", "n_sims": 200, "seed": 3, "use_official_draw": True,
        "config_sha256": bd.config_sha256, "board_sha256": bd.board_sha256,
    }
    np.savez_compressed(
        cache_path, points=bd.points, stages=bd.stages, team_ev=bd.team_ev,
        group_of=bd.group_of, names=np.array(bd.names), meta=json.dumps(old_meta),
    )
    rebuilt = B.load_or_build(n_sims=200, seed=3, scheme=scheme)  # must rebuild, not TypeError
    assert rebuilt.scheme == scheme
    assert rebuilt.board_sha256 == bd.board_sha256  # same seed/draw/scheme -> identical board


def test_repro_log_carries_scheme(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "_ARTIFACTS", tmp_path)
    monkeypatch.setattr(B, "_REPRO_DIR", tmp_path)
    B.save_board(B.build_board(n_sims=200, seed=9, scheme=B.RECOMMENDED_SCHEME))
    rec = json.loads(next(tmp_path.glob("repro_board_*.json")).read_text())
    assert rec["scoring_scheme"] == {
        "knockout_ladder": "triangular", "w_pts": 3.0, "d_pts": 1.0, "mix": 0.1,
    }
    assert rec["ladder"] == "triangular"  # back-compat alias retained


def test_recommended_blend_matches_published_integer_ladder():
    """Affine-equivalence proof (plan section 9f): the recommended continuous blend and the
    published integer ladder [9,27,54,90,135,189] + 3/1 group points are *one scoring rule up to a
    positive scale* -- algebraically ``blend.points == 0.1 * int_points`` (a = 0.1 > 0, c = 0).

    The objective ``W = (n-1)*P(1st) + P(2nd)`` is a rank functional of the per-drafter pool scores,
    so it is invariant to a positive rescale of the points: every drafter's score scales by 10, the
    per-sim ranking (hence the fractional 1st/2nd credit, hence P(1st)/P(2nd), hence W) is the same.
    W is in PROBABILITY units -- it does NOT itself scale. We assert:
      (1) identical emitted recommendation (TOP pick) -- the load-bearing claim;
      (4) identical per-sim pool-score argmax on a fixed full draft -- the exact invariance at the
          source (the P(1st) integrand);
      (2)/(3) the full ranking and per-team W coincide everywhere the W-gap clears the residual
          float-tie noise. That noise is NOT arbitrary: ``blend.points`` and ``0.1 * int_points``
          differ only in the last bits, flipping a MEASURED handful of pool-score ties; each flipped
          sim moves a team's W by at most ``(n-1)/n_sims``, so the per-team |dW| is bounded by
          ``(measured flips) * (n-1)/n_sims`` -- a data-derived tolerance, no magic number.
    """
    n_sims, seed = 1500, 20260609
    n_d, n_r = 8, 6
    blend = B.build_board(n_sims=n_sims, seed=seed, scheme=B.RECOMMENDED_SCHEME)
    stages, group, field = _independent_sim(seed, n_sims)

    # The published INTEGER scheme, built directly (not via ScoringScheme's normalised blend):
    # knockout terminal table 9*triangular, plus 3 points/win + 1 point/draw.
    int_points = (9.0 * get_ladder("triangular")[stages]
                  + 3.0 * group["wins"] + 1.0 * group["draws"])
    assert np.allclose(int_points, 10.0 * blend.points)  # the affine relationship, to float

    int_board = B.Board(
        points=int_points, stages=stages, team_ev=int_points.mean(axis=0),
        names=field.names, group_of=field.group_of,
        knockout_ladder="triangular_int9", w_pts=3.0, d_pts=1.0, mix=1.0,
        n_sims=n_sims, seed=seed, use_official_draw=True,
        config_sha256=field.config_sha256, board_sha256="int",
    )
    st = A.DraftState(n_drafters=n_d, n_rounds=n_r, our_seat=3)
    rec_blend = A.recommend(st, blend, r_samples=1, rng=np.random.default_rng(0))
    rec_int = A.recommend(st, int_board, r_samples=1, rng=np.random.default_rng(0))

    # (1) The emitted recommendation -- the TOP pick -- is identical (load-bearing).
    assert rec_blend.rows[0].team == rec_int.rows[0].team

    # (4) Source-level invariance on a deterministic full draft: the per-drafter pool scores are
    #     exactly 10x apart, so the winning SCORE is exactly 10x and the per-sim WINNER (the P(1st)
    #     integrand) agrees on every sim -- EXCEPT the few with a genuine integer-tie at the top,
    #     where the winner is arbitrary (the pool resolves it by the exogenous tiebreaker anyway)
    #     and the blend's last-bit float fuzz may pick the other tied drafter. This is the exact,
    #     no-tolerance statement of affine invariance under floating point.
    rosters = A._rosters_from_picks(list(range(n_d * n_r)), n_d, n_r)
    s_blend = M.pool_scores(blend.points, rosters)
    s_int = M.pool_scores(int_points, rosters)  # exact integers (no rounding)
    assert np.allclose(s_int.max(axis=1), 10.0 * s_blend.max(axis=1))  # winning score, 10x exact
    int_tie_at_top = (s_int == s_int.max(axis=1, keepdims=True)).sum(axis=1) > 1
    no_tie = ~int_tie_at_top
    assert np.array_equal(s_blend.argmax(axis=1)[no_tie], s_int.argmax(axis=1)[no_tie])

    # Data-derived W tolerance from the MEASURED count of those top-tie sims (the only sims whose
    # winner can differ between the two boards): each moves a team's 1st-place credit by at most 1,
    # so a team's W moves by at most ``n_tie * (n-1)/n_sims`` -- a measured bound, no magic number.
    n_tie = int(int_tie_at_top.sum())
    w_tol = max(n_tie, 1) * (n_d - 1) / n_sims  # >= one W-quantum even if no ties

    wb = {r.team: r.w0 for r in rec_blend.rows}
    wi = {r.team: r.w0 for r in rec_int.rows}
    assert max(abs(wb[t] - wi[t]) for t in wb) <= w_tol  # (2) per-team W agreement within the bound

    # (3) The ranking coincides through every rank whose W-gap exceeds that tie-flip tolerance;
    #     only W-ties below it may swap -- a harmless tie-break, not a different recommendation.
    wb_sorted = np.array([r.w0 for r in rec_blend.rows])
    sep = np.where(np.abs(np.diff(wb_sorted)) > w_tol)[0]
    k = (int(sep[-1]) + 1) if sep.size else len(wb_sorted)
    assert [r.team for r in rec_blend.rows[:k]] == [r.team for r in rec_int.rows[:k]]
