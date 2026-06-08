"""Live draft REPL -- the [A]-[J] interface.

Usage (run from the repo root)::

    uv run python -m draft_advisor.cli --players 8 --rounds 4 --ladder triangular --seat 5

Then, as the draft proceeds, type each team AS IT IS PICKED (in pick order). Ownership is
auto-attributed from the snake schedule, so you tag nothing. Commands:

    <team>     log the next pick (fuzzy-matched; e.g. "spain", "arg", "neth")
    rec        force-show the recommendation for the current pick
    undo       remove the last logged pick (re-renders the current turn)
    state      show the board / rosters so far
    help       list commands
    quit       exit

When it becomes your turn the recommendation prints automatically; after you log your pick,
your projected standing prints; when the draft completes, the post-draft summary prints.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import advisor as A
from . import board as B
from . import objective as O

# ASCII ceiling bands (a Windows console is cp1252; avoid non-encodable glyphs).
_DOTS = {2: "HHH", 1: "HH.", 0: "H.."}
_MAX_ROWS = 12  # recommendation rows shown (display cap)
_MAX_AMBIGUOUS = 8  # candidate names listed on an ambiguous match (display cap)


def match_team(token: str, names: list[str], taken: set[int]) -> int | None | list[int]:
    """Resolve a user token to a global team index.

    Returns the index on a unique match, ``None`` if nothing matches, or a list of candidate
    indices if ambiguous. Already-taken teams are excluded.
    """
    t = token.strip().lower()
    avail = [(i, n.lower()) for i, n in enumerate(names) if i not in taken]
    exact = [i for i, n in avail if n == t]
    if exact:
        return exact[0]
    prefix = [i for i, n in avail if n.startswith(t)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        return prefix
    sub = [i for i, n in avail if t in n]
    if len(sub) == 1:
        return sub[0]
    if len(sub) > 1:
        return sub
    return None


def _fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def print_banner(state: A.DraftState, board: B.Board) -> None:
    w = int(round(O.objective_weight(state.n_drafters)))
    draw = "official" if board.use_official_draw else "resampled"
    print(
        f"[A] draft_advisor | players {state.n_drafters} | rounds {state.n_rounds} | "
        f"ladder {board.ladder} | seat {state.our_seat + 1}/{state.n_drafters}"
    )
    print(
        f"    objective W = {w}*P1 + P2 | field {draw} draw, board sha {board.board_sha256[:8]} | "
        f"n_sims {board.n_sims}"
    )
    print(f"[B] your picks (snake {state.our_seat + 1}/{state.n_drafters}): "
          f"{', '.join(map(str, state.our_pick_numbers()))}")


def print_recommendation(rec: A.Recommendation, state: A.DraftState) -> None:
    overall = len(state.picks) + 1
    rnd = (overall - 1) // state.n_drafters + 1
    print(f"  -- [D] YOUR PICK . overall {overall} . round {rnd} " + "-" * 24)
    print(f"   {'#':>2}  {'team':<16}{'W':>8}  {'P1':>6}  {'top2':>6}  ceiling  bracket")
    shown = rec.rows[:_MAX_ROWS]
    bands = O.ceiling_bands(np.array([r.ceiling_value for r in shown]))  # band the shown set
    cliffset = set(rec.cliffs)
    for i, (r, band) in enumerate(zip(shown, bands, strict=True)):
        flag = "clear" if r.collision_partner is None else f"shares {r.collision_partner}"
        print(
            f"   {i + 1:>2}  {r.name:<16}{r.w0:8.3f}  {_fmt_pct(r.p1)}  {_fmt_pct(r.p_money)}  "
            f"  {_DOTS[int(band)]}   {flag}"
        )
        if i in cliffset and i + 1 < len(rec.rows):
            gap = rec.rows[i].w0 - rec.rows[i + 1].w0
            print(f"   {'':>2}  -- tier cliff ({gap:+.3f}) " + "-" * 12)
    top = rec.rows[0]
    if top.robust_top:
        rob = "rank-1 across opponent sweep"
    elif top.rank_by_temp:
        rob = f"rank varies {min(top.rank_by_temp)}-{max(top.rank_by_temp)} across opponent sweep"
    else:
        rob = "robustness not assessed"
    print(f"  [F] -> take {top.name.upper()}   [G] {rob}")
    rw = rec.reach_wait
    if rw and rw.get("gap"):
        print(f"  [H] ~{rw['gap']} picks before your next -> {rw['verdict']}")
        if rw["survival"]:
            alt_name, alt_p = next(iter(rw["survival"].items()))
            print(f"      P({alt_name} survives to your next pick) = {_fmt_pct(alt_p)}")


def print_standing(st: dict) -> None:
    print(f"[I] your standing (projected): P1 {_fmt_pct(st['p1'])} . "
          f"top2 {_fmt_pct(st['p_money'])} . mode: {st['mode']}")


def print_summary(s: dict) -> None:
    print("=" * 60)
    print("[J] POST-DRAFT SUMMARY")
    pairs = zip(s["roster"], s["roster_groups"], strict=True)
    print(f"   roster: {', '.join(f'{n} ({g})' for n, g in pairs)}")
    print(f"   P1 {_fmt_pct(s['p1'])} | 2nd {_fmt_pct(s['p2'])} | "
          f"in-money {_fmt_pct(s['p_money'])}")
    print(f"   score  mean {s['score_mean']:.1f} | P5 {s['score_p05']:.0f} | "
          f"P50 {s['score_p50']:.0f} | P95 {s['score_p95']:.0f}")
    print(f"   swing team (watch): {s['swing_team']}  (corr {s['swing_corr']:.2f})")
    print("=" * 60)


def print_state(state: A.DraftState, board: B.Board) -> None:
    avail = int(state.available_mask(board.n_teams).sum())
    print(f"[C] board: {len(state.picks)} gone . {avail} left")
    for d in range(state.n_drafters):
        roster = [board.names[t] for t in state.roster_of(d)]
        tag = " (you)" if d == state.our_seat else ""
        print(f"    seat {d + 1}{tag}: {', '.join(roster) if roster else '-'}")


def _log_pick(state: A.DraftState, board: B.Board, token: str) -> bool:
    taken = set(state.picks)
    m = match_team(token, board.names, taken)
    if m is None:
        print(f"   ? no available team matches '{token}'")
        return False
    if isinstance(m, list):
        opts = ", ".join(board.names[i] for i in m[:_MAX_AMBIGUOUS])
        print(f"   ? ambiguous '{token}': {opts}")
        return False
    state.picks.append(m)
    return True


def _render_turn(state: A.DraftState, board: B.Board, rng: np.random.Generator) -> None:
    """Print the recommendation if it is our turn, else nothing."""
    if state.is_our_turn():
        print_recommendation(A.recommend(state, board, rng=rng), state)


def _pick_context(state: A.DraftState) -> tuple[int, int, int]:
    """(overall 1-indexed pick number, round number, 0-indexed seat) for the next pick."""
    overall = len(state.picks) + 1
    rnd = (overall - 1) // state.n_drafters + 1
    seat = int(state.sequence[len(state.picks)])
    return overall, rnd, seat


def _prompt(overall: int, rnd: int, seat: int, is_us: bool) -> str:
    """The input prompt, labelled with whose pick it is; a distinct marker on our turn."""
    if is_us:
        return f"#{overall} R{rnd} seat {seat + 1} (YOU) >>> "
    return f"#{overall} R{rnd} seat {seat + 1} > "


def _echo_pick(board: B.Board, seat: int, overall: int, rnd: int, team: int, is_us: bool) -> None:
    """Confirm the resolved pick and attribute it to a seat (special-cased for us)."""
    name = board.names[team]
    if is_us:
        print(f"   >> YOU took {name}  (#{overall} R{rnd}) <<")
    else:
        print(f"   seat {seat + 1} took {name}  (#{overall} R{rnd})")


def print_turn_check(state: A.DraftState, board: B.Board) -> None:
    """One-line state confirmation so you can catch a drifted ('muffled') snake order."""
    avail = int(state.available_mask(board.n_teams).sum())
    nslot = len(state.picks)
    pointer = "complete" if state.is_complete else f"seat {int(state.sequence[nslot]) + 1}"
    agree = "matches" if state.is_our_turn() else "DIFFERS -> manual override"
    print(f"   [confirm] you are seat {state.our_seat + 1} | you own "
          f"{len(state.our_roster())}/{state.n_rounds} | {avail} available | "
          f"snake pointer: {pointer} ({agree})")


def _show_recommendation(state: A.DraftState, board: B.Board, rng: np.random.Generator) -> None:
    """On-demand recommendation for our next pick, valid regardless of the snake counter."""
    if state.is_complete:
        print("   draft complete -- no pick to make")
        return
    if len(state.our_roster()) >= state.n_rounds:
        print("   you already hold all your picks")
        return
    print_turn_check(state, board)
    print_recommendation(A.recommend(state, board, rng=rng), state)


def _cancellable(action, *args) -> bool:
    """Run a possibly-slow output action; Ctrl-C cancels just that action, not the session."""
    try:
        action(*args)
        return True
    except KeyboardInterrupt:
        print("\n   (cancelled)")
        return False


def _show_standing(state: A.DraftState, board: B.Board) -> None:
    print_standing(A.standing(state, board))


def run(state: A.DraftState, board: B.Board, rng: np.random.Generator, auto: bool = True) -> None:
    print_banner(state, board)
    print_state(state, board)
    if auto:
        _cancellable(_render_turn, state, board, rng)
    while not state.is_complete:
        overall, rnd, seat = _pick_context(state)
        is_us = seat == state.our_seat
        try:
            token = input(_prompt(overall, rnd, seat, is_us)).strip()
        except EOFError:  # piped input exhausted / Ctrl-D
            print()
            return
        except KeyboardInterrupt:  # Ctrl-C at the prompt clears the line; only 'quit' exits
            print("\n   (type 'quit' to exit)")
            continue
        if not token:
            continue
        cmd = token.lower()
        if cmd in {"quit", "exit", "q"}:
            return
        if cmd == "help":
            print("   <team> | me/rec | undo | state | help | quit  (Ctrl-C cancels a rec)")
            continue
        if cmd == "state":
            print_state(state, board)
            continue
        if cmd in {"me", "mine", "rec"}:  # manual trigger -- works even if the order drifted
            _cancellable(_show_recommendation, state, board, rng)
            continue
        if cmd == "undo":
            if state.picks:
                removed = state.picks.pop()
                print(f"   undone: {board.names[removed]}")
                if auto:
                    _cancellable(_render_turn, state, board, rng)  # re-render the rolled-back turn
            else:
                print("   nothing to undo")
            continue
        if _log_pick(state, board, token):
            _echo_pick(board, seat, overall, rnd, state.picks[-1], is_us)
            if is_us:
                _cancellable(_show_standing, state, board)
            if auto:
                _cancellable(_render_turn, state, board, rng)
    print_summary(A.post_draft_summary(state, board))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live World Cup draft-pick advisor")
    parser.add_argument("--players", type=int, required=True, help="number of drafters (7 or 8)")
    parser.add_argument("--rounds", type=int, default=4, help="teams per drafter (default 4)")
    parser.add_argument("--ladder", default="triangular",
                        choices=["linear", "triangular", "geometric"])
    parser.add_argument("--seat", type=int, required=True, help="our snake seat, 1-indexed")
    parser.add_argument("--n-sims", type=int, default=B.DEFAULT_N_SIMS)
    parser.add_argument("--seed", type=int, default=B.DEFAULT_SEED)
    parser.add_argument("--rng-seed", type=int, default=0, help="seed for the live sampler")
    parser.add_argument("--manual", action="store_true",
                        help="don't auto-show the rec on your turn; type 'me' to trigger")
    args = parser.parse_args(argv)

    if not (1 <= args.seat <= args.players):
        parser.error(f"--seat must be in 1..{args.players}")
    print(f"building/loading board (ladder={args.ladder}, n_sims={args.n_sims})...",
          file=sys.stderr)
    board = B.load_or_build(ladder=args.ladder, n_sims=args.n_sims, seed=args.seed)
    state = A.DraftState(n_drafters=args.players, n_rounds=args.rounds, our_seat=args.seat - 1)
    run(state, board, np.random.default_rng(args.rng_seed), auto=not args.manual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
