import numpy as np

from draft_advisor import advisor as A
from draft_advisor import cli


def test_match_team_variants():
    names = ["Spain", "Senegal", "Sweden", "Brazil"]
    assert cli.match_team("brazil", names, set()) == 3
    assert cli.match_team("sp", names, set()) == 0  # unique prefix
    ambiguous = cli.match_team("s", names, set())
    assert isinstance(ambiguous, list) and set(ambiguous) == {0, 1, 2}
    assert cli.match_team("xyz", names, set()) is None
    assert cli.match_team("spain", names, {0}) is None  # already taken excluded


def test_scripted_full_run(monkeypatch, capsys, make_board):
    pts = np.abs(np.random.default_rng(0).normal(5, 2, size=(80, 6)))
    names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    bd = make_board(pts, names=names)
    state = A.DraftState(2, 2, our_seat=0)
    inputs = iter(["Alpha", "Bravo", "Charlie", "Delta"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    cli.run(state, bd, np.random.default_rng(0))
    out = capsys.readouterr().out
    assert "YOUR PICK" in out  # recommendation shown on our turn
    assert "YOU took Alpha" in out  # our pick attributed to us
    assert "seat 2 took Bravo" in out  # opponent pick attributed to their seat
    assert "POST-DRAFT SUMMARY" in out


def test_manual_mode_suppresses_auto_and_me_triggers(monkeypatch, capsys, make_board):
    pts = np.abs(np.random.default_rng(0).normal(5, 2, size=(80, 6)))
    names = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    bd = make_board(pts, names=names)
    state = A.DraftState(2, 2, our_seat=0)
    inputs = iter(["me", "Alpha", "Bravo", "Charlie", "Delta"])  # 'me' manually triggers the rec
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(inputs))
    cli.run(state, bd, np.random.default_rng(0), auto=False)
    out = capsys.readouterr().out
    assert "[confirm]" in out  # turn-state confirmation from the manual trigger
    assert "snake pointer:" in out
    assert "YOUR PICK" in out  # rec shown only because 'me' was typed
    assert "POST-DRAFT SUMMARY" in out


def test_cancellable_catches_ctrl_c(capsys):
    def boom():
        raise KeyboardInterrupt

    assert cli._cancellable(boom) is False
    assert "cancelled" in capsys.readouterr().out


def test_cancellable_runs_action_normally():
    calls = []
    assert cli._cancellable(calls.append, 1) is True
    assert calls == [1]


def test_ctrl_c_at_prompt_does_not_exit_session(monkeypatch, capsys, make_board):
    bd = make_board(np.ones((20, 6)))
    state = A.DraftState(2, 2, our_seat=0)
    seq = iter(["ctrl_c", "quit"])

    def fake_input(*a, **k):
        if next(seq) == "ctrl_c":
            raise KeyboardInterrupt
        return "quit"

    monkeypatch.setattr("builtins.input", fake_input)
    cli.run(state, bd, np.random.default_rng(0), auto=False)  # returns cleanly, no traceback
    assert "type 'quit' to exit" in capsys.readouterr().out
