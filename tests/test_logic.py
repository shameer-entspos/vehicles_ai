"""No GPU/models needed:  python tests/test_logic.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.counter import LineCounter
from src.db import PlateDB
from src.platefmt import fix_by_position, normalize, similar
from src.recorder import Recorder, TrackState

PAT = [r"[A-Z]{2,4}\d{2,6}"]


class FakeRead:
    def __init__(self, text, conf, valid=True):
        self.text, self.conf, self.valid, self.crop = text, conf, valid, None


def test_counter():
    c = LineCounter((0, 100), (200, 100), "positive")
    assert c.check((50, 90), (50, 110)) == "entry"
    assert c.check((50, 110), (50, 90)) == "exit"
    assert c.check((50, 80), (50, 90)) is None
    assert c.check((500, 90), (500, 110)) is None
    print("counter OK")


def test_platefmt():
    assert normalize("lea-12 34") == "LEA1234"
    assert fix_by_position("LEA1Z34", PAT) == ("LEA1234", True)      # Z in digit area -> 2? no: Z->2 gives LEA1234
    assert fix_by_position("1EA1234", PAT) == ("IEA1234", True)      # digit in letter area -> letter
    assert fix_by_position("LE0123", PAT)[0] in ("LEO123", "LE0123")
    assert similar("LEA1234", "LEA1284") and similar("LEA1234", "LEA123")
    assert not similar("LEA1234", "ABC9999")
    print("platefmt OK")


def test_voting_and_db():
    d = tempfile.mkdtemp()
    cfg = dict(camera_id="t", snapshot_dir=d, cooldown_sec=60, log_unreadable=True,
               min_reads_accept=2, high_conf_accept=0.8, min_track_frames=3)
    db = PlateDB(os.path.join(d, "t.db"))
    rec = Recorder(db, cfg)

    tr = TrackState(1, "car", 1, 0.0); tr.last_frame = 20; tr.eligible = True
    for t, c in [("LEA1234", 0.9), ("LEA1284", 0.5), ("LEA1234", 0.8), ("XYZ999", 0.3)]:
        tr.add_read(FakeRead(t, c))
    assert tr.consensus()[0] == "LEA1234" and tr.consensus()[1] == 3
    assert tr.is_confirmed(3, 0.6)
    rid = rec.commit(tr)

    tr2 = TrackState(2, "car", 30, 1.0); tr2.last_frame = 50; tr2.eligible = True   # same car, new track id
    tr2.add_read(FakeRead("LEA1234", 0.7)); tr2.add_read(FakeRead("LEA1234", 0.7))
    assert rec.commit(tr2) == rid                                    # de-duplicated

    tr3 = TrackState(3, "truck", 60, 2.0); tr3.last_frame = 90; tr3.eligible = True   # never read
    rec.commit(tr3)

    tr4 = TrackState(4, "car", 60, 2.0); tr4.last_frame = 90                           # tiny/far vehicle
    assert rec.commit(tr4) is None

    rows = db.all()
    assert len(rows) == 2, rows
    assert rows[0]["plate"] == "LEA1234" and rows[0]["status"] == "confirmed"
    assert rows[1]["plate"] is None and rows[1]["status"] == "unreadable"
    db.update_direction(rid, "entry")
    assert db.search("lea")[0]["direction"] == "entry"
    db.close()
    print("voting + db OK")


if __name__ == "__main__":
    test_counter(); test_platefmt(); test_voting_and_db()
