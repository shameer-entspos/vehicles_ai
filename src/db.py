"""SQLite storage. One row = one vehicle with its number plate."""
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS plate_records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    plate         TEXT,                 -- NULL when unreadable
    plate_conf    REAL,
    status        TEXT NOT NULL,        -- confirmed / low_conf / unreadable
    vehicle_type  TEXT,
    camera_id     TEXT,
    direction     TEXT,                 -- entry / exit / NULL
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    reads         INTEGER DEFAULT 0,    -- number of agreeing OCR reads
    track_id      INTEGER,
    video_sec     REAL,
    vehicle_image TEXT,
    plate_image   TEXT
);
CREATE INDEX IF NOT EXISTS idx_plate ON plate_records(plate);
CREATE INDEX IF NOT EXISTS idx_first ON plate_records(first_seen);
"""

COLS = ["plate", "plate_conf", "status", "vehicle_type", "camera_id", "direction",
        "first_seen", "last_seen", "reads", "track_id", "video_sec", "vehicle_image", "plate_image"]


class PlateDB:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def insert(self, **f):
        vals = [f.get(c) for c in COLS]
        cur = self.conn.execute(
            f"INSERT INTO plate_records ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})", vals)
        self.conn.commit()
        return cur.lastrowid

    def touch(self, rid, last_seen, add_reads=0, plate=None, conf=None, status=None, plate_image=None):
        """Same vehicle seen again: refresh last_seen; optionally upgrade the plate."""
        sets, vals = ["last_seen=?", "reads=reads+?"], [last_seen, add_reads]
        for col, v in (("plate", plate), ("plate_conf", conf), ("status", status), ("plate_image", plate_image)):
            if v is not None:
                sets.append(f"{col}=?")
                vals.append(v)
        self.conn.execute(f"UPDATE plate_records SET {','.join(sets)} WHERE id=?", vals + [rid])
        self.conn.commit()

    def update_direction(self, rid, direction):
        self.conn.execute("UPDATE plate_records SET direction=? WHERE id=?", (direction, rid))
        self.conn.commit()

    def all(self):
        cur = self.conn.execute("SELECT * FROM plate_records ORDER BY id")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def search(self, plate_like):
        cur = self.conn.execute("SELECT * FROM plate_records WHERE plate LIKE ? ORDER BY id",
                                (f"%{plate_like.upper()}%",))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def close(self):
        self.conn.close()
