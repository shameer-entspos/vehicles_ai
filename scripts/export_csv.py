"""Show records and export to CSV.
  python scripts/export_csv.py                 # all records
  python scripts/export_csv.py --plate LEA     # search by (part of) plate
  python scripts/export_csv.py --db data/plates.db --out data/plates.csv
"""
import argparse
import csv
import sqlite3

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/plates.db")
ap.add_argument("--out", default="data/plates.csv")
ap.add_argument("--plate", help="filter: plate contains this text")
a = ap.parse_args()

conn = sqlite3.connect(a.db)
q, params = "SELECT * FROM plate_records", ()
if a.plate:
    q, params = q + " WHERE plate LIKE ?", (f"%{a.plate.upper()}%",)
cur = conn.execute(q + " ORDER BY id", params)
cols = [d[0] for d in cur.description]
rows = [dict(zip(cols, r)) for r in cur.fetchall()]

with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print(f"{len(rows)} records -> {a.out}\n")
print(f"{'id':>4}  {'first_seen':<19}  {'plate':<11} {'conf':>5}  {'status':<10} {'type':<10} {'dir':<6} reads")
for r in rows[-25:]:
    conf = f"{r['plate_conf']:.2f}" if r["plate_conf"] else "  -"
    print(f"{r['id']:>4}  {r['first_seen']:<19}  {r['plate'] or '-':<11} {conf:>5}  {r['status']:<10} "
          f"{r['vehicle_type'] or '':<10} {r['direction'] or '-':<6} {r['reads']}")
