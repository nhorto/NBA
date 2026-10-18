import gzip
import json
from datetime import date, datetime, timezone

from courtside import archive

DAY = date(2026, 10, 20)


def test_round_trip_final(data_dir):
    p = archive.save_raw("nba_cdn", "boxscore", DAY, "0022600001", {"a": [1, 2]})
    assert p == data_dir / "raw/nba_cdn/boxscore/2026-10-20/0022600001.json.gz"
    assert archive.load_raw(p) == {"a": [1, 2]}
    assert json.loads(gzip.decompress(p.read_bytes())) == {"a": [1, 2]}
    assert archive.exists_final("nba_cdn", "boxscore", DAY, "0022600001")
    assert not archive.exists_final("nba_cdn", "boxscore", DAY, "other")


def test_final_overwrites(data_dir):
    archive.save_raw("s", "k", DAY, "x", {"v": 1})
    p = archive.save_raw("s", "k", DAY, "x", {"v": 2})
    assert archive.load_raw(p) == {"v": 2}


def test_snapshot_naming_never_overwrites():
    t1 = datetime(2026, 10, 20, 14, 5, 9, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 20, 15, 5, 9, tzinfo=timezone.utc)
    p1 = archive.save_raw("espn", "odds", DAY, "all", [1], snapshot=True, now=t1)
    p2 = archive.save_raw("espn", "odds", DAY, "all", [2], snapshot=True, now=t2)
    assert p1.name == "all__20261020T140509Z.json.gz"
    assert p1 != p2 and archive.load_raw(p1) == [1] and archive.load_raw(p2) == [2]
    assert not archive.exists_final("espn", "odds", DAY, "all")


def test_snapshot_converts_to_utc():
    from zoneinfo import ZoneInfo
    t = datetime(2026, 10, 20, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    p = archive.raw_path("espn", "odds", DAY, "k", snapshot_at=t)
    assert p.name == "k__20261020T140000Z.json.gz"


def test_atomic_write_leaves_no_temp_and_survives_failure(monkeypatch):
    p = archive.save_raw("s", "k", DAY, "x", {"ok": True})
    assert [f.name for f in p.parent.iterdir()] == ["x.json.gz"]

    def boom(*a, **k):
        raise OSError("disk")
    monkeypatch.setattr(archive.os, "replace", boom)
    try:
        archive.save_raw("s", "k", DAY, "x", {"ok": False})
    except OSError:
        pass
    assert archive.load_raw(p) == {"ok": True}
    assert [f.name for f in p.parent.iterdir()] == ["x.json.gz"]


def test_bytes_variant():
    p = archive.save_raw_bytes("injury_report", "injury_report", DAY, "r", b"%PDF-1.4",
                               snapshot=True,
                               now=datetime(2026, 10, 20, 1, 2, 3, tzinfo=timezone.utc))
    assert p.name == "r__20261020T010203Z.pdf"
    assert archive.load_raw(p) == b"%PDF-1.4"
