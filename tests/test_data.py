import struct

import pytest

from scope_eval.data import (
    discover_label_directories,
    inspect_mp4_atoms,
    parse_label_file,
    parse_label_identity,
    parse_video_identity,
)


def test_six_column_label_keeps_object_id(tmp_path):
    label = tmp_path / "00000012.txt"
    label.write_text("0,7,0.5,0.5,0.2,0.1\n", encoding="utf-8")
    assert parse_label_file(label, {0: "Harmonic0"}) == [{
        "class_id": 0, "class_name": "Harmonic0", "object_id": 7,
        "cx": 0.5, "cy": 0.5, "box_width": 0.2, "box_height": 0.1,
    }]


def test_invalid_label_is_rejected(tmp_path):
    label = tmp_path / "bad.txt"
    label.write_text("0,0,1.5,0.5,0.2,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid normalized box"):
        parse_label_file(label, {0: "Harmonic0"})


def test_border_crossing_label_is_accepted_for_later_clipping(tmp_path):
    label = tmp_path / "00001191.txt"
    label.write_text("0,0,0.642448,0.0462963,0.0661458,0.17963\n", encoding="utf-8")
    parsed = parse_label_file(label, {0: "Harmonic0"})
    assert len(parsed) == 1
    assert parsed[0]["cy"] - parsed[0]["box_height"] / 2 < 0


def test_video_and_label_identity():
    from pathlib import Path
    assert parse_video_identity(Path("VID001_B_5fps.mp4")) == ("VID001", "VID001_B")
    assert parse_label_identity(Path("VID001_Bseg")) == ("VID001", "VID001_B")
    override = ("CH2001", "CH2001_CH001")
    assert parse_video_identity(Path("Ch2_001_CH001_V_5fps.mp4"), override) == override
    assert parse_label_identity(Path("CH2001CH001"), override) == override


def test_configured_label_directory_is_discovered(tmp_path):
    label_root = tmp_path / "label"
    label_root.mkdir()
    (label_root / "VID001_Bseg").mkdir()
    (label_root / "CH2001CH001").mkdir()
    config = {
        "label_dir": label_root,
        "identity_overrides": {"labels": {"CH2001CH001": ["CH2001", "CH2001_CH001"]}},
    }
    assert [path.name for path in discover_label_directories(config)] == ["CH2001CH001", "VID001_Bseg"]


def test_truncated_mdat_is_detected(tmp_path):
    video = tmp_path / "broken.mp4"
    video.write_bytes(struct.pack(">I4s", 100, b"mdat") + b"short")
    atom, declared_end = inspect_mp4_atoms(video)
    assert atom == "mdat"
    assert declared_end == 100
