"""Download the public DFL/Sportec tracking dataset (Bassek et al. 2025, figshare 28196177, CC BY 4.0).

Pinned file ids and published MD5s; verified files are skipped. Stdlib only.
Target: $MTAH_EXTERNAL_DIR/dfl, default ~/.mtah2027/external/dfl. Usage: scripts/py scripts/fetch_external.py
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

URL = "https://ndownloader.figshare.com/files/{}"
# (figshare file id, md5, file name); events files are not needed and skipped.
FILES = [
    (51643472, "37b0f85588c0acfb995fe633c7ac2e0d", "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml"),
    (51643475, "bc5d5b21dbcf36aa30f4cd0428309f6b", "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml"),
    (51643478, "b9710781d400dd812f62b0a6e3d7bce8", "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml"),
    (51643490, "5e2d580d3d8ed97e32959de90d04992b", "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml"),
    (51643487, "65a4568e4f374af94d3c1015cbc7c4f5", "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml"),
    (51643484, "5979209af316a914ce57347d0633df6c", "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml"),
    (51643481, "7b021e577fca917a63111d55cc29ae5b", "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml"),
    (51643514, "e4273a2adf5dbd9d4ec42def2f01905c", "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml"),
    (51643517, "bacb2f609c1d1b0bfc906f6b9d4109f2", "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml"),
    (51643520, "3efff25a86a61768e7d253b4bba137a3", "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml"),
    (51643523, "4e1063e0a0ae499920cf68ac1aa3ab4e", "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml"),
    (51643526, "e56ff1336a334f8da51110fbe7cab764", "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml"),
    (51643529, "9105bad987191d3c3c362c1fce9a4fcd", "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml"),
    (51643532, "e060bb7d7a5a3bfbd15bbf9bc631024d", "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml"),
]


def dfl_dir() -> Path:
    return Path(os.environ.get("MTAH_EXTERNAL_DIR", Path.home() / ".mtah2027" / "external")) / "dfl"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(file_id: int, expected_md5: str, path: Path) -> str:
    if path.exists() and md5(path) == expected_md5:
        return "ok (cached)"
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(URL.format(file_id), timeout=60) as response, open(tmp, "wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)
    if md5(tmp) != expected_md5:
        tmp.unlink()
        raise RuntimeError(f"MD5 mismatch for figshare file {file_id}")
    tmp.replace(path)
    return "ok (downloaded)"


def main() -> int:
    out = dfl_dir()
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for file_id, expected_md5, name in FILES:
        try:
            status = fetch(file_id, expected_md5, out / name)
        except Exception as error:  # report every file, fail at the end
            status, failed = f"FAILED: {error}", failed + 1
        print(f"{name}: {status}", flush=True)
    print(f"{len(FILES) - failed}/{len(FILES)} files verified in {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
