import json
from pathlib import Path
from tqdm import tqdm

MAX_MB = 45  # 保守：避免踩到 50MB / 100MB
MAX_BYTES = MAX_MB * 1024 * 1024


def byte_len(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def split_featurecollection(in_path: Path, max_bytes: int = MAX_BYTES):
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError("Input must be a GeoJSON FeatureCollection")

    base = {k: v for k, v in data.items() if k != "features"}
    features = data.get("features", [])

    out_dir = in_path.parent
    stem = in_path.stem      # historical_db
    suffix = in_path.suffix  # .geojson

    # 空殼大小
    empty = dict(base)
    empty["features"] = []
    overhead = byte_len(empty)

    parts = []
    cur = []
    part_idx = 0
    cur_bytes = overhead

    def flush():
        nonlocal part_idx, cur, cur_bytes
        if not cur:
            return
        obj = dict(base)
        obj["features"] = cur
        name = f"{stem}.part{part_idx:03d}{suffix}"
        (out_dir / name).write_text(
            json.dumps(obj, ensure_ascii=False),
            encoding="utf-8"
        )
        parts.append(name)
        part_idx += 1
        cur = []
        cur_bytes = overhead

    # 🔹 tqdm 進度條
    for feat in tqdm(
        features,
        desc=f"Splitting {in_path.name}",
        unit="feature",
    ):
        # 試著加一個 feature
        trial = dict(base)
        trial["features"] = cur + [feat]
        trial_bytes = byte_len(trial)

        # 超過上限就先 flush
        if trial_bytes > max_bytes and cur:
            flush()
            # 新 part 從這個 feature 開始
            cur.append(feat)
            cur_bytes = byte_len(dict(base, features=cur))
        else:
            cur.append(feat)
            cur_bytes = trial_bytes

    flush()

    index_name = f"{stem}.index.json"
    (out_dir / index_name).write_text(
        json.dumps(parts, indent=2),
        encoding="utf-8"
    )

    print(f"✅ Wrote {len(parts)} parts + {index_name} in {out_dir}")


def main():
    # ==== 只改這一行成你要切的那個檔案 ====
    target = Path("melway_outputs/2020/buffer_compare/historical_db.geojson")

    if not target.exists():
        raise FileNotFoundError(f"Not found: {target}")

    split_featurecollection(target)


if __name__ == "__main__":
    main()
