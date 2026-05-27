import os
import json
from pathlib import Path

# ===== 你可以改這個路徑 =====
BASE_DIR = Path("melway_outputs")  # 如果在同層就不用改

BASE_TYPE_FILE = "historical_db.geojson"

def generate_blocks_for_year(year_path: Path):
    blocks = []

    for item in year_path.iterdir():
        if item.is_dir():
            geojson_path = item / BASE_TYPE_FILE

            # 只收錄真的有 historical_db.geojson 的資料夾
            if geojson_path.exists():
                blocks.append(item.name)

    blocks.sort()
    return blocks


def main():
    if not BASE_DIR.exists():
        print(f"❌ Directory not found: {BASE_DIR}")
        return

    print(f"Scanning: {BASE_DIR.resolve()}")
    print("-" * 50)

    for year_dir in BASE_DIR.iterdir():
        if not year_dir.is_dir():
            continue

        year = year_dir.name

        # 如果資料夾名稱不是數字年份，可以選擇跳過
        if not year.isdigit():
            print(f"Skipping non-year folder: {year}")
            continue

        blocks = generate_blocks_for_year(year_dir)

        if not blocks:
            print(f"⚠ No valid blocks found for {year}")
            continue

        output_file = year_dir / "blocks.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(blocks, f, indent=2)

        print(f"✔ {year}: {len(blocks)} blocks written to {output_file.name}")

    print("-" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
