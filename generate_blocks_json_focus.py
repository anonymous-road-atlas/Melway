import json
from pathlib import Path

# Adjust this if needed.
BASE_DIR = Path("melway_outputs")
BASE_TYPE_FILE = "historical_db.geojson"

FOCUS_CENTERS = [4, 43, 168, 220,121]
FOCUS_RADIUS = 5


def build_focus_block_names():
    block_numbers = set()
    for center in FOCUS_CENTERS:
        for n in range(center - FOCUS_RADIUS, center + FOCUS_RADIUS + 1):
            if n >= 0:
                block_numbers.add(n)

    return [f"m{n:03d}" for n in sorted(block_numbers)]


def generate_blocks_for_year(year_path: Path, focus_blocks):
    blocks = []
    for block_name in focus_blocks:
        block_dir = year_path / block_name
        geojson_path = block_dir / BASE_TYPE_FILE

        if block_dir.is_dir() and geojson_path.exists():
            blocks.append(block_name)

    return blocks


def main():
    if not BASE_DIR.exists():
        print(f"Directory not found: {BASE_DIR}")
        return

    focus_blocks = build_focus_block_names()

    print(f"Scanning: {BASE_DIR.resolve()}")
    print(f"Focus blocks: {', '.join(focus_blocks)}")
    print("-" * 50)

    for year_dir in BASE_DIR.iterdir():
        if not year_dir.is_dir():
            continue

        year = year_dir.name
        if not year.isdigit():
            print(f"Skipping non-year folder: {year}")
            continue

        blocks = generate_blocks_for_year(year_dir, focus_blocks)
        if not blocks:
            print(f"No valid blocks found for {year}")
            continue

        output_file = year_dir / "blocks.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(blocks, f, indent=2)

        print(f"{year}: {len(blocks)} blocks written to {output_file.name}")

    print("-" * 50)
    print("Done.")


if __name__ == "__main__":
    main()
