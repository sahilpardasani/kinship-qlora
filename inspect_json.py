"""Compare saved Hindi kinship evaluations. Requires only Python 3."""
import argparse
import json
from pathlib import Path


def find_results():
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for folder in (
            base, base / "results", base / "kinship-v2/results",
            base / "kinship-mac/kinship-v2/results",
        ):
            if all((folder / f"{v}-expanded.json").is_file() for v in ("v1", "v2")):
                return folder
    raise SystemExit("Could not find both expanded results. Run from kinship-mac or pass --results-dir PATH.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("relationship", nargs="?", default="husband_sister_husband")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--list", action="store_true", help="List available relationships")
    args = parser.parse_args()
    folder = args.results_dir.expanduser() if args.results_dir else find_results()
    records = {}
    for version in ("v1", "v2"):
        path = folder / f"{version}-expanded.json"
        if not path.is_file():
            parser.exit(1, f"Missing file: {path.resolve()}\n")
        records[version] = json.loads(path.read_text(encoding="utf-8"))["results"]

    relationships = sorted({row["relationship"] for row in records["v1"]})
    if args.list:
        print("\n".join(relationships))
        return
    if args.relationship not in relationships:
        parser.error(f"Unknown relationship {args.relationship!r}. Use --list to see available names.")

    v2 = {row["prompt"]: row for row in records["v2"] if row["relationship"] == args.relationship}
    for before in records["v1"]:
        if before["relationship"] != args.relationship:
            continue
        after = v2.get(before["prompt"])
        print(f"\n{before['occasion'].upper()} | Expected term: {before['term']}")
        print(before["prompt"])
        for version, row in (("V1", before), ("V2", after)):
            if row is None:
                print(f"{version}: No matching saved prompt")
            else:
                status = "PASS" if row["correct"] else "MISS"
                print(f"{version} [{status}]: {row['answer']}")


if __name__ == "__main__":
    main()
