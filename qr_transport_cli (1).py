#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path

from qr_transport import audit, read_csv_radius_carrier, sha256_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Universal Q-R / Q^alpha transport CLI")
    ap.add_argument("--csv", required=True, help="CSV with columns dataset,R,qbase")
    ap.add_argument("--dataset", default=None, help="Optional dataset key")
    ap.add_argument("--output-json", default="qr_transport_certificate.json")
    args = ap.parse_args()

    groups = read_csv_radius_carrier(args.csv, dataset=args.dataset)
    if not groups:
        raise SystemExit("No valid datasets found")
    results = {name: audit(g["R"], g["q"]) for name, g in groups.items()}
    certificate = {
        "engine": "universal_qr_transport_cli_v1",
        "input_csv": args.csv,
        "datasets": list(results.keys()),
        "results": results,
    }
    certificate["certificate_hash"] = sha256_json(certificate)
    Path(args.output_json).write_text(json.dumps(certificate, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_json": args.output_json,
        "certificate_hash": certificate["certificate_hash"],
        "datasets": list(results.keys()),
        "all_100": all(
            r["canonical"]["metric"]["spectral_match_percent"] == 100.0 and
            r["permutation_jump"]["metric"]["spectral_match_percent"] == 100.0 and
            r["phase_scrambled_signal"]["metric"]["spectral_match_percent"] == 100.0
            for r in results.values()
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
