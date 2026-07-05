# Portable Q-Power CSV Suite

This package is self-contained.

It reads only:

```text
qpower_raw_data.csv
qpower_sources.csv
```

It does **not** read:
- JSON certificates
- Planck archives
- local `/mnt/data` paths
- external websites
- raw binary data files

## Core rule

```text
R_i = s_* q_i ^ alpha_i(q)
alpha_i(q) = log(R_i/s_*) / log(q_i)
```

If the carrier changes, alpha transports with it.

## Included data

`qpower_raw_data.csv` has 458 numeric rows across six datasets:

1. faithful_current_transport
2. gwtc_predictive_events
3. planck_lollipop_gramdet
4. nanograv_arxiv2606
5. strict_final_test
6. deterministic_final

Each row includes source labels, references, URLs or included-source markers, and notes.

## Run

```bash
python portable_qpower_csv_suite.py \
  --input-csv qpower_raw_data.csv \
  --sources-csv qpower_sources.csv \
  --output-json portable_qpower_csv_certificate.json \
  --output-summary portable_qpower_csv_summary.json \
  --output-png portable_qpower_csv_visual.png
```

Expected result:

```text
all_100: true
```
