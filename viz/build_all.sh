#!/bin/bash
# Regenerates every figure, GIF and table from results/ and traces/. Needs only numpy, matplotlib, pillow.
set -e
cd "$(dirname "$0")/.."
python3 viz/make_figures.py
python3 viz/make_tables.py
python3 viz/make_gifs.py            # real traces only (traces/q3_*_s0.json); prints a note if none exist
python3 viz/make_fixture.py && python3 viz/make_gifs.py --trace viz/fixtures/synthetic_trace.json --out viz/fixtures/out   # synthetic self-test, never used in README
echo "done"
