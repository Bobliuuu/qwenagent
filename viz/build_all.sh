#!/bin/bash
# Regenerates every figure, GIF and table from results/ and traces/. Needs only numpy, matplotlib, pillow.
# Headline = newest experiment with >=5 modes and fixed register names; the q7 (random names) section appears when results/q7_*.json exist.
set -e
cd "$(dirname "$0")/.."
python3 viz/make_figures.py
python3 viz/make_gifs.py            # headline traces: multi, oracle, and side-by-side; prints a note if none exist
python3 viz/make_tables.py          # after figures/GIFs: injects README blocks (tables, probes, facts, timeline, q7, hero)
python3 viz/make_fixture.py && python3 viz/make_gifs.py --trace viz/fixtures/synthetic_trace.json --out viz/fixtures/out   # synthetic self-test, never used in README
echo "done"
