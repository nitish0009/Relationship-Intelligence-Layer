#!/bin/zsh
# One command for every remaining IJCAI experiment (see PREREGISTRATION_IJCAI.md):
#
#     bash scripts/run_ijcai_overnight.sh
#
# Lanes (parallel, memory-safe on a 16-18 GB machine):
#   1. rolling-origin AMLworld HI-Small  -> then IEEE-CIS TVW v2 (needs memory freed by lane 1)
#   2. rolling-origin AMLworld LI-Small
#   3. simulator objective baselines (H4)
# then: pre-registered report (results/ijcai/REPORT.md) + paper tables/figures.
# Every experiment is resumable: re-running this script skips finished units.
# caffeinate keeps the Mac awake (sleep paused a run for an hour before).
set -u
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
L=results/logs/ijcai
mkdir -p "$L"
S="$L/status.txt"
stamp() { echo "$(date '+%Y-%m-%d %H:%M') $*" | tee -a "$S"; }

stamp "START (pid $$)"
caffeinate -i -w $$ &

(
  stamp "lane1 start rolling HI-Small"
  python3 -c "import sys; sys.path.insert(0,'.'); import src; from experiments import rolling_aml as R; R.run(prefix='HI-Small')" \
      > "$L/rolling_hi.log" 2>&1; stamp "lane1 end rolling HI-Small exit=$?"
  stamp "lane1 start IEEE-CIS TVW v2"
  python3 -c "import sys; sys.path.insert(0,'.'); import src; from experiments import tvw; tvw.run(datasets=('ieee',), out_name='tvw_ieee')" \
      > "$L/tvw_ieee.log" 2>&1; stamp "lane1 end IEEE-CIS exit=$?"
) &
P1=$!
(
  stamp "lane2 start rolling LI-Small"
  python3 -c "import sys; sys.path.insert(0,'.'); import src; from experiments import rolling_aml as R; R.run(prefix='LI-Small')" \
      > "$L/rolling_li.log" 2>&1; stamp "lane2 end rolling LI-Small exit=$?"
) &
P2=$!
(
  stamp "lane3 start simulator objective baselines"
  python3 run_experiment.py objectives_sim > "$L/objectives_sim.log" 2>&1; stamp "lane3 end objectives exit=$?"
) &
P3=$!

wait $P1 $P2 $P3
stamp "all lanes finished -> report"
python3 -c "import sys; sys.path.insert(0,'.'); import src; from experiments import ijcai_report; ijcai_report.main()" \
    > "$L/report.log" 2>&1; stamp "report exit=$? -> results/ijcai/REPORT.md"
python3 scripts/make_paper_tables.py > "$L/paper_tables.log" 2>&1
python3 scripts/make_paper_figures.py > "$L/paper_figures.log" 2>&1
stamp "DONE"
