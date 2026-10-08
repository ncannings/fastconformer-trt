#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nigel Cannings. Part of fastconformer-trt (see LICENSE and NOTICE).
# Acts for the recorded demo, one process per act (a crash in one act, e.g. NeMo's decoder graph re-capture on the
# H100, ends only that act); the terminal shows only the dashboards. usage: video_acts.sh OUTDIR ACT [ACT ...]
# ACT = arm:r:n[:env assignments separated by +]
OUT=$1; shift
export PYTHONWARNINGS=ignore::FutureWarning
k=0
for act in "$@"; do
  k=$((k + 1))
  IFS=: read -r arm r n envs <<< "$act"
  EV=${envs//+/ }
  if [ $r = 0 ]; then LABEL=${OURS_LABEL_R0:-OURS: FP16 + mel graph + fused decoder}; else LABEL=${OURS_LABEL_R13:-OURS: FP16 + fused decoder}; fi
  env $EV python /w/demo_live.py --acts $arm:$r:$n --act-s ${ACT_S:-45} --pause-s 3 --log $OUT/demo_act${k}_${arm}_r${r}_n${n}.log \
    --title-prefix "${TITLE_PREFIX:-}" --engines "${DEMO_ENGINES:-}" --ours-label "$LABEL" \
    --mel-graph-rs "${MG_RS:-0}"
  rc=$?
  if [ $rc != 0 ]; then
    printf '\n\033[1;31m%s\033[0m\n' "act $k ($arm, $n streams) ended with an error: $(grep -m1 -oE '[A-Za-z]*Error[^\n]{0,120}' $OUT/demo_act${k}_${arm}_r${r}_n${n}.log | head -1)"
    sleep 4
  fi
done
