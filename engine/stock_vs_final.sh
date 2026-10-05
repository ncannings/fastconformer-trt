#!/bin/bash
# Controlled stock-versus-final comparison: run with the GPU clocks at driver default (nvidia-smi -rgc) and the machine
# otherwise idle. Heat-soak first so both arms run at operating temperature, then each arm under the
# wall-plug meter (optional, POWER_LOG_CMD) with GPU clock/temperature/power logged, a short re-soak between arms. One model per call.
#   [FINAL_FLAGS="--frame-budget 4800 --max-batch 128"] stock_vs_final.sh MODEL ENGINE OUT_DIR [SOAK_S]
#   (OUT_DIR under $ASR_DATA_DIR; ENGINE as /data/... path; MODEL a hub name or /data/<name>.nemo)
set -uo pipefail
MODEL=$1; ENGINE=$2; OUT=$3; SOAK=${4:-300}
H=$(cd "$(dirname "$0")" && pwd); D=${ASR_DATA_DIR:-$HOME/asr_data}/$OUT; mkdir -p $D 2>/dev/null || sudo -n mkdir -p $D
# optional wall-plug power logger: POWER_LOG_CMD="<cmd>" must accept --duration S --out FILE.csv and write t_unix,watts rows
PL=${POWER_LOG_CMD:-}
gpu_log() { nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader -l 2 > $D/gpu_$1.csv & echo $!; }
soak() {   # sustained FP8 GEMM load for $1 seconds (gemm_hold.py runs 25 s per call)
  local end=$((SECONDS + $1)); while [ $SECONDS -lt $end ]; do $H/run_in_container.sh gemm_hold.py > /dev/null 2>&1; done
}
echo "clock policy: $(nvidia-smi -q -d CLOCK | grep -A2 'Applications Clocks' | tr -s ' ' | tr '\n' ' ')"
[ -n "$PL" ] && $PL --duration 60 --out $D/idle.csv > /dev/null 2>&1
G=$(gpu_log soak1); soak $SOAK; kill $G
for arm in stock final; do
  G=$(gpu_log $arm); P=""; if [ -n "$PL" ]; then $PL --out $D/power_$arm.csv --duration 7200 > /dev/null 2>&1 & P=$!; fi
  if [ $arm = stock ]; then
    ASR_MODEL=$MODEL $H/run_in_container.sh stock_rtf.py /data/$OUT/$arm.json --wer-sets test_clean --wer-n 3000 \
      --rtf-set test_clean --reps 5 2>&1 | grep -E '^\{"|Traceback' | cut -c1-300
  else
    ASR_MODEL=$MODEL FAST_JOINT_FP8=1 FFN_PLUGIN_LIB=/w/plugins/libffn_fp8.so $H/run_in_container.sh trt_rtf.py $ENGINE \
      /data/$OUT/$arm.json --batch 32 ${FINAL_FLAGS:-} --dec-batch 256 --wer-sets test_clean --wer-n 3000 --rtf-set test_clean --reps 5 \
      --dec-bf16 --pipeline --pre-compile --fast-joint --fast-hyps 2>&1 | grep -E '^\{"|Traceback' | cut -c1-300
  fi
  kill $P $G 2>/dev/null
  [ $arm = stock ] && { G=$(gpu_log soak2); soak 120; kill $G; }
done
python3 - $D <<'PY'
import csv, json, statistics, sys
D = sys.argv[1]
import os
power = os.path.exists(f"{D}/idle.csv")
idle = statistics.mean(float(r["watts"]) for r in csv.DictReader(open(f"{D}/idle.csv")) if r["watts"]) if power else None
out = {"idle_w": round(idle, 1) if power else None}
for arm in ("stock", "final"):
    r = json.load(open(f"{D}/{arm}.json")); t0, t1 = r["timed_window"]
    h = r["audio_s"] / 3600 * len(r["walls_s"])
    if power:
        pw = [(float(x["t_unix"]), float(x["watts"])) for x in csv.DictReader(open(f"{D}/power_{arm}.csv")) if x["watts"]]
        win = [w for t, w in pw if t0 <= t <= t1]; mw = statistics.mean(win)
    g = [l.split(", ") for l in open(f"{D}/gpu_{arm}.csv")]
    mhz = [float(x[1].split()[0]) for x in g if len(x) > 4 and float(x[4].split()[0]) > 50]
    temp = [float(x[2]) for x in g if len(x) > 4]
    out[arm] = {"wer_test_clean": round(r["test_clean"], 3), "rtf": round(r["rtf"]),
                **({"mean_w": round(mw, 1), "max_w": max(win), "J_per_audio_h_gross": round(mw * (t1 - t0) / h, 1),
                    "J_per_audio_h_net": round((mw - idle) * (t1 - t0) / h, 1)} if power else {}),
                "sm_mhz_loaded_mean": round(statistics.mean(mhz)) if mhz else None, "temp_max_c": max(temp) if temp else None}
out["speedup"] = round(out["final"]["rtf"] / out["stock"]["rtf"], 2)
json.dump(out, open(f"{D}/summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
PY
