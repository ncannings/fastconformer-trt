#!/bin/bash
# GPU temperature watchdog for the unthrottled stock-vs-final runs: logs every 2 s; at >= 92 C stops this session's
# speech containers (image ${ASR_IMAGE:-fastconformer-trt:25.11}, by container ID) and writes ~/asr_data/WATCHDOG_TRIPPED.
LOG=${ASR_DATA_DIR:-$HOME/asr_data}/temp_watchdog.log; LIMIT=${1:-92}
while true; do
  read -r T C P <<< "$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits | tr -d ',')"
  echo "$(date +%T) ${T}C ${C}MHz ${P}W" >> $LOG
  if [ "${T%.*}" -ge "$LIMIT" ]; then
    for id in $(docker ps -q --filter ancestor=${ASR_IMAGE:-fastconformer-trt:25.11}); do docker kill $id > /dev/null; done
    echo "$(date +%T) TRIPPED at ${T}C" | tee -a $LOG > ${ASR_DATA_DIR:-$HOME/asr_data}/WATCHDOG_TRIPPED
  fi
  sleep 2
done
