#!/bin/bash
# Evaluation data into $ASR_DATA_DIR (default ~/asr_data), in the parquet layout the runners read.
#   LibriSpeech test-clean / dev-clean: openslr/librispeech_asr (clean/test, clean/validation)
#   Earnings-22: the Open ASR Leaderboard test set (hf-audio/open-asr-leaderboard, earnings22/test, 2,741 utterances),
#                scored in full as set earnings22_full. (Development runs used 2,000-utterance samples from the first
#                shard of distil-whisper/earnings22 as set earnings22; see docs/05-measurement-method.md.)
#   FLEURS (25 languages of parakeet-tdt-0.6b-v3): google/fleurs test split (scored) and dev split (FP8 calibration only)
# usage: fetch_data.sh [english] [fleurs] [fleurs_dev]      (default: all three)
set -e
D=${ASR_DATA_DIR:-$HOME/asr_data}; mkdir -p $D; H=$(cd "$(dirname "$0")" && pwd)
WHAT=${*:-english fleurs fleurs_dev}
LANGS="bg_bg cs_cz da_dk de_de el_gr en_us es_419 et_ee fi_fi fr_fr hr_hr hu_hu it_it lt_lt lv_lv mt_mt nl_nl pl_pl pt_br ro_ro ru_ru sk_sk sl_si sv_se uk_ua"
get() { [ -f "$2" ] || curl -sfL --retry 20 --retry-all-errors -C - -o "$2" "$1"; }
if [[ " $WHAT " == *" english "* ]]; then
  L=https://huggingface.co/datasets/openslr/librispeech_asr/resolve/main/clean
  get $L/test/0000.parquet $D/test_clean.parquet
  get $L/validation/0000.parquet $D/dev_clean.parquet
  mkdir -p $D/earnings22_full
  for i in 0 1 2 3 4; do
    get https://huggingface.co/datasets/hf-audio/open-asr-leaderboard/resolve/main/earnings22/test-0000$i-of-00005.parquet $D/earnings22_full/test-0000$i-of-00005.parquet
  done
fi
for split in test dev; do
  [[ " $WHAT " == *" fleurs$([ $split = dev ] && echo _dev) "* ]] || continue
  F=$D/fleurs$([ $split = dev ] && echo _dev); mkdir -p $F
  for c in $LANGS; do
    [ -f $F/$c/done ] && continue
    mkdir -p $F/$c; U=https://huggingface.co/datasets/google/fleurs/resolve/main/data/$c
    get $U/$split.tsv $F/$c/$split.tsv
    get $U/audio/$split.tar.gz $F/$c.tar.gz
    gzip -t $F/$c.tar.gz && rm -rf $F/$c/$split && tar -xzf $F/$c.tar.gz -C $F/$c && rm $F/$c.tar.gz && touch $F/$c/done
    echo "fleurs $split $c: $(ls $F/$c/$split | wc -l) files"
  done
  FLEURS_SPLIT=$split $H/run_in_container.sh fleurs_to_parquet.py
done
