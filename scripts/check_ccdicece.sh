#!/bin/bash
# Health check for the CC-DiceCE run (read-only: changes nothing).
# Usage on the server:  bash check_ccdicece.sh
source ~/SmallLesionMRI/scripts/setvars.sh >/dev/null 2>&1
R="$nnUNet_results/Dataset333_MSLesSeg/nnUNetTrainerCCDiceCE__nnUNetPlans__3d_fullres"
LOG_SCRIPT=~/SmallLesionMRI/scripts/ccdicece.log
line() { printf '\n===== %s =====\n' "$1"; }

line "1. Processes (expect ONE nnUNetv2_train ... nnUNetTrainerCCDiceCE)"
ps -eo pid,etime,%cpu,%mem,stat,args | grep -E "nnUNetv2_(train|predict)|run_ccdicece" | grep -v grep \
  || echo "!! no training process found"
echo "data-loader workers (child python processes): $(pgrep -f nnUNetv2_train | wc -l) python processes total"
echo "(STAT 'D' = waiting on disk; 'Z' = zombie/dead worker)"

line "2. GPU"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

line "3. Latest training log"
LOG=$(ls -t "$R"/fold_*/training_log_*.txt 2>/dev/null | head -1)
if [ -z "$LOG" ]; then
  echo "!! no training_log found under $R"
else
  echo "file: $LOG"
  AGE=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  echo "last write: $((AGE/60)) min $((AGE%60)) s ago"
  echo "epochs finished: $(grep -c 'Epoch time' "$LOG")   (150 = done; validation/export follows)"
  echo "--- last epoch times:"; grep 'Epoch time' "$LOG" | tail -3
  echo "--- last 15 lines:"; tail -15 "$LOG"
fi

line "4. Checkpoints in fold folders"
ls -l --time-style=+%m-%d_%H:%M "$R"/fold_*/checkpoint_*.pth 2>/dev/null || echo "(none yet; nnU-Net saves checkpoint_latest every 50 epochs)"

line "5. Script log (Voronoi GPU/CPU, errors)"
if [ -f "$LOG_SCRIPT" ]; then
  grep -E "Voronoi on|cupy Voronoi failed|=== \[" "$LOG_SCRIPT" | tail -8
  echo "--- errors/warnings in script log:"
  grep -inE "error|traceback|killed|out of memory|died|exception" "$LOG_SCRIPT" | tail -10 || echo "none"
else
  echo "(no $LOG_SCRIPT — if started with tee elsewhere, check that file)"
fi

line "6. tmux pane (last 20 lines)"
tmux capture-pane -pt ccdicece -S -20 2>/dev/null || echo "(no tmux session named ccdicece)"

line "7. Disk space and kernel OOM kills"
df -h "$nnUNet_results" "$nnUNet_preprocessed" | awk 'NR==1||!seen[$0]++'
dmesg -T 2>/dev/null | grep -iE "out of memory|oom-kill|killed process" | tail -3 || true
free -h | head -2

line "8. What the main process is doing right now (needs py-spy; optional)"
PID=$(pgrep -of "nnUNetv2_train.*CCDiceCE")
if [ -n "$PID" ] && command -v py-spy >/dev/null; then
  py-spy dump --pid "$PID" 2>&1 | head -40
else
  echo "skip (install with: pip install py-spy, then rerun)"
fi
