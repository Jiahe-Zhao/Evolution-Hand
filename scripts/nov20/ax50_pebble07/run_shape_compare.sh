#!/bin/bash
set -euo pipefail
cd /home/zjh/Evolution_PC
P=/home/zjh/Evolution_PC_flip/IsaacLab3/.venv/bin/python
S=scripts/nov20/ax50_pebble07
O=outputs/nov20/ax50_pebble07
for variant in original topwide; do
 tag=shape_${variant}_v1
 RUN_TAG=$tag OBJECT_XML=$PWD/$S/assets/object_${variant}_v1.xml "$P" "$S/run_shape_compare.py" > "$O/${tag}.log" 2>&1
 "$P" "$S/report_trial.py" "$tag" > "$O/${tag}_report.log" 2>&1
 "$P" "$S/export_contacts.py" "$tag" >> "$O/${tag}_report.log" 2>&1
 "$P" "$S/render_pebble.py" "$tag" > "$O/${tag}_render.log" 2>&1
 done
"$P" "$S/postprocess_shape_compare.py"
printf 'both completed\n'
