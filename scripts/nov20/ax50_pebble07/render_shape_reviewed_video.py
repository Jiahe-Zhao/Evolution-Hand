"""Compact reviewed comparison for delivery; uses recorded renders only."""
from pathlib import Path
import subprocess,json
O=Path('/home/zjh/Evolution_PC/outputs/nov20/ax50_pebble07')
r=json.loads((O/'shape_comparison_result.json').read_text());assert all(x['acceptance_passed'] is False for x in r['variants'].values())
f='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
vf=f"drawbox=x=0:y=0:w=iw:h=40:color=0x203244:t=fill,drawtext=fontfile={f}:text='原石子：未抓起 | 同动作，2倍慢放':fontcolor=white:fontsize=20:x=12:y=8,drawtext=fontfile={f}:text='新石子：穿入拇指，未通过 | 2倍慢放':fontcolor=white:fontsize=20:x=612:y=8"
p=O/'shape_comparison_reviewed.mp4'
subprocess.run(['ffmpeg','-v','error','-y','-i',str(O/'shape_comparison.mp4'),'-vf',vf,'-c:v','libx264','-crf','26','-pix_fmt','yuv420p','-movflags','+faststart',str(p)],check=True)
subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(p),'-f','null','-'],check=True)
print('reviewed video bytes',p.stat().st_size)
