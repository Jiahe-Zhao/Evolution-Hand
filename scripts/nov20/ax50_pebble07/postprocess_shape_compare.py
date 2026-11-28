
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import subprocess,time,json
R=Path((str(_EVOLUTION_ROOT)));O=R/'outputs/nov20/ax50_pebble07';P=(_evo_sys.executable)
while not (O/'shape_topwide_v1_render.json').exists():time.sleep(20)
for script,args in [('prepare_shape_cameras.py',[]),('render_pebble.py',['shape_original_v1']),('render_pebble.py',['shape_topwide_v1'])]:
 subprocess.run([P,str(R/'scripts/nov20/ax50_pebble07'/script)]+args,cwd=R,check=True)
subprocess.run([P,str(R/'scripts/nov20/ax50_pebble07/compare_shapes_report.py')],cwd=R,check=True)
font=str(_EVOLUTION_ROOT / 'assets/fonts/NotoSansCJK-Regular.ttc')
filters=[]
for i,label in enumerate(['原石子','圆滑、轻微上宽下窄石子']):
 filters.append(f"[{i}:v]crop=600:534:600:66,pad=600:574:0:40:color=0x203244,drawtext=fontfile={font}:text='{label} | 同动作，2倍慢放':fontcolor=white:fontsize=20:x=12:y=8[v{i}]")
filters.append('[v0][v1]hstack=inputs=2:shortest=1[out]')
subprocess.run(['ffmpeg','-v','error','-y','-i',str(O/'shape_original_v1.mp4'),'-i',str(O/'shape_topwide_v1.mp4'),'-filter_complex',';'.join(filters),'-map','[out]','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(O/'shape_comparison.mp4')],check=True)
for tag in ['shape_original_v1','shape_topwide_v1','shape_comparison']:
 subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(O/(tag+'.mp4')),'-f','null','-'],check=True)
print('POSTPROCESS_COMPLETE',flush=True)
