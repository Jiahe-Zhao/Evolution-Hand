from pathlib import Path
import subprocess,time,json
R=Path('/home/zjh/Evolution_PC');O=R/'outputs/nov20/ax50_pebble07';P='/home/zjh/Evolution_PC_flip/IsaacLab3/.venv/bin/python'
while not (O/'shape_topwide_v1_render.json').exists():time.sleep(20)
subprocess.run([P,str(R/'scripts/nov20/ax50_pebble07/compare_shapes_report.py')],cwd=R,check=True)
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
filters=[]
for i,label in enumerate(['原石子','圆滑、轻微上宽下窄石子']):
 filters.append(f"[{i}:v]crop=600:534:600:66,pad=600:574:0:40:color=0x203244,drawtext=fontfile={font}:text='{label} | 同动作，2倍慢放':fontcolor=white:fontsize=20:x=12:y=8[v{i}]")
filters.append('[v0][v1]hstack=inputs=2:shortest=1[out]')
subprocess.run(['ffmpeg','-v','error','-y','-i',str(O/'shape_original_v1.mp4'),'-i',str(O/'shape_topwide_v1.mp4'),'-filter_complex',';'.join(filters),'-map','[out]','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(O/'shape_comparison.mp4')],check=True)
for tag in ['shape_original_v1','shape_topwide_v1','shape_comparison']:
 subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(O/(tag+'.mp4')),'-f','null','-'],check=True)
print('POSTPROCESS_COMPLETE',flush=True)
