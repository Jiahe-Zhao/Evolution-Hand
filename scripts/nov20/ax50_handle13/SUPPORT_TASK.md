任务：为AX50活动提梁13编写验收/渲染辅助脚本，不运行物理仿真，不改物体/手/运行器。
目录 /home/zjh/Evolution_PC，允许写scripts/nov20/ax50_handle13/audit_handle.py、render_handle.py 和 SUPPORT_NOTES.md，其他文件只读。不要改运行器、共享代码或远端md。
先读 docs/TASK13_AX50_HANDLE.md、scripts/nov20/ax50_handle13/handle_runtime.py、scripts/nov20/ax50_leaf04/contact_audit.py、render_audit.py、run_hand_v8.py。参考原 scripts/nov20/env.py handle成功条件。
拟定输入 outputs/nov20/ax50_handle13/{tag}.npz，字段time,body_q,body_names,particles,tri,targets；物体最后两个body=item,bail。{tag}_trace.csv字段time_s,phase,hinge_rad,lift_m,tilt_deg,body_contact_N,bail_contact_N,finite,substeps,qualified_substeps。原物体object.xml。
验收: 相对初态提梁转起>=45度，器身相对初态升高>=.025m，器身倾斜<=15度，手不得接触器身(阈值.0005N)，仅经提梁载荷>.01N，连续>=.5s。逐子步的qualified_substeps==substeps并且有限。初态是仿真第一帧，不随手动作移动。不要把IK或导入算通过。
render_handle.py 用AX50 VisualSkin和保存实际软体particles/tri、原object.xml几何按记录的body姿态重放；两个清晰视角，显示器身底部与地面净距、提梁铰链，中文不能字体缺失则英文。视频2x慢放，输出mp4和png曲线。可以复用smooth_render.panel/render_core.strip/encode，注意strip返回PIL.Image。不能编造未生成数据。报告通过也必须由实际文件计算。
仅可使用 /home/zjh/Evolution_PC_flip/IsaacLab3/.venv/bin/python 做py_compile/读取文件。不使用系统python导入仿真依赖。不启动GPU物理仿真。完成写SUPPORT_NOTES.md并退出，限制本次任务在这两个脚本。
