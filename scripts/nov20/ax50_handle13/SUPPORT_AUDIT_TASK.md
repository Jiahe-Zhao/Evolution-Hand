请直接完成一个小脚本，不作长篇规划。项目/home/zjh/Evolution_PC，Python=/home/zjh/Evolution_PC_flip/IsaacLab3/.venv/bin/python。只允许创建 scripts/nov20/ax50_handle13/audit_open_support.py；不能改模型/运行器/md，不能启动GPU。
读取run_open_support.py输出格式，生成只读审计：输入tag命令行参数，读取outputs/nov20/ax50_handle13/{tag}_trace.csv、_contacts.csv、_model.json、{tag}.npz。
1 按phase和object_shape统计接触：grip/arm0/arm1/item，记录次数、载荷峰值、哪些手骨段；只计normal_N>1e-5。
2 hand_local_y_m为Newton刚体局部接触坐标，手指掌侧+Y、背侧-Y。对link_3_1/2/3、link_4_1/2/3，统计grip有载接触中y>1mm比例、y<-1mm比例，中央区域标未知；不要将端点统计称逐子步统计。
3 npz新增joint_q实际位置，model.action_names与action_indices匹配digit_3_flex_1/2/3及digit_4_flex_1/2/3，输出全程实际最大绝对角度（rad/deg），不能只看targets。
4 输出最大提梁角、是否超过90度、最后0.2秒提梁角范围、hand-item全程max_body_contact_N、最后0.2秒min_bail_contact_N（逐子步窗口），该阶段仅扶梁，不做完整抓持成功声明。
写出{tag}_open_support_audit.json。文件未就绪时仅完成脚本和语法检查，不等待循环；不要用其他标签伪造数据。只执行CPU脚本。最后简短汇报。
