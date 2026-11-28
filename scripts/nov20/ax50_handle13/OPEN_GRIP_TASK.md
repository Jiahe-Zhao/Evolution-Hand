任务：为13活动提梁提出“先张手扶起，再曲指握持”CPU几何候选，不能改物体或启动GPU仿真。
项目/home/zjh/Evolution_PC，Python=/home/zjh/Evolution_PC_flip/IsaacLab3/.venv/bin/python。
读 scripts/nov20/ax50_handle13/fit_hook.py、handle_runtime.py、outputs/nov20/ax50_handle13/plan.json/object.xml，以及ax50_leaf04/fit_leaf_tipclear.py、inspect_tip_surface.py。旧hand_v3_full只有末节承托，用户不接受作为自然握持。
只可写 scripts/nov20/ax50_handle13/fit_open_grip.py、outputs/nov20/ax50_handle13/open_grip_plan.json、scripts/nov20/ax50_handle13/OPEN_GRIP_NOTES.md。不改共享文件/原plan/运行器/md。
要求：先中指/无名指PIP=DIP=0、MCP轻微弯曲<=0.3，近似张手，利用指腹扶起横杆。横杆初始中心(-.112,-.022,.052)，6x6mm、长60mm沿y；原器身中心(-.176,-.022,.026)，半宽x16mm/半宽y25mm，高约50mm。接触目标在横杆底面z=.049附近、y=-.035/-.010；从右侧进入，末端不要延伸至器身。使用原plan朝向(可微调yaw)，拟合手腕与指位，报告实际掌侧接触点及指尖几何范围/器身间隙。参考leaf正确局部+Y掌侧，检查真实mesh不能猜tip长度。
再给出提梁已扶起到约75–85度时，一指保持、另一指逐步曲指的几何建议。允许移动手腕使掌侧点持续在横杆附近，不能将背侧承托称为握持。不一定要拟合完所有阶段，优先完成可靠张手扶起初态，难点诚实记录。
MuJoCo MJB bodies多一个world，Newton编号差1；按body名称匹配。MJ四元数WXYZ，Newton XYZW。所有关节在限位内。CPU FK不是物理通过。
执行fit脚本生成JSON，用简短notes给出拟合误差、姿态、推荐控制阶段和未解决问题。不要读取密钥。
