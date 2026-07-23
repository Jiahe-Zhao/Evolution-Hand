"""主要的进化函数"""
import os

from class_population import *
from pprint import pprint
from variation import *
from evaluation_interface import evaluation

# 定义基本信息，不考虑更底层的性别因素
max_generation = 40 # 最大变异代数
max_population = 10 # 环境最大承载量，超过此量就要末位淘汰
max_variation = 2 # 每个个体下一代的最大变异数量，也是每个个体最多能出生的后代数，因为事实上每个个体都在变异
initial_population_size = 8  # 初始种群数量（包含基准个体）
initial_population_attempts = 200
initial_population_variation = 0.05
initial_population_length = 0.02
next_task_reward=5000#进入下一任务的reward
stable_success_generations = 2  # 连续几代成功才切任务
max_generation_per_task=10
variation_probabilities = {  # 各项变异的发生概率
    "change_link_length": 1,
    "change_link_radius": 0,
    "remove_link": 0,
    "add_link": 0,
    "change_joint_origin_translation": 0,
    "change_joint_origin_rpy": 0, }
experiment_save_path = "exp_20250508_1" # 保存谱系图的路径

evaluation_taks=["Isaac-EvolutionHand-Grasp-v0",
                 "Isaac-EvolutionHand-Manipulation-v0",
                 "Isaac-EvolutionHand-Strike-v0",
                 "Isaac-EvolutionHand-StoneGrind-v0"]
# evaluation_taks={"Isaac-Hand-Cube-Evolution-v0"}
check_point = "human"

# 给到 isaaclab 的数据存放地点
HOME_DIR = os.path.expanduser("~")
EVOLUTION_ROOT = os.path.join(HOME_DIR, "Evolution_PC")
ISAACLAB_ROOT = os.path.join(HOME_DIR, "IsaacLab")
ISAACLAB_TASK_ROOT = os.path.join(
    ISAACLAB_ROOT, "source", "isaaclab_tasks", "isaaclab_tasks", "evolution_tasks"
)
ISAACLAB_OTHER_ROOT = os.path.join(EVOLUTION_ROOT, "Isaaclab_other")
EVOLUTION_LOG_ROOT = os.path.join(EVOLUTION_ROOT, "evolution_tasks", "logs", "evolution_task")

isaaclab_urdf_path = os.path.join(ISAACLAB_OTHER_ROOT, "agent_for_isaaclab", "urdf", "current_agent.urdf")
isaaclab_urdf_mesh_path = os.path.join(ISAACLAB_OTHER_ROOT, "agent_for_isaaclab", "mesh")
isaaclab_urdf_code_path = os.path.join(
    ISAACLAB_TASK_ROOT, "current_right_hand", "current_right_hand_cfg.py"
)  # 这个要放到 isaaclab 文件夹中
isaaclab_env_code_path = [
    os.path.join(ISAACLAB_TASK_ROOT, "task_grasp", "evolution_grasp_env_cfg.py"),
    os.path.join(ISAACLAB_TASK_ROOT, "task_manipulation", "evolution_manipulation_env_cfg.py"),
    os.path.join(ISAACLAB_TASK_ROOT, "task_strike", "evolution_strike_env_cfg.py"),
    os.path.join(ISAACLAB_TASK_ROOT, "task_stone", "evolution_stone_grind_env_cfg.py"),
]

isaaclab_test_result_path = [
    EVOLUTION_LOG_ROOT,
    EVOLUTION_LOG_ROOT,
    EVOLUTION_LOG_ROOT,
    EVOLUTION_LOG_ROOT,
]

# mirror
isaaclab_mirror_urdf_path = os.path.join(
    ISAACLAB_OTHER_ROOT, "agent_for_isaaclab_mirror", "urdf", "current_agent.urdf"
)
isaaclab_mirror_urdf_mesh_path = os.path.join(ISAACLAB_OTHER_ROOT, "agent_for_isaaclab_mirror", "mesh")
isaaclab_mirror_urdf_code_path = os.path.join(
    ISAACLAB_TASK_ROOT, "current_left_hand", "current_left_hand_cfg.py"
)  # 这个要放到 isaaclab 文件夹中
# isaaclab_mirror_env_code_path =("${ISAACLAB_ROOT}/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/task_stone/evolution_stone_grind_env_cfg.py"
#                            )

# isaaclab_mirror_test_result_path = ("${ISAACLAB_ROOT}/source/isaaclab_tasks/isaaclab_tasks/evolution_tasks/logs/evolution_hand_stone_grind"
#                              )#${ISAACLAB_ROOT}/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/evolution_tasks/logs/evaluation_hand



new_lineage = True

if new_lineage:
    # 初始化谱系
    hand_lineage = Lineage()



# 导入初始对象
if check_point == "human":
    from human_hand_agent import initial_agent_hand
elif check_point == "gorilla":
    from gorilla_hand_agent import initial_agent_hand

initial_population = seed_initial_population(
    initial_agent_hand,
    population_size=initial_population_size,
    include_base=True,
    max_attempts=initial_population_attempts,
    standard_variation=initial_population_variation,
    standard_length=initial_population_length,
)
for idx, urdf in enumerate(initial_population):
    new_id = uuid.uuid4().hex
    urdf["evolution_id"] = new_id
    hand_lineage.add_individual(-1, idx, urdf, 0, new_id)
# print(hand_lineage.lineage)
# 执行变异
# Iterate through generations
mutation_stats = {}  # 放在最外层，跨代共享
task_count = len(evaluation_taks)
mutation_stats_by_task = {}  # 每个任务单独一套统计数据

task_index=0
task_generation=0#记录该任务进化的代数
task_changed = True#任务切换标志
for current_generation in range(0, max_generation+1): #max_generation+1
    print(f"Generation {current_generation}: Starting mutation and evaluation.")
    if task_index>=len(evaluation_taks):
        break
    print(f"=== Task {task_index + 1}: {evaluation_taks[task_index]} ===")
    current_task = evaluation_taks[task_index]
    print(f"task:{current_task}")
    current_env_code_path = isaaclab_env_code_path[task_index]
    # ///////////print(f"111:{current_env_code_path}")
    current_result_path = isaaclab_test_result_path[task_index]
    if current_task not in mutation_stats:
        mutation_stats[current_task] = {}
    task_generation +=1
    # 记录连续成功的 reward 达标的代数
    successful_generations = 0
    max_reward_this_generation =float('-inf')
    # 获取这一代中的全部存活个体的代码，组成一个list
    surviving_individuals = hand_lineage.get_surviving_individuals_in_generation(current_generation)
    # 遍历这一代中的全部存活个体
    for current_individual in surviving_individuals:
        # 读取urdf信息
        # print(current_generation,current_individual)
        current_urdf = hand_lineage.lineage[(current_generation,current_individual)]['urdf_info']
        original_score = hand_lineage.lineage[(current_generation, current_individual)]['task_score']
        # success_num=0
        # print("current urdf:",current_urdf)
        for trail in range(max_variation):#
            while True :
                # 生成变异
                link_code, task_code, strength = choose_target_reward(current_urdf, mutation_stats)
                # link_code, task_code, strength = choose_target(current_urdf, variation_probabilities)
                print("link_code, task_code, strength:",link_code, task_code, strength)
                # 执行变异
                success_tag, new_urdf = variation(current_urdf, link_code, task_code, strength,
                                                standard_variation=0.1, standard_length=0.05)
                print("success_tag:", success_tag)
                if success_tag:
                    break
            # print("link_code, task_code, strength:",link_code, task_code, strength)
            # print("success_tag:", success_tag)
            print(new_urdf)
            # 如果变异符合物理约束
            if success_tag:
                # success_num+=1
                # 进行仿真评分
                print(current_task)
                current_score = evaluation(new_urdf, {current_task},
                                           isaaclab_urdf_path, isaaclab_urdf_mesh_path, isaaclab_urdf_code_path,
                                           isaaclab_mirror_urdf_path, isaaclab_mirror_urdf_mesh_path, isaaclab_mirror_urdf_code_path,
                                           current_env_code_path, current_result_path)
                # 将个体放入谱系图
                hand_lineage.add_individual(current_generation,current_individual,new_urdf,current_score,uuid.uuid4().hex)
                max_reward_this_generation = max(max_reward_this_generation, current_score)
                # print(f"trail;{trail}")
                if not task_changed: #只在任务为切换的时候统计
                    key = (task_code, link_code)
                    if key not in mutation_stats[current_task]:
                        mutation_stats[current_task][key] = {"trials": 0, "improvements": 0, "score_delta": 0.0}
                    mutation_stats[current_task][key]["trials"] += 1
                    if current_score > original_score:
                        mutation_stats[current_task][key]["improvements"] += 1
                        mutation_stats[current_task][key]["score_delta"] += (current_score - original_score)
                # 保存 mutation_stats 到 JSON 文件
                with open(experiment_save_path + '_mutation_stats.txt', 'w') as f:
                    pprint(mutation_stats, stream=f)

                print(f"mutation_stats has been update to {experiment_save_path + '_mutation_stats.txt'}")
            if task_changed:
                print("任务刚刚切换，本次将不会记录 mutation 的 score delta 和提升情况")
                task_changed = False  # 下一代开始统计 score delta
            # if trail==max_variation-1 and success_num==0: #没有成功的 变异
            #     trail -=1
    # 每个个体都发生完变异后评价整个种群的状态，进行淘汰
    hand_lineage.evaluate_and_eliminate_individuals_in_generation(current_generation+1, max_population)
    # 储存当前谱系
    hand_lineage.save_to_file(experiment_save_path + '.json')
    
    
    # 判断是否连续成功
    if max_reward_this_generation >= next_task_reward:
        successful_generations += 1
    else:
        successful_generations = 0  # 重置计数器
    if successful_generations >= stable_success_generations:#stable_success_generations
        task_index +=1
        task_generation=0
        task_changed = True
        print(f"Task {task_index + 1} completed successfully. Moving to next task.")
    if task_generation>=max_generation_per_task:#max_generation_per_task
        task_index +=1
        task_generation=0
        task_changed = True
        print(f"Task {task_index + 1} reached max generations.")
    
# 保存 mutation_stats 到 JSON 文件
with open(experiment_save_path + '_mutation_stats.txt', 'w') as f:
    pprint(mutation_stats, stream=f)

print(f"mutation_stats has been saved to {experiment_save_path + '_mutation_stats.txt'}")

# 编写另一种加载训练一半的数据的方式
