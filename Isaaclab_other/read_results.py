"""从isaaclab的project中读取最新模型的raw"""



import os
import re
from datetime import datetime


def get_latest_reward(raw_path):
    """
    Reads the highest reward value from the most recent folder in the specified path.

    Args:
        raw_path (str): The root path containing the date-time named folders.

    Returns:
        float: The highest reward value found in the latest folder, or None if no valid files are found.
    """
    # Get all date-time folders
    folders = [f for f in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, f))]

    # Parse folders into datetime objects
    datetime_folders = []
    for folder in folders:
        try:
            folder_datetime = datetime.strptime(folder, "%Y-%m-%d_%H-%M-%S")
            datetime_folders.append((folder_datetime, folder))
        except ValueError:
            # Skip folders not matching the date-time format
            continue

    if not datetime_folders:
        print("No valid date-time folders found.")
        return None

    # Find the most recent folder
    latest_folder = max(datetime_folders, key=lambda x: x[0])[1]
    latest_folder_path = os.path.join(raw_path, latest_folder, "nn")

    if not os.path.exists(latest_folder_path):
        print(f"Folder 'nn' not found in {latest_folder}.")
        return None

    # Pattern to match the reward in filenames
    reward_pattern = re.compile(r"rew_([-+]?\d*\.\d+|\d+)")
    max_reward = None

    # Iterate through model files in the "nn" folder
    for file_name in os.listdir(latest_folder_path):
        match = reward_pattern.search(file_name)
        if match:
            reward = float(match.group(1))
            if max_reward is None or reward > max_reward:
                max_reward = reward

    if max_reward is None:
        print("No valid model files found in the latest folder.")
        return None

    print(f"Maximum reward in the latest folder ({latest_folder}): {max_reward}")
    return max_reward


def get_reward_from_run_dir(run_dir):
    """
    Read the highest reward from a specific run directory.

    Args:
        run_dir (str): Run directory such as .../logs/evolution_task/<run_name>

    Returns:
        float | None: Highest reward in the run, or None if not found.
    """
    latest_folder_path = os.path.join(run_dir, "nn")
    if not os.path.exists(latest_folder_path):
        print(f"Folder 'nn' not found in run dir: {run_dir}")
        return None

    reward_pattern = re.compile(r"rew_([-+]?\d*\.\d+|\d+)")
    max_reward = None
    for file_name in os.listdir(latest_folder_path):
        match = reward_pattern.search(file_name)
        if match:
            reward = float(match.group(1))
            if max_reward is None or reward > max_reward:
                max_reward = reward

    if max_reward is None:
        print(f"No valid model files found in run dir: {run_dir}")
        return None

    print(f"Maximum reward in run dir ({run_dir}): {max_reward}")
    return max_reward




# Example usage
#raw_path = "/home/p/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/evolution_tasks/logs/evaluation_hand"
  # Replace with your actual path
#get_latest_reward(raw_path)

def get_latest_folder(raw_path):
    """
    获取指定路径下最新日期-时间格式的文件夹。

    Args:
        raw_path (str): 父目录路径。

    Returns:
        str: 最新文件夹的完整路径，如果不存在符合要求的文件夹，则返回 None。
    """
    # 获取目录下所有子文件夹
    subfolders = [
        os.path.join(raw_path, folder) for folder in os.listdir(raw_path)
        if os.path.isdir(os.path.join(raw_path, folder)) and
           folder.count('-') == 4 and folder.count('_') == 1
    ]

    # 根据日期-时间格式排序
    subfolders.sort(key=lambda x: datetime.strptime(os.path.basename(x), "%Y-%m-%d_%H-%M-%S"), reverse=True)

    # 返回最新的文件夹路径
    return subfolders[0] if subfolders else None


def check_finished_folder_exists(raw_path):
    """
    检查最新日期-时间的文件夹中是否存在 'finished' 文件夹。

    Args:
        raw_path (str): 父目录路径。

    Returns:
        bool: 如果存在则返回 True，否则返回 False。
    """
    latest_folder = get_latest_folder(raw_path)
    if latest_folder:
        finished_folder = os.path.join(latest_folder, "finished")
        exists = os.path.isdir(finished_folder)
        print("fale:",latest_folder)
        print(f"'finished' 文件夹是否存在: {exists}")
        return exists
    else:
        print("未找到符合日期-时间命名的文件夹。")
        return False


def check_finished_folder_exists_in_run_dir(run_dir):
    """
    Check whether a specific run directory has a finished marker.

    Args:
        run_dir (str): Run directory such as .../logs/evolution_task/<run_name>

    Returns:
        bool: Whether the finished marker exists.
    """
    finished_folder = os.path.join(run_dir, "finished")
    exists = os.path.isdir(finished_folder)
    print(f"run_dir: {run_dir}")
    print(f"'finished' 文件夹是否存在: {exists}")
    return exists
