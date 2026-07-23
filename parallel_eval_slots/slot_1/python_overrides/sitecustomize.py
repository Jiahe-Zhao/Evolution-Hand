import importlib
import os

_OVERRIDE_ROOT = '/home/zjh/Evolution_PC/parallel_eval_slots/slot_1/python_overrides'
_ISAACLAB_TASKS_ROOT = os.path.join(_OVERRIDE_ROOT, "isaaclab_tasks")
_EVOLUTION_TASKS_ROOT = os.path.join(_ISAACLAB_TASKS_ROOT, "evolution_tasks")
_SUBPKG_PATHS = {
    "isaaclab_tasks": _ISAACLAB_TASKS_ROOT,
    "isaaclab_tasks.evolution_tasks": _EVOLUTION_TASKS_ROOT,
    "isaaclab_tasks.evolution_tasks.current_right_hand": os.path.join(_EVOLUTION_TASKS_ROOT, "current_right_hand"),
    "isaaclab_tasks.evolution_tasks.current_left_hand": os.path.join(_EVOLUTION_TASKS_ROOT, "current_left_hand"),
    "isaaclab_tasks.evolution_tasks.task_grasp": os.path.join(_EVOLUTION_TASKS_ROOT, "task_grasp"),
    "isaaclab_tasks.evolution_tasks.task_strike": os.path.join(_EVOLUTION_TASKS_ROOT, "task_strike"),
    "isaaclab_tasks.evolution_tasks.task_manipulation": os.path.join(_EVOLUTION_TASKS_ROOT, "task_manipulation"),
    "isaaclab_tasks.evolution_tasks.task_stone": os.path.join(_EVOLUTION_TASKS_ROOT, "task_stone"),
}

for _module_name, _path in _SUBPKG_PATHS.items():
    if not os.path.isdir(_path):
        continue
    _module = importlib.import_module(_module_name)
    _module_path = list(getattr(_module, "__path__", []))
    if _path not in _module_path:
        _module.__path__.insert(0, _path)
