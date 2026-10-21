from pathlib import Path
p=Path(__file__).parent
s=(p/'probe_original.py').read_text()
s=s.replace("str(Path(__file__).resolve().parents[1] / 'evolution_tasks' / 'task_suite')", "'/home/zjh/Evolution_PC/evolution_tasks/task_suite'")
s=s.replace('str(Path(__file__).resolve().parents[1] / "evolution_tasks" / "logs")', "'/home/zjh/Evolution_PC/evolution_tasks/logs'")
s=s.replace("Path(args.mechanical_asset_dir).resolve() / 'left/urdf/current_agent.urdf'", "Path(args.mechanical_asset_dir).resolve() / 'right/urdf/current_agent.urdf'").replace('Path(args.mechanical_asset_dir).resolve() / "left/urdf/current_agent.urdf"', 'Path(args.mechanical_asset_dir).resolve() / "right/urdf/current_agent.urdf"')
s=s.replace("if args.task!='strike':raise ValueError('Driven wrist diagnostic currently supports Strike only')", "if args.task not in ('strike','forage'):raise ValueError('Wrist only for moving-wrist tasks')")
s=s.replace('        raw_env = gym.make(env_id,', "        from baseline_hooks import install as install_baseline\n        install_baseline(env_id,env_cfg,output_dir,args.task)\n        raw_env = gym.make(env_id,")
s=s.replace("        if os.environ.get('GRAPH_LOAD_PLAN')=='1':", "        from four_task_rollout import run\n        run(raw_env,graph,output_dir,args,_task_evidence)\n        os._exit(0)\n        if os.environ.get('GRAPH_LOAD_PLAN')=='1':",1)
(p/'verify_four_tasks.py').write_text(s)
