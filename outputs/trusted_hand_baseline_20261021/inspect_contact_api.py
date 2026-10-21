import os,argparse
from isaaclab.app import AppLauncher
import omni.ext._impl.fast_importer as fi
old=fi._fast_walk
def walk(path,*a,**k):
 if os.path.realpath(path)=='/home/zjh/Evolution_PC/evolution_tasks/logs':return
 yield from old(path,*a,**k)
fi._fast_walk=walk
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
from pxr import PhysxSchema
import omni.physx
print('BASELINE_MATERIAL_API',[x for x in dir(PhysxSchema.PhysxMaterialAPI) if 'ompliant' in x or 'riction' in x],flush=True)
print('BASELINE_REPORT_API',[x for x in dir(PhysxSchema.PhysxContactReportAPI) if 'Threshold' in x],flush=True)
api=omni.physx.get_physx_simulation_interface()
print('BASELINE_CALLBACK',[x for x in dir(api) if 'contact' in x],flush=True)
print('BASELINE_CALLBACK_DOC',api.subscribe_contact_report_events.__doc__,flush=True)
import omni.physics.tensors as tensors
print('BASELINE_TENSOR_VIEW',[x for x in dir(tensors) if 'View' in x],flush=True)
os._exit(0)
