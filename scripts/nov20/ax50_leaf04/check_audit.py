from contact_audit import *
import json
p=json.loads((OUT_DIR/'plan_tipclear.json').read_text());h=AuditedLeafRuntime(side='left',wrist='six_axis',position=p['position'],orientation=p['orientation'],gravity=(0,0,-9.81));h.vbd.leaf_contact_ke=500.;h.vbd.leaf_contact_kd=.05
h.advance(.001,dt=.0000625);print('AUDIT1',h.audit_values(),flush=True);h.advance(.001,dt=.0000625);print('AUDIT2',h.audit_values(),flush=True)
assert h.audit_values()[0]==16
