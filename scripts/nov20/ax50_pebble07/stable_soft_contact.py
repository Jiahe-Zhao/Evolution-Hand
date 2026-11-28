"""Scene07 candidate: deduplicate equivalent contacts and condition penalty stiffness.

Only target soft contacts are affected. Tissue elasticity, damping, mu, rigid
contacts and self-contact remain unchanged. 1125 N/m is a numerical contact
penalty, not a fitted human tissue modulus. Validate overlap and force feedback.
"""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import warp as wp
from contact_dedup import install as dedup_install
from soft_coupling_probe import set_target_kd

def install(h,ke=1125.0,kd=0.5):
 if getattr(h,"stable_contact_info",None) is not None:return
 dedup_install(h)
 old=h.vbd._refresh_body_particle_contact_state
 def refresh(contacts,refresh):
  old(contacts,refresh)
  if contacts is not None:
   for values in [h.vbd.body_particle_contact_penalty_k,h.vbd.body_particle_contact_material_ke]:
    wp.launch(set_target_kd,contacts.soft_contact_max,inputs=[contacts.soft_contact_count,contacts.soft_contact_shape,h.model.shape_body,h.item,float(ke),values])
   wp.launch(set_target_kd,contacts.soft_contact_max,inputs=[contacts.soft_contact_count,contacts.soft_contact_shape,h.model.shape_body,h.item,float(kd),h.vbd.body_particle_contact_material_kd])
 h.vbd._refresh_body_particle_contact_state=refresh;h._advance_graph=None
 h.stable_contact_info=dict(scope='target soft contact only',ke_N_per_m=ke,kd=kd,endpoint_bary_threshold=.99999,duplicate_position_tolerance_m=1e-6,normal_dot_threshold=.99999,mu_changed=False,tissue_changed=False,self_contact_disabled=False)
