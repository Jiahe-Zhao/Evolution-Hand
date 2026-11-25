from pebble_runtime import *
h=PebbleRuntime(side='left',wrist='six_axis',position=(.25,.25,.35),gravity=(0,0,-9.81))
j=json.loads((OUT/'warning_probe.json').read_text());bad=np.array(j['records'][0]['bad_edges']);m=h.model;edges=m.edge_indices.numpy();keys=np.sort(edges[:,2:4],axis=1);props=m.edge_bending_properties.numpy();a=m.soft_mesh_adjacency;tri=m.tri_indices.numpy();rows=[]
for e in bad:
 ids=np.flatnonzero(np.all(keys==e,axis=1));assert len(ids)==1;idx=int(ids[0]);adj=[]
 for v in e:
  incident=np.asarray(a.v_adj_edges)[int(a.v_adj_edges_offsets[v]):int(a.v_adj_edges_offsets[v+1])];adj.append(bool(np.any(incident==idx)))
 faces=np.flatnonzero(np.any(tri==e[0],axis=1)&np.any(tri==e[1],axis=1));rows.append(dict(endpoints=e.tolist(),edge_index=idx,vertex_edge_adjacency_present=adj,incident_faces=len(faces),bending_properties=props[idx].tolist()))
r=dict(scope='topology audit; no remeshing or contact filtering',edges=rows,all_present=all(all(x['vertex_edge_adjacency_present']) for x in rows),all_zero_bending=all(np.all(np.array(x['bending_properties'])==0) for x in rows),surface_faces_retained=all(x['incident_faces']==4 for x in rows))
(OUT/'verified_topology.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
