"""Versioned morphology-derived contact geometry; no task-specific geometry.

Terminal pads are physical polymer sleeves over the bearing proxies. Internal
bearing collision surfaces are replaced by that sleeve, but their mass remains.
This is a simulation design, not a calibrated manufactured assembly.
"""
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

VERSION = "motion_contact_v3"


def pose(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        t[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return t


def rounded_box(extents, rounding):
    """Convex rounded box with flat faces and explicitly bounded dimensions."""
    half = np.asarray(extents) / 2
    rounding = min(rounding, float(half.min()) * .45)
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=rounding)
    points = np.concatenate([sphere.vertices + np.array([x, y, z]) * (half - rounding)
                             for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return trimesh.convex.convex_hull(points)


def bearing(radius, axis, center):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    m = trimesh.creation.cylinder(radius=radius, height=2 * radius, sections=32)
    m.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], axis))
    m.apply_translation(center)
    return m


def emit(link, directory, name, mesh, color, collision=True):
    filename = directory / (link.get("name") + "_" + name + ".stl")
    mesh.export(filename)
    for kind in (("visual", "collision") if collision else ("visual",)):
        node = ET.SubElement(link, kind, name=name)
        ET.SubElement(node, "origin", xyz="0 0 0", rpy="0 0 0")
        geometry = ET.SubElement(node, "geometry")
        ET.SubElement(geometry, "mesh", filename="../meshes/" + filename.name)
        if kind == "visual":
            mat = ET.SubElement(node, "material", name=name)
            ET.SubElement(mat, "color", rgba=" ".join(map(str, color)))


def inertia(link, pieces):
    """Add density-weighted solid moments, including displaced sleeve material."""
    masses, centers, moments = [], [], []
    for mesh, density in pieces:
        mesh = mesh.copy(); mesh.density = density
        masses.append(mesh.mass); centers.append(mesh.center_mass); moments.append(mesh.moment_inertia)
    mass = sum(masses)
    center = sum(m * c for m, c in zip(masses, centers)) / mass
    tensor = sum(I + m * (np.dot(c-center, c-center) * np.eye(3) - np.outer(c-center, c-center))
                 for m, c, I in zip(masses, centers, moments))
    eig = np.linalg.eigvalsh(tensor)
    if mass <= 0 or eig.min() <= 0 or eig[-1] > sum(eig[:2]) + 1e-12:
        raise ValueError("Invalid physical inertia: " + link.get("name"))
    old = link.find("inertial")
    if old is not None: link.remove(old)
    node = ET.SubElement(link, "inertial")
    ET.SubElement(node, "origin", xyz=" ".join(map(str, center)), rpy="0 0 0")
    ET.SubElement(node, "mass", value=str(mass))
    ET.SubElement(node, "inertia", **{k:str(tensor[i,j]) for k,i,j in
        [("ixx",0,0),("ixy",0,1),("ixz",0,2),("iyy",1,1),("iyz",1,2),("izz",2,2)]})
    return mass


def upgrade_contact_model(source, destination, manifest):
    source, destination = Path(source), Path(destination)
    tree = ET.parse(destination); root = tree.getroot()
    original = ET.parse(source).getroot()
    source_links = {x.get("name"):x for x in original.findall("link")}
    incoming = {j.find("child").get("link"):j for j in root.findall("joint")}
    outgoing = {}
    for j in root.findall("joint"):
        outgoing.setdefault(j.find("parent").get("link"), []).append(j)
    # Backward-compatible compilation of archived bone assets also corrects the
    # known redundant spread axis. An already independent axis is preserved.
    axes = []
    for j in root.findall("joint"):
        if not re.fullmatch(r"link_[2-5]_mcp_spread_joint", j.get("name")): continue
        flex = next((x for x in outgoing[j.find("child").get("link")]
                     if x.get("type") != "fixed"), None)
        a = np.fromstring(j.find("axis").get("xyz"), sep=" ")
        b = pose(flex.find("origin"))[:3,:3] @ np.fromstring(flex.find("axis").get("xyz"), sep=" ")
        old = a.copy()
        if abs(np.dot(a,b)) > .999:
            length_axis = pose(flex.find("origin"))[:3,:3] @ np.array([0.,0.,1.])
            a = np.cross(length_axis, b); a /= np.linalg.norm(a)
            j.find("axis").set("xyz", " ".join(map(str,a)))
        axes.append({"joint":j.get("name"),"old_axis":old.tolist(),"axis":a.tolist(),
                     "flexion_dot":float(np.dot(a,b))})
    folder = destination.parent.parent / "meshes"
    rows = []
    for link in root.findall("link"):
        name = link.get("name")
        if not re.fullmatch(r"link_[1-5]_\d+", name): continue
        src = source_links[name]; visual = src.find("visual")
        node = visual.find("geometry/mesh")
        m = trimesh.load(source.parent / node.get("filename"), force="mesh")
        m.apply_scale(np.fromstring(node.get("scale","1 1 1"),sep=" "))
        m.apply_transform(pose(visual.find("origin")))
        # Radius is measured transverse to the source bone; curvature in Y must
        # not silently inflate it. The source generated capsule is circular in X.
        radius = max(.0015, float(m.extents[0]) / 2)
        hr = .9 * radius
        children = outgoing.get(name, [])
        next_j = next((j for j in children if re.fullmatch(r"link_[1-5]_\d+",j.find("child").get("link"))),None)
        is_tip = not children
        end = pose(next_j.find("origin"))[:3,3] if next_j is not None else np.array([0.,0.,float(m.bounds[1,2]+m.bounds[0,2])])
        span = float(np.linalg.norm(end))
        if span < 2.5 * hr: raise ValueError(f"{name}: bearing span is too short for its radius")
        direction = end / span
        parent = incoming[name]
        axis0 = np.fromstring(parent.find("axis").get("xyz"),sep=" ") if parent.find("axis") is not None else np.array([1.,0.,0.])
        axis1 = pose(next_j.find("origin"))[:3,:3] @ np.fromstring(next_j.find("axis").get("xyz"),sep=" ") if next_j is not None and next_j.find("axis") is not None else axis0
        metal = [bearing(hr,axis0,[0,0,0]),bearing(hr,axis1,end)]
        for old in link.findall("visual")+link.findall("collision"):link.remove(old)
        pieces=[]
        if is_tip:
            # A real enclosing sleeve: both visual and collision are enlarged.
            # Clearance scales with morphology, rather than a fixed-world pad.
            skin = max(.0003, min(.0008, radius / 6))
            bounds=np.vstack([part.bounds for part in metal]);low=bounds.min(0)-skin;high=bounds.max(0)+skin
            body=rounded_box(high-low,skin*.8);body.apply_translation((low+high)/2)
            emit(link,folder,"finger_pad",body,[.08,.52,.35,1])
            pieces=[(body,1100.)]+[(part,2700.-1100.) for part in metal]
            for i,part in enumerate(metal):emit(link,folder,["proximal_bearing","distal_bearing"][i],part,[.8,.43,.18,1],False)
            rows.append({"link":name,"role":"pad_sleeve","skin_m":skin,"span_m":span,"radius_m":radius,
                         "contact_geometry":"rounded_convex_sleeve","bearings_internal":True})
        else:
            # Follow the source bone's bow using four contiguous convex pieces.
            # Joint translations, including lateral mutations, define endpoints.
            zvalues=np.unique(np.round(m.vertices[:,2],9))
            centers=[]
            for t in np.linspace(hr/span,1-hr/span,5):
                z=float(t * span);nearest=zvalues[np.argmin(abs(zvalues-z))]
                xy=m.vertices[abs(m.vertices[:,2]-nearest)<1e-7,:2].mean(0)
                centers.append(end*t+np.r_[xy,0.])
            corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.array([.85,.8])*radius
            for i,(c0,c1) in enumerate(zip(centers[:-1],centers[1:])):
                pts=np.vstack([np.c_[corners,np.zeros(4)]+c0,np.c_[corners,np.zeros(4)]+c1])
                body=trimesh.convex.convex_hull(pts)
                emit(link,folder,f"rigid_link_{i}",body,[.35,.43,.48,1]);pieces.append((body,1100.))
            for i,part in enumerate(metal):
                emit(link,folder,["proximal_bearing","distal_bearing"][i],part,[.8,.43,.18,1]);pieces.append((part,2700.))
            rows.append({"link":name,"role":"curved_rigid_link","span_m":span,"radius_m":radius,
                         "centerline_m":np.asarray(centers).tolist(),"bearing_axes":[axis0.tolist(),axis1.tolist()]})
        rows[-1]["mass_kg"]=inertia(link,pieces)
    # Give each added spread hinge a shaft along its actual independent axis.
    for link in root.findall("link"):
        if not re.fullmatch(r"link_[2-5]_mcp_spread",link.get("name")):continue
        joint=incoming[link.get("name")];digit=link.get("name").split('_')[1]
        radius=next(row['radius_m'] for row in rows if row['link']==f'link_{digit}_0')*.65
        shaft=bearing(radius,np.fromstring(joint.find('axis').get('xyz'),sep=' '),[0,0,0])
        emit(link,folder,'spread_shaft',shaft,[.8,.43,.18,1],True)
        inertia(link,[(shaft,2700.)])
    # Keep the established mechanical-family prefix for actuator adapters.
    root.set('name','mechanical_v2_'+VERSION+'_'+original.get('name','hand'))
    ET.indent(tree,space='  ');tree.write(destination,encoding='unicode',xml_declaration=True)
    manifest.update(version=3,contact_model=VERSION,contact_parts=rows,spread_axes=axes,
                    mass_kg=sum(float(l.find('inertial/mass').get('value')) for l in root.findall('link') if l.find('inertial/mass') is not None),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    urdf_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                    caveats=['polymer sleeve is a physical design proxy; material properties uncalibrated',
                             'nonterminal bearing/bar mass uses additive proxy solids',
                             'new contact and joint semantics invalidate old BC/checkpoint compatibility'])
    physical_links={l.get('name'):l for l in root.findall('link')}
    for part in manifest['parts']:
        part['mass_kg']=float(physical_links[part['link']].find('inertial/mass').get('value'))
        part['contact_revision']=next((r for r in rows if r['link']==part['link']),None)
    for row in manifest['joint_mapping']:
        row['axis']=root.find(f"joint[@name='{row['joint']}']/axis").get('xyz')
    destination.with_suffix('.mapping.json').write_text(json.dumps(manifest,indent=2))
    return manifest
