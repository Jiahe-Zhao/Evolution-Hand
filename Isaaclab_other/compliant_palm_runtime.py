"""PhysX volume-deformable palm layer for the installed legacy deformable API.

Call after rigid-hand spawning and before simulation reset. This module requests
physical rigid/deformable attachments; runtime validation is mandatory.
"""
import json
from pathlib import Path
import numpy as np


def spawn_compliant_palm(stage, asset_dir, hand_path='/World/Hand', palm_path='/World/CompliantPalm', root_position=(0., 0., 0.), root_orientation=(1., 0., 0., 0.)):
    from pxr import Gf, Sdf, UsdGeom, UsdShade, UsdPhysics, PhysxSchema, Vt, Usd
    from omni.physx.scripts import deformableUtils, physicsUtils
    import trimesh

    asset_dir = Path(asset_dir)
    manifest = json.loads((asset_dir / 'compliant_palm.json').read_text())
    data = np.load(asset_dir / 'palm_tetrahedra.npz')
    points, tetrahedra = data['points'], data['tetrahedra']
    surface = trimesh.load(asset_dir / 'continuous_palm.stl', force='mesh')
    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(palm_path))
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(surface.vertices.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(surface.faces))
    mesh.CreateFaceVertexIndicesAttr(surface.faces.flatten().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.AddTranslateOp().Set(Gf.Vec3d(*root_position))
    mesh.AddOrientOp().Set(Gf.Quatf(root_orientation[0], Gf.Vec3f(*root_orientation[1:])))
    rest_points = Vt.Vec3fArray.FromNumpy(points.astype(np.float32))
    indices = tetrahedra.flatten().astype(int).tolist()
    success = deformableUtils.add_physx_deformable_body(
        stage, mesh.GetPath(), collision_rest_points=rest_points,
        collision_indices=indices, simulation_rest_points=rest_points,
        simulation_indices=indices, kinematic_enabled=False,
        solver_position_iteration_count=int(manifest.get("solver_position_iterations", 32)), vertex_velocity_damping=0.05,
        self_collision=True, self_collision_filter_distance=0.0008)
    if not success:
        raise RuntimeError('PhysX rejected the tetrahedral palm mesh')
    collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    collision.CreateContactOffsetAttr(0.0005)
    collision.CreateRestOffsetAttr(0.0)
    material_path = Sdf.Path(palm_path + '_Material')
    material = UsdShade.Material.Define(stage, material_path)
    values = manifest['material_design_assumptions']
    deformableUtils.add_deformable_body_material(
        stage, material_path, youngs_modulus=values['youngs_modulus_pa'],
        poissons_ratio=values['poissons_ratio'], density=values['density_kg_m3'],
        dynamic_friction=0.8)
    physicsUtils.add_physics_material_to_prim(stage, mesh.GetPrim(), material_path)
    shader = UsdShade.Shader.Define(stage, material_path.AppendChild('Surface'))
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.06, 0.48, 0.38))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.8)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    from omni.physx import get_physx_cooking_interface
    cooked = get_physx_cooking_interface().cook_deformable_body_mesh(str(mesh.GetPath()))
    print('[PALM_COOK_RESULT]', cooked, flush=True)
    print('[PALM_ATTACHMENT_SCHEMA]', list(PhysxSchema.PhysxPhysicsAttachment.GetSchemaAttributeNames()), flush=True)
    bindings = []
    for link in manifest['attachment_proposal']:
        rigid_path = Sdf.Path(hand_path + '/' + link)
        rigid = stage.GetPrimAtPath(rigid_path)
        if not rigid or not rigid.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f'Attachment target is not an imported rigid body: {rigid_path}')
        # URDF importer authors CollisionAPI on a container; attachments need
        # concrete geometry. Expand instances before authoring leaf colliders.
        for _ in range(3):
            instances = [p for p in Usd.PrimRange(rigid) if p.IsInstance()]
            if not instances:
                break
            for instance in instances:
                instance.SetInstanceable(False)
        colliders = [p for p in Usd.PrimRange(rigid)
                     if p.IsA(UsdGeom.Mesh) and '/collisions/' in str(p.GetPath())]
        if not colliders:
            raise ValueError(f'No collision mesh under {rigid_path}: {[(str(p.GetPath()), p.GetTypeName()) for p in Usd.PrimRange(rigid)]}')
        for p in Usd.PrimRange(rigid):
            if p.HasAPI(UsdPhysics.CollisionAPI) and not p.IsA(UsdGeom.Boundable):
                p.RemoveAPI(UsdPhysics.CollisionAPI)
        for geometry in colliders:
            UsdPhysics.CollisionAPI.Apply(geometry)
            UsdPhysics.MeshCollisionAPI.Apply(geometry).CreateApproximationAttr('convexHull')
            shape_properties = PhysxSchema.PhysxCollisionAPI.Apply(geometry)
            shape_properties.CreateContactOffsetAttr(0.003)
            shape_properties.CreateRestOffsetAttr(0.0)
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], False, True)
        collider = max(colliders, key=lambda p: max(bounds.ComputeWorldBound(p).ComputeAlignedBox().GetSize()))
        rigid_box = bounds.ComputeWorldBound(collider).ComputeAlignedBox()
        soft_box = bounds.ComputeWorldBound(mesh.GetPrim()).ComputeAlignedBox()
        if rigid_box.IsEmpty():
            raise ValueError(f'Collision mesh has empty bounds: {collider.GetPath()}')
        print('[PALM_OVERLAP_BOUNDS]', str(collider.GetPath()), collider.GetTypeName(), list(rigid_box.GetMin()), list(rigid_box.GetMax()), 'soft', list(soft_box.GetMin()), list(soft_box.GetMax()), flush=True)
        attachment_path = mesh.GetPath().AppendChild('Attachment_' + link)
        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
        attachment.CreateActor0Rel().SetTargets([mesh.GetPath()])
        attachment.CreateActor1Rel().SetTargets([collider.GetPath()])
        # Match the installed official example after aligning root poses.
        auto_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
        print('[PALM_AUTO_PARAMETERS]', {name: attachment.GetPrim().GetAttribute(name).Get() for name in auto_api.GetSchemaAttributeNames()}, flush=True)
        from omni.physx.bindings import _physx
        attachment_interface = _physx.acquire_physx_attachment_interface()
        computed = attachment_interface.compute_attachment_points(str(attachment_path))
        generated_points = attachment.GetPoints0Attr().Get()
        point_count = len(generated_points) if generated_points is not None else 0
        print('[PALM_ATTACHMENT_COMPUTED]', link, computed, point_count, flush=True)
        if not computed or point_count == 0:
            raise RuntimeError(f'No physical attachment points generated for {link}')
        computed_count = point_count
        if manifest.get('attachment_mode') == 'support_centerline':
            # Keep paired coordinates produced by PhysX, but bond only the
            # designed lower support line instead of its whole overlap volume.
            reference = points[manifest['attachment_proposal'][link]['node_indices']]
            generated = np.asarray(generated_points, dtype=np.float64)
            paired = np.asarray(attachment.GetPoints1Attr().Get(),dtype=np.float64)
            # Infer the rigid coordinate transform from PhysX's paired points;
            # imported mesh/body frames differ and must not be guessed.
            center0, center1 = generated.mean(0), paired.mean(0)
            u, _, vt = np.linalg.svd((generated-center0).T@(paired-center1))
            rotation = u@vt
            if np.linalg.det(rotation) < 0:
                u[:,-1] *= -1
                rotation = u@vt
            offset = center1-center0@rotation
            residual = np.linalg.norm(generated@rotation+offset-paired,axis=-1).max()
            if residual > 1e-5:
                raise ValueError(f'Attachment coordinate pairs are not a rigid transform: {link}, residual={residual}')
            attachment.GetPrim().RemoveAPI(PhysxSchema.PhysxAutoAttachmentAPI)
            attachment.CreatePoints0Attr(Vt.Vec3fArray.FromNumpy(reference.astype(np.float32)))
            attachment.CreatePoints1Attr(Vt.Vec3fArray.FromNumpy((reference@rotation+offset).astype(np.float32)))
            point_count = len(reference)
            print('[PALM_SUPPORT_LINE]',link,point_count,flush=True)
        # Bonded support meshes occupy the elastomer volume by design. Their
        # unilateral contacts oppose the attachment constraints; filter only
        # these embedded constituents, preserving all external palm contacts.
        filtered = UsdPhysics.FilteredPairsAPI.Apply(mesh.GetPrim())
        relation = filtered.CreateFilteredPairsRel()
        for geometry in colliders:
            relation.AddTarget(geometry.GetPath())
        bindings.append({'attachment': str(attachment_path), 'rigid_body': str(rigid_path), 'collider': str(collider.GetPath()), 'mode': 'explicitly_computed_native_attachment', 'generated_point_count': point_count, 'original_overlap_point_count': computed_count, 'attachment_mode': manifest.get('attachment_mode','overlap_volume')})
    return {'mesh_path': str(mesh.GetPath()), 'attachments': bindings,
            'requested_solver': 'PhysX tetrahedral FEM with native rigid attachments',
            'runtime_verified': False}
