
import bpy
import bmesh
import mathutils
import math

# https://devtalk.blender.org/t/understanding-matrix-operations-in-blender/10148/2

from typing import NamedTuple

from numpy.random import default_rng
import numpy as np


def select_activate_only(objects=[]):
    for obj in bpy.data.objects:
        obj.select_set(False)
    bpy.context.view_layer.objects.active = None 
    for obj in objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

def apply_loc_rot_scale(obj, location=False, rotation=False, scale=False):
    select_activate_only([obj])
    bpy.ops.object.transform_apply(location=location, rotation=rotation, scale=scale)


def create_icosphere(radius=1.0, subdivisions=1, name="ico_sphere", collection_name=None):
    bm = bmesh.new()
    # Create icosphere.
    # https://docs.blender.org/api/current/bmesh.ops.html#bmesh.ops.create_icosphere
    bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=radius, matrix=mathutils.Matrix.Identity(4), calc_uvs=False)
    object_mesh = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(object_mesh)
    obj = bpy.data.objects.new(name + "_obj", object_mesh)
    if collection_name == None:
        bpy.context.collection.objects.link(obj)
    else:
        create_collection_if_not_exists(collection_name)
        bpy.data.collections[collection_name].objects.link(obj)
    bm.free()
    return obj

# https://stackoverflow.com/questions/19045971/random-rounding-to-integer-in-python
def probabilistic_round(x):
    return int(math.floor(x + mathutils.noise.random()))

# https://github.com/blender/blender/blob/master/source/blender/nodes/geometry/nodes/node_geo_distribute_points_on_faces.cc
# base_obj - MUST BE TRIANGULATED!
# returns: list of touples: (p, N, w)
def mesh_uniform_weighted_sampling(base_obj, n_samples, base_density=5.0, use_weight_paint=False):
    rng = default_rng()
    samples = [] # (p, N, w)
    samples_all = []
    n_polygons = len(base_obj.data.polygons)
    samples_density = math.ceil(n_samples / n_polygons) + base_density
    for polygon in base_obj.data.polygons: # must be triangulated mesh!
        # Extract triangle vertices and their weights.
        triangle_vertices = []
        triangle_vertex_weights = []
        for v_idx in polygon.vertices:
            v = base_obj.data.vertices[v_idx]
            triangle_vertices.append(v.co)
            if len(v.groups) < 1:
                triangle_vertex_weights.append(0.0)
            else:
                triangle_vertex_weights.append(v.groups[0].weight) # TODO: only one group? Investigate! float in [0, 1], default 0.0
        # Create samples.
        polygon_density = 1
        if use_weight_paint:
            polygon_density = (triangle_vertex_weights[0] + triangle_vertex_weights[1] + triangle_vertex_weights[2]) / 3.0
        point_amount = probabilistic_round(polygon.area * polygon_density + samples_density)
        for i in range(point_amount):
            a = mathutils.noise.random()
            b = mathutils.noise.random()
            c = mathutils.noise.random()
            s = a + b + c
            un = (a / s)
            vn = (b / s)
            wn = (c / s)
            p = un * triangle_vertices[0] + vn * triangle_vertices[1] + wn * triangle_vertices[2]
            w = un * triangle_vertex_weights[0] + vn * triangle_vertex_weights[1] + wn * triangle_vertex_weights[2] # interpolate weight
            n = polygon.normal # TODO: vertex normals?
            samples_all.append([p,n,w])
    print("Number of all samples:", len(samples_all), ". Number of desired samples:", n_samples)
    if len(samples_all) > n_samples:
        random_sample_indices = rng.integers(len(samples_all), size=n_samples)
        for i in random_sample_indices:
            samples.append(samples_all[i])
    else:
        print("Number of all samples is smaller than desired number of samples! Using only number of found samples")
        samples = samples_all
    return samples

# https://blender.stackexchange.com/questions/220072/check-using-name-if-a-collection-exists-in-blend-is-linked-to-scene
def create_collection_if_not_exists(collection_name):
    if collection_name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(new_collection) #Creates a new collection

def transform_obj_rot_pos(base_obj, pos_vec=mathutils.Vector((0,0,0)), rot_vec=mathutils.Vector((0,0,0))):
    # Extract position and scale.
    #original_scale_vec = base_obj.matrix_world.to_scale()
    #original_mat_scale = mathutils.Matrix.Scale(0.5, 4, (0.0, 0.0, 1.0))
    #original_pos_vec = base_obj.matrix_world.to_translation()
    #original_mat_trans = mathutils.Matrix.Translation(original_pos_vec)
    # 
    # zero out curr rotation matrix first: https://blender.stackexchange.com/a/159992
    curr_rot_mat_inv = base_obj.matrix_basis.to_3x3().transposed().to_4x4()
    base_obj.matrix_basis = base_obj.matrix_basis @ curr_rot_mat_inv
    # Orient object using vector.
    new_rot_z = rot_vec.normalized()
    new_rot_x, new_rot_y = pixar_onb(new_rot_z)
    rot_basis = mathutils.Matrix((new_rot_x, new_rot_y, new_rot_z))
    rot_basis = rot_basis.transposed()
    rot_basis.resize_4x4()
    rot_mat = rot_basis.to_euler().to_matrix().to_4x4() # extract only rotation!
    base_obj.matrix_basis = base_obj.matrix_basis @ rot_mat # https://blender.stackexchange.com/questions/35125/what-is-matrix-basis
    # Transform object using vector.
    new_pos = pos_vec
    trans_mat = mathutils.Matrix.Translation(new_pos)
    base_obj.matrix_basis = trans_mat @ base_obj.matrix_basis


def create_instance(base_obj, pos_vec=mathutils.Vector((0,0,0)), rot_vec=mathutils.Vector((0,0,0)), collection_name=None):
    # Create instance.
    inst_obj = bpy.data.objects.new(base_obj.name+"_inst", base_obj.data)
    # Transform.
    transform_obj_rot_pos(inst_obj, pos_vec, rot_vec)
    # Store.
    if collection_name == None:
        bpy.context.collection.objects.link(inst_obj)
    else:
        create_collection_if_not_exists(collection_name)
        bpy.data.collections[collection_name].objects.link(inst_obj)
    return inst_obj


class Flyer:
    def __init__(self, _obj, _flying):
        self.obj = _obj # bpy.types.Object
        self.flying = _flying # bool
        self.fly_prob = mathutils.noise.random()

    def transform(self, rot_vec, pos_vec):
        transform_obj_rot_pos(self.obj, pos_vec, rot_vec)

class DandelionHead:
    def __init__(self, location=mathutils.Vector((0,0,0)), location_deviation=10):
        self.obj = create_icosphere(radius=3.0, subdivisions=3, name="dandelion_head", collection_name="code_instance")
        self.ini_pos = mathutils.Vector(location + mathutils.Vector((mathutils.noise.random(), mathutils.noise.random(), mathutils.noise.random())) * location_deviation)
        self.obj.location = self.ini_pos
        apply_loc_rot_scale(obj=self.obj, location=True, rotation=False, scale=False) # location is applied to initial position, the object has local location (0,0,0)
        self.max_deviation_from_ini_pos = 10.0

# https://graphics.pixar.com/library/OrthonormalB/paper.pdf
# NOTE: n must be normalized!
def pixar_onb(_n):
    n = _n.normalized()
    t = mathutils.Vector((0,0,0))
    b = mathutils.Vector((0,0,0))
    if(n[2] < 0.0):
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        t = mathutils.Vector((1.0 - n[0] * n[0] * a, -b, n[0]))
        b = mathutils.Vector((b, n[1] * n[1] * a - 1.0, -n[1]))
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        t = mathutils.Vector((1.0 - n[0] * n[0] * a, b, -n[0]))
        b = mathutils.Vector((b, 1 - n[1] * n[1] * a, -n[1]))
    return t, b

# This code is valid when object is moved from its local position and rotation to world for the first time.
# Later axis change and rotation become unpredictable!
# Rotation matrix must be nullified before applying new rotation matrix! See how it is done in: transform_obj_rot_pos()
def sanity_check():
    t = bpy.data.objects["test"]
    test_obj = bpy.data.objects.new(name="test obj", object_data=t.data)
    bpy.context.view_layer.active_layer_collection.collection.objects.link(test_obj)
    print("ORIGINAL BASIS, LOCAL, WORLD MATRICES:")
    print(test_obj.matrix_basis)
    print(test_obj.matrix_local)
    print(test_obj.matrix_world)
    # Orient object using vector.
    new_rot_z = mathutils.Vector((1,0,-1))
    new_rot_x, new_rot_y = pixar_onb(new_rot_z)
    rot_basis = mathutils.Matrix((new_rot_x, new_rot_y, new_rot_z))
    rot_basis = rot_basis.transposed()
    rot_basis.resize_4x4()
    rot_mat = rot_basis.to_euler().to_matrix().to_4x4() # extract only rotation!
    test_obj.matrix_world = rot_mat @ test_obj.matrix_world # https://blender.stackexchange.com/questions/35125/what-is-matrix-basis
    # Transform object using vector.
    new_pos = mathutils.Vector((3,2,1))
    trans_mat = mathutils.Matrix.Translation(new_pos)
    test_obj.matrix_world = trans_mat @ test_obj.matrix_world

# https://blenderscripting.blogspot.com/2011/05/blender-25-python-bezier-from-list-of.html
def create_curve(name="curve", points=[], collection_name=None):
    # Create curve data.
    curve_data = bpy.data.curves.new(name=name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = 0.3
    # Populate curve data.
    poly_curve = curve_data.splines.new("POLY")
    poly_curve.points.add(len(points)-1)
    for pi in range(len(points)):
        poly_curve.points[pi].co = (points[pi][0], points[pi][1], points[pi][2], 1)
    # Create curve object from curve data and add it to the scene.
    curve_object = bpy.data.objects.new(name=name+"_obj", object_data=curve_data)
    if collection_name == None:
        bpy.context.collection.objects.link(curve_object)
    else:
        create_collection_if_not_exists(collection_name)
        bpy.data.collections[collection_name].objects.link(curve_object)
    return curve_object

def create_curve_from_two_points(name="curve", starting_point=mathutils.Vector((0,0,0)), ending_point=mathutils.Vector((0,0,0)), n_subdivisions=0, collection_name=None):
    points = []
    t = 0
    dt = 1.0
    if n_subdivisions > 0:
        dt = 1.0 / (n_subdivisions+1)
    for subdiv_i in range(n_subdivisions+2):
        pi = (1.0 - t) * ending_point + t * starting_point
        points.append(mathutils.Vector(pi))
        t = t + dt
    curve_object = create_curve(name=name, points=points, collection_name=collection_name)
    return curve_object

def create_randomized_curve(name="wind_curve", starting_point=mathutils.Vector((-300,0,0)), ending_point=mathutils.Vector((300,0,0)), n_subdivisions=7, collection_name="wind_path"):
    curve_obj = create_curve_from_two_points(name=name, starting_point=starting_point, ending_point=ending_point, n_subdivisions=n_subdivisions, collection_name=collection_name)
    randomization_factor = mathutils.noise.random() * 10.0
    for point in curve_obj.data.splines[0].points:
        point_co = mathutils.Vector((point.co[0], point.co[1], point.co[2]))
        trans_vec = mathutils.noise.turbulence_vector(
            point_co * randomization_factor, 
            3, # octaves
            False, #hard
            noise_basis='PERLIN_ORIGINAL', 
            amplitude_scale=10.0, 
            frequency_scale=10.0) * 1.0
        new_point_co = point_co + trans_vec
        point.co = (new_point_co[0], new_point_co[1], new_point_co[2], 0.0)
    return curve_obj

# https://b3d.interplanety.org/en/how-to-apply-transformations-to-a-mesh-with-the-blender-python-api/
def apply_transformation_to_curve_spline_points(curve_obj):
    curve_obj_mat = curve_obj.matrix_basis.copy()
    for point in curve_obj.data.splines[0].points:
        point_co = mathutils.Vector((point.co[0], point.co[1], point.co[2]))
        new_point_co = curve_obj_mat @ point_co
        point.co = (new_point_co[0], new_point_co[1], new_point_co[2], 0.0)
    curve_obj.matrix_world.identity()

def test_noise_field():
    n_curves = 25
    # Create curve lines
    curves = []
    for i in range(n_curves):
        for j in range(n_curves):
            curve_obj = create_curve_from_two_points(name="curve", starting_point=mathutils.Vector((-300,0,0)), ending_point=mathutils.Vector((300,0,0)), n_subdivisions=7, collection_name="code_instance")
            curve_obj.location = mathutils.Vector((0,i*20,j*20))
            apply_transformation_to_curve_spline_points(curve_obj)
            curves.append(curve_obj)
    for curve in curves:
        for point in curve.data.splines[0].points:
            point_co = mathutils.Vector((point.co[0], point.co[1], point.co[2]))
            trans_vec = mathutils.noise.turbulence_vector(
                point_co, 
                3, # octaves
                False, #hard
                noise_basis='PERLIN_ORIGINAL', 
                amplitude_scale=10.0, 
                frequency_scale=15.0) * 5.0
            new_point_co = point_co + trans_vec
            point.co = (new_point_co[0], new_point_co[1], new_point_co[2], 0.0)

# remap(x): [m,n] -> [k,l]
def remap(x, m, n, k, l):
    a = (l - k) / (n - m)
    b = k - a * m
    return a * x + b

# nexp(x): [0, n] -> [0,1]
def nexp(x, n):
    return (1.0 / (math.exp(n) - 1.0)) * (math.exp(x) - 1.0)

def main():

    # Animation information.
    total_frames = 500
    delta_keypoint_frame = 20

    # Scene parameters.
    # Ground.
    ground_obj = bpy.data.collections["ground"].all_objects[0]
    ground_bm = bmesh.new()
    ground_bm.from_mesh(ground_obj.data)
    groud_bvh = mathutils.bvhtree.BVHTree.FromBMesh(ground_bm)
    # Dandelion centers.
    dandelion_head_positions = []
    for in_dandelion_head_positions in bpy.data.collections["dandelion_head_positions"].all_objects:
        dandelion_head_positions.append(in_dandelion_head_positions.location)
    # Dandelion fliers.
    flyer_obj = bpy.data.collections["petals_light"].all_objects[0]
    n_flyers = 200
    
    # Initialize animation objects.
    dandelion_i = 0
    for dandelion_head_position in dandelion_head_positions:
        dandelion_i += 1
        # Create dandelion flyers head.
        dandelion_head = DandelionHead(location=dandelion_head_position, location_deviation=10.0)
        dandelion_head.obj.keyframe_insert(data_path="location", frame=0)
        dandelion_head.obj.keyframe_insert(data_path="rotation_euler", frame=0)
        # Create dandelion stem by connection dandelion head and closest point on ground.
        nearest = groud_bvh.find_nearest(dandelion_head.ini_pos)
        dandelion_stem = create_curve_from_two_points(name="dandelion_stem_"+str(dandelion_i), starting_point=nearest[0], ending_point=dandelion_head.ini_pos, n_subdivisions=2+2*dandelion_i, collection_name="code_instance")
        dandelion_stem_vector = mathutils.Vector(dandelion_head.obj.location - nearest[0])
        dandelion_stem_dir = dandelion_stem_vector.normalized()
        dandelion_stem_length = dandelion_stem.data.splines[0].calc_length() # we only have one spline in the curve! See `create_curve_from_two_points()`. Equivalent: mathutils.Vector(dandelion_stem.data.splines[0].points[0].co-dandelion_stem.data.splines[0].points[2].co).length
        dandelion_stem_n_points = dandelion_stem.data.splines[0].point_count_u # dandelion_stem.data.splines[0].point_count_v
        dandelion_stem_root_point = dandelion_stem.data.splines[0].points[dandelion_stem_n_points-1]
        dandelion_spline_point_max_weight = math.exp(dandelion_stem_n_points)
        for spline_point in dandelion_stem.data.splines[0].points:
            spline_point.keyframe_insert(data_path="co", frame=0)
        dandelion_stem.keyframe_insert(data_path="location", frame=0) 
        dandelion_stem.keyframe_insert(data_path="rotation_euler", frame=0)
        # Initialize fliers.
        flyers = []
        samples = mesh_uniform_weighted_sampling(dandelion_head.obj, n_flyers, False)
        for sample in samples: # (p, N, w)
            flyer_inst_obj = create_instance(base_obj=flyer_obj, pos_vec=sample[0], rot_vec=sample[1], collection_name="code_instance")
            flyer_inst_obj.keyframe_insert(data_path="location", frame=0)
            flyer_inst_obj.keyframe_insert(data_path="rotation_euler", frame=0)
            flyers.append(Flyer(flyer_inst_obj, False))   

        # Prepare wind.
        # Wind is a global property.
        # Its direction is defined for points in space using noise or given by user.
        wind_vectors = []
        # Generate using noise.
        wind_curve_obj = create_randomized_curve(name="wind_curve", starting_point=mathutils.Vector((-300,0, 300)), ending_point=mathutils.Vector((300,0,0.00)), n_subdivisions=7, collection_name="wind_path")
        # OR Read as curve.
        #wind_curve_obj = bpy.data.collections["wind_path"].all_objects[0]
        for wind_curve_point_i in range(wind_curve_obj.data.splines[0].point_count_u-1):
            wp1 = wind_curve_obj.data.splines[0].points[wind_curve_point_i].co
            wp2 = wind_curve_obj.data.splines[0].points[wind_curve_point_i+1].co
            vec1 = mathutils.Vector((wp1[0], wp1[1], wp1[2]))
            vec2 = mathutils.Vector((wp2[0], wp2[1], wp2[2]))
            wind_vectors.append(mathutils.Vector(vec2-vec1))
        
        # Animate.
        fly_prob = 0.0
        wind_vec_idx = 0
        keypoint_frame = 0
        # In this animation, we have in total `total_frames` frames.
        # Each `delta_keypoint_frame` we calculate new state (i.e., position using nose) and keyframe it.
        while(True):
            # Terminate animation if total number of frames is reached; also add last keypoints.
            if (keypoint_frame > total_frames):
                flyers[i].obj.keyframe_insert(data_path="location", frame=keypoint_frame)
                flyers[i].obj.keyframe_insert(data_path="rotation_euler", frame=keypoint_frame)
                break
            #
            # Set the wind: animation force.
            #
            # Wind.
            if (keypoint_frame % 50 == 0):
                wind_vec_idx = (wind_vec_idx + 1) % len(wind_vectors)
            wind_vec = wind_vectors[wind_vec_idx]
            wind_dir = wind_vec.normalized()
            wind_strength_min = 30.0
            wind_strength_max = 50.0
            #
            # Animate dandelion head and stem.
            #
            # TODO: use wind to move dandelion head
            dandelion_turbulence_strenght_min = 7.0
            dandelion_turbulence_strenght_max = 10.0
            dandelion_turbulence_dir = wind_dir + mathutils.noise.turbulence_vector(dandelion_head.obj.location * 5.0, 3, False, noise_basis='PERLIN_ORIGINAL', amplitude_scale=0.5, frequency_scale=2.0)
            dandelion_turbulence_dir[2] = 0.0 # turbulence only moves on plane orthogonal to dandelion stem. TODO use `dandelion_stem_dir`
            # Add location constraints to movement
            if dandelion_head.obj.location.length > dandelion_head.max_deviation_from_ini_pos: # remember that location is applied to initial position so check only the current location.
                dandelion_turbulence_dir = -dandelion_head.obj.location.normalized()
            dandelion_turbulence_strenght_fact = mathutils.noise.random()
            dandelion_turbulence_strenght = (1.0 - dandelion_turbulence_strenght_fact) * dandelion_turbulence_strenght_min + dandelion_turbulence_strenght_fact * dandelion_turbulence_strenght_max
            current_dandelion_spline_point_idx = 1
            for curr_spline_point in dandelion_stem.data.splines[0].points: 
                # NOTE: see how points are added in `create_curve_from_two_points()` - the first one is closest to dandelion head.
                # As we move away from the first sppline point, the wind influence is less and less.
                curr_stem_point_weight = 1.0 - math.exp(current_dandelion_spline_point_idx) / dandelion_spline_point_max_weight  # maximal weight is at stem point closest to the head since wind will cause the most movement there.
                curr_stem_point_trans = mathutils.Vector(dandelion_turbulence_dir * dandelion_turbulence_strenght) * curr_stem_point_weight
                if current_dandelion_spline_point_idx == 1:  # Dandelion head moves as point on stem closest to head moves.
                    if (keypoint_frame % delta_keypoint_frame == 0):
                        dandelion_head.obj.location += curr_stem_point_trans
                        dandelion_head.obj.keyframe_insert(data_path="location", frame=keypoint_frame)
                        dandelion_head.obj.keyframe_insert(data_path="rotation_euler", frame=keypoint_frame)
                        # All the fliers that are not fyling must move with the head!
                        for i in range(len(samples)):
                            if not flyers[i].flying:
                                flyers[i].obj.location += curr_stem_point_trans
                curr_spline_point.co[0] += curr_stem_point_trans[0]
                curr_spline_point.co[1] += curr_stem_point_trans[1]
                curr_spline_point.co[2] += curr_stem_point_trans[2]
                if (keypoint_frame % delta_keypoint_frame == 0):
                    curr_spline_point.keyframe_insert(data_path="co", frame=keypoint_frame)
                current_dandelion_spline_point_idx = current_dandelion_spline_point_idx + 1
            #
            # Animate dandelion fliers.
            #
            # Calculte probability for takeoff.
            frame_remap_min = 0.0
            frame_remap_max = 1.0
            frame_remapped = remap(keypoint_frame, 0, total_frames, frame_remap_min, frame_remap_max)
            fly_prob = nexp(frame_remapped, frame_remap_max)
            # fly_prob = 1.0 - fly_prob # inverse takeoff probability.
            # For each object, calculate if it takes off and how much it moves. Keyframe it every `delta_keypoint_frame`
            for i in range(len(samples)):
                if not flyers[i].flying:
                    if flyers[i].fly_prob < fly_prob:
                        flyers[i].flying = True
                else:
                    # Small turbulence claculated individually for each flier.
                    flier_turbulence_dir = mathutils.noise.turbulence_vector(flyers[i].obj.location * i * 5.0, 3, False, noise_basis='PERLIN_ORIGINAL', amplitude_scale=0.5, frequency_scale=2.0) #+ mathutils.Vector((1, 1, 1)).normalized()
                    flier_turbulence_strength_fact = mathutils.noise.random()
                    flier_turbulence_strength = (1.0 - flier_turbulence_strength_fact) * 10.0 + flier_turbulence_strength_fact * 20.0
                    # Use wind direction to new petals transform.
                    wind_strength_fact = mathutils.noise.random()
                    wind_strength = (1.0 - wind_strength_fact) * wind_strength_min + wind_strength_fact * wind_strength_max
                    rot_vec = wind_dir + flier_turbulence_dir # TODO: other wind and turbulence
                    trans_vec = wind_dir * wind_strength + flier_turbulence_dir * flier_turbulence_strength
                    flyers[i].transform(rot_vec, trans_vec)
                # Keyframe current state for current flier.
                if (keypoint_frame % delta_keypoint_frame == 0):
                    flyers[i].obj.keyframe_insert(data_path="location", frame=keypoint_frame) 
                    flyers[i].obj.keyframe_insert(data_path="rotation_euler", frame=keypoint_frame) # https://blenderartists.org/t/animating-rotation-using-keyframes-in-python/590243/2
            
            # Move to next keypoint frame.
            keypoint_frame += delta_keypoint_frame
        
        ground_bm.free()
        
main()
