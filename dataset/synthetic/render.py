import bpy
import math
import random
import os
from mathutils import Vector, Matrix
import shutil
from typing import Tuple, Dict, Literal
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
model_dirs = []

for i in range(0, 150):
    folder_name = f"000-{i:03d}"
    path = f'/path/objs/{folder_name}'
    model_dirs.append(path)

output_dir = r'/path/data/synthetic'
root_dir = r'/path/materials'
hdri_dir = r'/path/hdri/hdri_files'


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


clear_console()


def select_all_mesh_objects():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.select_set(True)


def get_selected_objects_with_geometry():
    return [obj for obj in bpy.context.selected_objects if obj.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}]


def add_uv_mapping(obj):
    if obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode='OBJECT')
    else:
        print(f"Object {obj.name} is not a mesh and cannot be UV unwrapped.")


def detect_pbr_textures(folder_path):
    """
    Detect and return a dictionary of PBR texture maps from the material folder.
    """
    texture_dict = {}

    for file_name in os.listdir(folder_path):
        if "_Color" in file_name:
            texture_dict['Color'] = os.path.join(folder_path, file_name)
        elif "_NormalGL" in file_name:
            texture_dict['NormalGL'] = os.path.join(folder_path, file_name)
        elif "_Roughness" in file_name:
            texture_dict['Roughness'] = os.path.join(folder_path, file_name)
        elif "_Displacement" in file_name:
            texture_dict['Displacement'] = os.path.join(folder_path, file_name)
        elif "_Metalness" in file_name:
            texture_dict['Metalness'] = os.path.join(folder_path, file_name)
            print(f"Detected Metalness texture: {file_name}")

    if 'Metalness' not in texture_dict:
        print(f"Metalness texture not detected in folder: {folder_path}")

    return texture_dict


def create_pbr_material(pbr_textures):
    mat_id = len(bpy.data.materials)
    material = bpy.data.materials.new(name=f"PBR_Material_{mat_id:03d}")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)

    print(f"Creating nodes: {material.name}")

    for idx, (texture_type, path) in enumerate(pbr_textures.items()):
        if path == 'Empty':
            continue

        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-400, -200 * idx)
        tex_image.image = bpy.data.images.load(path)
        tex_image.image.name = f"{path.split('/')[-1]}.{mat_id:03d}"
        if texture_type != 'Color':
            tex_image.image.colorspace_settings.is_data = True

        print(f"Loading and creating {texture_type} texture: {path}")

        if texture_type == 'Color':
            material.node_tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
            print(f"Linking color texture to Base Color.")
        elif texture_type == 'NormalGL':
            normal_map = nodes.new(type='ShaderNodeNormalMap')
            normal_map.location = (-200, -200 * idx)
            material.node_tree.links.new(tex_image.outputs['Color'], normal_map.inputs['Color'])
            material.node_tree.links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
            print(f"Linking normal map to Normal input.")
        elif texture_type == 'Roughness':
            material.node_tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Roughness'])
            print(f"Linking roughness texture to Roughness input.")
        elif texture_type == 'Displacement':
            displacement = nodes.new(type='ShaderNodeDisplacement')
            displacement.location = (-200, -200 * idx)
            material.node_tree.links.new(tex_image.outputs['Color'], displacement.inputs['Height'])
            output = nodes.get('Material Output') or nodes.new(type='ShaderNodeOutputMaterial')
            output.location = (200, 0)
            material.node_tree.links.new(displacement.outputs['Displacement'], output.inputs['Displacement'])
            print(f"Linking displacement texture to Displacement input.")
        elif texture_type == 'Metalness':
            material.node_tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Metallic'])
            print(f"Linking metalness texture to Metallic input.")

    output = nodes.get('Material Output') or nodes.new(type='ShaderNodeOutputMaterial')
    if not output.inputs['Surface'].is_linked:
        material.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        print(f"Linking BSDF to the material output Surface.")

    print(f"Creating PBR material: {material.name}")
    return material


texture_types = ['Color', 'NormalGL', 'Roughness', 'Displacement', 'Metalness']
pbr_textures_list = []

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    if os.path.isdir(folder_path):
        texture_dict = {}
        for texture_type in texture_types:
            texture_file = f"{folder}_{texture_type}.jpg"
            texture_path = os.path.join(folder_path, texture_file)
            if os.path.exists(texture_path):
                texture_dict[texture_type] = texture_path
            else:
                texture_dict[texture_type] = 'Empty'
        pbr_textures_list.append(texture_dict)


def scene_bbox(objects, ignore_matrix=False):
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in objects:
        if hasattr(obj.data, 'vertices'):
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                if not ignore_matrix:
                    coord = obj.matrix_world @ coord
                bbox_min = Vector((min(bbox_min[i], coord[i]) for i in range(3)))
                bbox_max = Vector((max(bbox_max[i], coord[i]) for i in range(3)))
    if not found:
        raise RuntimeError("No objects with geometry in scene to compute bounding box for")
    return bbox_min, bbox_max


def remove_armatures_and_animations():
    delete_objects_by_type('ARMATURE')
    bpy.context.view_layer.objects.active = None
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)


def delete_objects_by_type(obj_type):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type=obj_type)
    bpy.ops.object.delete()


def merge_meshes(selected_objects):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in selected_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = selected_objects[0]
    bpy.ops.object.join()
    return bpy.context.active_object


def normalize_scene():
    selected_objects = list(get_selected_objects_with_geometry())
    if not selected_objects:
        raise RuntimeError("No objects selected for normalization.")
    if len(selected_objects) > 1:
        merged_object = merge_meshes(selected_objects)
        selected_objects = [merged_object]
    try:
        bbox_min, bbox_max = scene_bbox(selected_objects)
    except RuntimeError as e:
        print(f"Error in computing bounding box: {e}")
        return
    size = bbox_max - bbox_min
    max_size = max(size)
    if max_size == 0:
        raise RuntimeError("Bounding box has zero size, cannot normalize scene.")
    scale = 1 / max_size
    for obj in selected_objects:
        obj.scale = obj.scale * scale
        obj.matrix_world = obj.matrix_world @ Matrix.Scale(scale, 4)
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bbox_min, bbox_max = scene_bbox(selected_objects)
    offset = -(bbox_min + bbox_max) / 2
    for obj in selected_objects:
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    if "Camera" in bpy.data.objects:
        bpy.data.objects["Camera"].parent = None
    print(f"Object positions and scales after normalization: {[obj.matrix_world for obj in selected_objects]}")
    return selected_objects


def reset_cameras() -> None:
    delete_objects_by_type('CAMERA')
    bpy.ops.object.camera_add()
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"
    bpy.context.scene.camera = new_camera


def setup_cameras() -> Dict[str, bpy.types.Object]:
    def create_camera(name: str, location: Tuple[float, float, float], scale: float) -> bpy.types.Object:
        scene = bpy.context.scene
        sensor_width_in_mm = 36
        resolution_x_in_px = 1920
        resolution_y_in_px = 1080

        scene.render.resolution_x = int(resolution_x_in_px / scale)
        scene.render.resolution_y = int(resolution_y_in_px / scale)
        scene.render.resolution_percentage = int(scale * 100)

        bpy.ops.object.camera_add(location=location)
        cam_object = bpy.context.object
        cam = cam_object.data

        cam.name = name
        cam.type = 'PERSP'
        cam.lens = 50
        cam.lens_unit = 'MILLIMETERS'
        cam.sensor_width = sensor_width_in_mm

        bpy.ops.object.constraint_add(type='TRACK_TO')
        constraint = cam_object.constraints[-1]
        constraint.target = bpy.data.objects.new("Empty", None)
        bpy.context.collection.objects.link(constraint.target)
        constraint.target.location = Vector((0, 0, 0))
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'

        return cam_object

    delete_objects_by_type('CAMERA')

    nRows = 9
    nCols = 14
    latitude_angles = [5, 15, 25, 35, 45, 55, 65, 75, 95]
    radius = 3

    camera_positions = []

    for j in range(nRows):
        latitude = latitude_angles[j] / 180.0 * math.pi
        for i in range(nCols):
            longitude = (i * 360.0 / nCols - 180.0) / 180.0 * math.pi
            x = radius * math.sin(latitude) * math.cos(longitude)
            y = radius * math.sin(latitude) * math.sin(longitude)
            z = radius * math.cos(latitude)

            T_location = Vector((x, y, z))
            camera_positions.append(T_location)

    selected_positions = random.sample(camera_positions, 8)

    cameras = {}
    for i, location in enumerate(selected_positions):
        name = f"Camera_{i + 1}"
        cameras[name] = create_camera(name, location, 1)

    return cameras


def setup_world_lighting(hdri_path):
    if not bpy.context.scene.world:
        new_world = bpy.data.worlds.new("NewWorld")
        bpy.context.scene.world = new_world

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    nodes.clear()

    background = nodes.new(type='ShaderNodeBackground')
    background.location = 0, 0

    env_texture = nodes.new(type='ShaderNodeTexEnvironment')
    env_texture.location = -400, 0
    env_texture.image = bpy.data.images.load(hdri_path)

    output = nodes.new(type='ShaderNodeOutputWorld')
    output.location = 200, 0

    background.inputs['Strength'].default_value = 2.0

    links.new(env_texture.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])

    print(f"Environment lighting set to: {hdri_path}")


def _create_light(
        name: str,
        light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
        location: Tuple[float, float, float],
        rotation: Tuple[float, float, float],
        energy: float,
        use_shadow: bool = False,
        specular_factor: float = 1.0,
) -> bpy.types.Object:
    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def setup_lighting() -> Dict[str, bpy.types.Object]:
    delete_objects_by_type('LIGHT')

    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(6, 6, 10),
        rotation=(math.radians(45), 0, math.radians(45)),
        energy=5,
        use_shadow=False,
    )

    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(-6, 6, 10),
        rotation=(math.radians(45), 0, math.radians(-45)),
        energy=3,
        use_shadow=False,
    )

    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, -6, 10),
        rotation=(math.radians(-45), 0, 0),
        energy=2,
        use_shadow=False,
    )

    extra_fill_light_1 = _create_light(
        name="Extra_Fill_Light_1",
        light_type="POINT",
        location=(0, 0, 5),
        rotation=(0, 0, 0),
        energy=1,
        use_shadow=False,
    )

    extra_fill_light_2 = _create_light(
        name="Extra_Fill_Light_2",
        light_type="POINT",
        location=(5, 0, 5),
        rotation=(0, 0, 0),
        energy=1,
        use_shadow=False,
    )

    return {
        "key_light": key_light,
        "fill_light": fill_light,
        "rim_light": rim_light,
        "extra_fill_light_1": extra_fill_light_1,
        "extra_fill_light_2": extra_fill_light_2,
    }


def create_mask_output_node(output_path):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.location = 0, 0

    id_mask = tree.nodes.new('CompositorNodeIDMask')
    id_mask.index = 1
    id_mask.location = 200, 0

    mask_output = tree.nodes.new('CompositorNodeOutputFile')
    mask_output.location = 400, 0
    mask_output.format.file_format = 'PNG'
    mask_output.base_path = output_path
    mask_output.file_slots[0].path = "mask"

    links.new(render_layers.outputs['IndexOB'], id_mask.inputs['ID value'])
    links.new(id_mask.outputs['Alpha'], mask_output.inputs['Image'])

    return mask_output


def set_render_settings():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    scene.view_settings.view_transform = 'Standard'

    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()

    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        if device.type == 'GPU':
            device.use = False

    scene.cycles.device = 'GPU'
    scene.cycles.samples = 128
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPTIX'
    scene.render.image_settings.color_mode = 'RGBA'
    for obj in bpy.context.view_layer.objects:
        if obj.type == 'MESH':
            obj.pass_index = 1

    scene.view_layers[0].use_pass_object_index = True


def process_model(model_path, output_dir, pbr_textures_list, hdri_files, start_index, log_file, used_textures,
                  used_hdri, record_file):
    print(f"Starting to process model: {model_path}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=model_path)

    selected_objects = list(get_selected_objects_with_geometry())

    if not selected_objects:
        print(f"Failed to import model {model_path} or the model is empty.")
        return start_index

    bpy.ops.object.select_all(action='DESELECT')
    for obj in selected_objects:
        obj.select_set(True)
    print(f"Imported and selected objects: {[obj.name for obj in selected_objects]}")

    remove_armatures_and_animations()

    bpy.ops.object.select_all(action='DESELECT')
    for obj in selected_objects:
        obj.select_set(True)
    print(f"Selected objects after removing armatures and animation data: {[obj.name for obj in selected_objects]}")

    try:
        selected_objects = normalize_scene()
    except RuntimeError as e:
        print(f"Model {model_path} normalization failed, skipping. Error message: {e}")
        return start_index

    bpy.ops.object.select_all(action='DESELECT')
    for obj in selected_objects:
        obj.select_set(True)
    print(f"Selected objects before applying material: {[obj.name for obj in selected_objects]}")

    if len(used_textures) == len(pbr_textures_list):
        used_textures.clear()
        save_used_records(used_textures, used_hdri, record_file)

    available_textures = [t for t in pbr_textures_list if frozenset(t.items()) not in used_textures]
    if available_textures:
        random.shuffle(available_textures)
        pbr_textures = available_textures[0]
    else:
        random.shuffle(pbr_textures_list)
        pbr_textures = pbr_textures_list[0]

    used_textures.add(frozenset(pbr_textures.items()))
    save_used_records(used_textures, used_hdri, record_file)

    print(f"Selected PBR material: {pbr_textures}")
    new_material = create_pbr_material(pbr_textures)

    for obj in selected_objects:
        if obj.type == 'MESH':
            try:
                add_uv_mapping(obj)
                print(f"UV mapping applied to object: {obj.name}")
            except RuntimeError as e:
                print(f"Skipping UV mapping for object {obj.name} due to error: {e}")
                continue
            obj.data.materials.clear()
            obj.data.materials.append(new_material)
            print(f"Material {new_material.name} has been applied to object: {obj.name}")
        else:
            print(f"Object {obj.name} is not a mesh, skipping material application.")

    cameras = setup_cameras()
    set_render_settings()

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    results = []
    folder_index = start_index

    for i, (name, camera) in enumerate(cameras.items(), start=1):
        bpy.context.scene.camera = camera

        if len(used_hdri) == len(hdri_files):
            used_hdri.clear()
            save_used_records(used_textures, used_hdri, record_file)

        available_hdri = [h for h in hdri_files if h not in used_hdri]
        if available_hdri:
            random.shuffle(available_hdri)
            hdri_path = available_hdri[0]
        else:
            random.shuffle(hdri_files)
            hdri_path = hdri_files[0]

        used_hdri.add(hdri_path)
        save_used_records(used_textures, used_hdri, record_file)
        setup_world_lighting(hdri_path)
        output_dir_camera = os.path.join(output_dir, str(folder_index))
        os.makedirs(output_dir_camera, exist_ok=True)
        fig_file = os.path.join(output_dir_camera, f"fig.png")
        mask_file = os.path.join(output_dir_camera, f"mask0001.png")
        bpy.context.scene.render.filepath = fig_file

        mask_output_node = create_mask_output_node(output_dir_camera)
        mask_output_node.file_slots[0].path = "mask"

        bpy.ops.render.render(write_still=True)

        mask_dst = os.path.join(output_dir_camera, "mask.png")
        if os.path.exists(mask_file):
            shutil.move(mask_file, mask_dst)

        results.append((fig_file, mask_dst, pbr_textures, model_name, name, folder_index))
        print(f"Rendering completed: {fig_file}")

        texture_folder = os.path.dirname(next(iter(pbr_textures.values())))
        texture_name = os.path.basename(texture_folder).replace('_1K-JPG', '')
        texture_path = os.path.join(texture_folder, f"{texture_name}.png")
        with open(log_file, mode='a') as file:
            file.write(f"file_{folder_index}:{texture_path}\n")

        folder_index += 1

    return results, folder_index


def get_next_model_index(output_dir):
    if not os.path.exists(output_dir):
        return 1

    folder_indices = [int(folder) for folder in os.listdir(output_dir) if folder.isdigit()]
    if not folder_indices:
        return 1

    max_folder_index = max(folder_indices)
    next_model_index = (max_folder_index // 8) + 1
    return next_model_index


def reorder_folders_and_update_log(output_dir, log_file):
    folders = [folder for folder in os.listdir(output_dir) if folder.isdigit()]
    folders.sort(key=int)

    with open(log_file, 'r') as f:
        lines = f.readlines()

    new_log_lines = []
    for new_index, folder in enumerate(folders, start=1):
        old_path = os.path.join(output_dir, folder)
        new_path = os.path.join(output_dir, str(new_index))
        if old_path != new_path:
            shutil.move(old_path, new_path)

        for line in lines:
            if f'file_{folder}:' in line:
                new_line = line.replace(f'file_{folder}:', f'file_{new_index}:')
                new_log_lines.append(new_line)
                break

    with open(log_file, 'w') as f:
        f.writelines(new_log_lines)


def save_used_records(used_textures, used_hdri, filepath):
    data = {
        "used_textures": [list(item) for item in used_textures],
        "used_hdri": list(used_hdri)
    }
    with open(filepath, 'w') as file:
        json.dump(data, file)


def load_used_records(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
            used_textures = {frozenset(tuple(subitem) for subitem in item) for item in data.get("used_textures", [])}
            used_hdri = set(data.get("used_hdri", []))
            return used_textures, used_hdri
    return set(), set()


hdri_files = [os.path.join(hdri_dir, f) for f in os.listdir(hdri_dir) if f.endswith('.hdr')]

os.makedirs(output_dir, exist_ok=True)

log_file = os.path.join(output_dir, 'render_log.txt')

if not os.path.exists(log_file):
    open(log_file, 'w').close()

record_file = os.path.join(output_dir, 'used_records.json')
used_textures, used_hdri = load_used_records(record_file)

start_index = get_next_model_index(output_dir)

for model_dir in model_dirs:
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.glb')]
    model_files.sort()

    print(f"Starting from model index: {start_index} for directory: {model_dir}")

    for idx, model_file in enumerate(model_files, start=start_index):
        model_path = os.path.join(model_dir, model_file)
        try:
            results, start_index = process_model(
                model_path,
                output_dir,
                pbr_textures_list,
                hdri_files,
                (idx - 1) * 8 + 1,
                log_file,
                used_textures,
                used_hdri,
                record_file
            )
            save_used_records(used_textures, used_hdri, record_file)
        except Exception as e:
            print(f"Encountered an error while processing model {model_path}: {e}")

    reorder_folders_and_update_log(output_dir, log_file)
    start_index = int((start_index - 1) / 8) + 1