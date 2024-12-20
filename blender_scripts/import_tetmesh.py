"""
Blender version 3.8

This file is intended to:
- Import .mesh tetrahedral mesh

ACKNOWLEDGEMENTS: 
    - For creating a Blender mesh out of given vertices, thanks to:
    - https://blenderartists.org/t/create-vertex-and-edges-using-this-code/648786/3
    I've adjusted last few lines to fix the errors
    - See: https://blender.stackexchange.com/questions/140789/what-is-the-replacement-for-scene-update
    - See: https://blender.stackexchange.com/questions/162256/bpy-prop-collection-object-has-no-attribute-link
"""

import os
import bpy

import igl
import meshio

# ===========================================================
# Editable variables
# ===========================================================
modelname = "spot"
path_to_data = f"/Users/bartu/Documents/Github/Spring-Decomp/data/{modelname}"
validate = False # Adjust the geometry within Blender if necessary
                            # set to False because we'd need to export .mesh
                            # if the geometry is changed. For now, I assume the
                            # input is valid. (e.g. modelname="spot" is changed
                            # when .validate() is called so I set it to False.)
                            
basename = "tetmesh" # To check if this script already imported tetmesh
# ===========================================================
# Functions
# ===========================================================
# DISCLAIMER: This function is adopted from
# https://github.com/yoharol/Controllable_PBD_3D
def load_tet(filename: str, reverse_face=False):
  mesh = meshio.read(filename)
  verts = mesh.points
  assert 'tetra' in mesh.cells_dict, 'only tetra mesh is supported'
  tets = mesh.cells_dict['tetra']
  faces = mesh.cells_dict['triangle']
  if reverse_face:
    faces = np.flip(faces, 1)
  return verts, tets, faces

def object_exists(objname : str, use_basename : bool):
    # If use_basename is set True, objname is assumed to be the basename
    # E.g. object_exists("bone", True)
    # returns True if there's object "bone.0001.R"
    # because its name before the first dot matches with objname.
    for o in bpy.context.scene.objects:
        name = o.name
        if use_basename:
            name = name.split('.')[0]
        
        if name == objname:
            return True
        
    return False

# ===========================================================
# M A I N
# ===========================================================
if not os.path.exists(path_to_data):
    print(f">>> ERROR: Data path {path_to_data} is not found.")

# Check if tetmesh is already loaded, otherwise load it.
if object_exists(basename, use_basename=True):
    print(f">>> INFO: {basename} already exists. No tetmesh is imported.")

else:
    # Load .mesh file into Vertices, Faces arrays
    mesh_path = os.path.join(path_to_data, f"{modelname}.mesh")
    V, E, F = load_tet(mesh_path)
    print(f">>> INFO: Loaded vertices {V.shape} and faces {F.shape}")

    # Create new blender mesh out of V, F
    me = bpy.data.meshes.new(basename)
    me.from_pydata(V, [], F)
    
    if validate:
        if me.validate(verbose=True): 
            # Mesh.validate() returns True when the mesh 
            # has had invalid geometry corrected/removed
            # See: https://docs.blender.org/api/current/bpy.types.Mesh.html#bpy.types.Mesh.validate
            print(">>> WARNING: Mesh validation had to adjust geometry!")
        
    me.update()

    ob = bpy.data.objects.new(basename, me)
    bpy.context.collection.objects.link(ob)
    bpy.context.view_layer.update()
