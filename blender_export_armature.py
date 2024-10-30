"""
HOW TO USE THIS SCRIPT?
--------------------------------------------------------------------------------------------------
In Blender,
- Setup an armature with a single root bone. This armature will be used as helper rig in Spring
  Decomposition pipeline, excluding the root.
- Bind the armature to the mesh via automatic weights.
- Edit the weights via weight painting. Note that not all the bones have to have weights.
- (Optional) Make all the weights of root bone 1.0 to check the influence of helper bones clearly.

Given this procedure, this script assumes the root bone will NOT be exported. 
It will export "helper_data.npz" which includes:
    
    - Kintree : Parent-child relationships between bones. The root bone index is -1. It will be
                replaced with the actual rig bone index in Spring Decomposition pipeline. 
                Every entry has (parent_idx, bone_idx) tuples.
                Has shape (n_helpers, 2). Access via data['kintree'] after loading the .npz.
                
    - J : Enpdpoint locations of each bone (for now, please make sure there's no
          offset between bones, offset configuration has not been tested on our pipeline yet).
          Has shape (n_helpers, 2, 3). Access via data['joints'] after loading the .npz.
    
    - W : Vertex-bone binding weights that are via weight paintign or automatic methods.
          Has shape(n_verts, n_helpers). Access via data['weights'] after loading the .npz.

In this script,
    - Simply edit the parameters listed right after the imports.
    
"""
import bpy
import numpy as np


# Editable Parameters
# ---------------------------------------------------------------------------------------
armature_name = "Armature" # Name of the armature on the right panel, by default it is "Armature"
mesh_name = "smpl_rest"    # Use the mesh name on the right panel
OUT_FILENAME = "helper_rig_data.npz" 
PATH = "/Users/bartu/Documents/Github/Spring-Decomp/data/" # Directory to save the output
RESET_ROOT_WEIGHTS = True  # Set True you want to set root bone weights to zero
# ---------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
# Part 1 - Extract the bone parent-child tree of shape (n_bones, 2)
# ---------------------------------------------------------------------------------------
def get_kintree(armature_obj):
    
    armature = armature_obj.data
    bone_names = [b.name for b in armature.bones]
    
    kintree = []
    for bone_idx, bone in enumerate(armature.bones):
        if bone.parent:
            parent_idx = bone_names.index(bone.parent.name)
        else:
            parent_idx = -1
        kintree.append([parent_idx, bone_idx])
        
    return kintree 

# ---------------------------------------------------------------------------------------
# Part 2 - Extract the bone head and tail positions of shape (n_bones, 2, 3)
# ---------------------------------------------------------------------------------------
def get_joint_locations(armature_obj):
    
    t =  armature_obj.location
    print("Armature location:", t)
    
    armature = armature_obj.data
    n_bones = len(armature.bones)
    J = np.empty((n_bones,2,3))
    
    for i, bone in enumerate(armature.bones):
        J[i,:,:] = np.array([bone.head_local + t, bone.tail_local + t])
        
    return J

# ---------------------------------------------------------------------------------------
# Part 3 - Extract the vertex-bone weights of shape (n_verts, n_bones)
# ---------------------------------------------------------------------------------------
def get_binding_weights(mesh_obj):
    
    mesh = mesh_obj.data 
    verts = mesh.vertices #[v for v in mesh.vertices]

    n_verts = len(verts)
    n_bones = len(mesh_obj.vertex_groups)
    print("Number of verts: ", n_verts)
    print("Number of bones: ", n_bones)

    n_nonzero_weights = 0 # For sanity check
    W = np.zeros((n_verts, n_bones))
    bone_vertex_groups = mesh_obj.vertex_groups

    # Loop over bones in Armature
    for bone_group in bone_vertex_groups:
        j = bone_group.index
        # Loop over vertices
        for i, v in enumerate(verts): 
            try:
                weight = bone_group.weight(i)
            except:
                weight = 0.0
            W[i, j] = weight
            
            if weight > 0 and weight < 1:
                n_nonzero_weights += 1
                
    print("Number of nonzero weights: ", n_nonzero_weights)
    assert n_nonzero_weights > 0, f"Expected some weights in range (0, 1), are you sure you weight painted correctly?"
    return W

# ---------------------------------------------------------------------------------------
# Save the extracted data Kintree, Joints, Weights.
# ---------------------------------------------------------------------------------------

mesh_obj = bpy.data.objects[mesh_name]
armature_obj = bpy.data.objects[armature_name]

kintree = get_kintree(armature_obj)
J = get_joint_locations(armature_obj)
W = get_binding_weights(mesh_obj)

print("Obtained kintree (parent_idx, bone_idx):\n", kintree)
print("Obtained joint locations:\n", np.round(J,4))

if RESET_ROOT_WEIGHTS:
    W[0] = np.zeros_like(W[0]) # Assumes first bone is the root 
  
data = {}
data['kintree'] = kintree
data['joints'] = J
data['weights'] = W

np.savez(PATH + OUT_FILENAME, **data) 
print(f">> Kintree {kintree} is saved.")
print(f">> Joints of shape {J.shape} is saved.")
print(f">> Weights of shape {W.shape} saved. Are root weights set to zero? {RESET_ROOT_WEIGHTS}.")

     