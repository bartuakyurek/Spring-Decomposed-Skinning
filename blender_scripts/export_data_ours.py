"""
    This script is intended to export the necessary data for our pipeline.
    That includes:
        - data['kintree'] : (parent, child) pairs of bone indices  (x, 2)
        - data['joints'] : Endpoint locations for bones (n_bones, 2, 3)
        - data['weights'] : vertex-bone binding weights (n_verts, n_bones)
        - data['rigid_idxs'] : indices of rigid bones in the rig (n_rigid)
"""
import os
import bpy
import mathutils
import numpy as np

# Editable Parameters
# ---------------------------------------------------------------------------------------
modelname = "spot_helpers"
armature_name = "armature" # Name of the armature on the right panel, by default it is "Armature"
mesh_name = "tetmesh"     # Use the mesh name on the right panel
RIGID_NAME = "Rigid" # Set None or Rigid bone basename

save_txt_weights = True
#save_regular_tgf = True  # Named regular to distinguish Wu et al.'s point handle tgf format with regular tgf
RESET_ROOT_WEIGHTS = False  # Set True you want to set root bone weights to zero

OUT_FILENAME = f"{modelname}_rig_data.npz" 
PATH = f"/Users/bartu/Documents/Github/Spring-Decomp/data/{modelname}/" # Directory to save the output
weight_path = os.path.join(PATH, f"{modelname}_w.txt")
#regular_tgf_path = os.path.join(PATH, f"{modelname}_regular.tgf")

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

def get_bone_name_idx(armature_obj, name):
    # Get the bone indices corresponding to name basename.
    armature = armature_obj.data
    name_idxs = []
    for bone_idx, bone in enumerate(armature.bones):
        if str.lower(bone.basename) == str.lower(name):
            name_idxs.append(bone_idx)
        
    return name_idxs

# ---------------------------------------------------------------------------------------
# Part 2 - Extract the bone head and tail positions of shape (n_bones, 2, 3)
# ---------------------------------------------------------------------------------------
def get_joint_locations(armature_obj, name="", local=True):
    """
    Get the joint locations in the tree.
    NOTE: Every bone is assumed to have two joints: bone.head and bone.tail
    """
    if local: t = mathutils.Vector([0,0,0])
    else: t =  armature_obj.location
    print("Armature location is taken as:", t)
    
    armature = armature_obj.data
    n_bones = len(armature.bones)
    J = np.empty((n_bones,2,3))
    
    name_idxs = []
    for i, bone in enumerate(armature.bones):
        J[i,:,:] = np.array([bone.head_local + t, bone.tail_local + t])
        
        if str.lower(bone.basename) == str.lower(name):
            name_idxs.append(i)
            
    return J, name_idxs

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
J, rigid_idxs  = get_joint_locations(armature_obj, RIGID_NAME, local=True)
W = get_binding_weights(mesh_obj)#[:,3:]

print("Obtained kintree (parent_idx, bone_idx):\n", kintree)
print("Obtained joint locations:\n", np.round(J,4))

if RESET_ROOT_WEIGHTS:
    W[0] = np.zeros_like(W[0]) # Assumes first bone is the root 
  
data = {}
data['kintree'] = kintree
data['joints'] = J
data['weights'] = W

#if RIGID_NAME is not None: 
    #rigid_idxs = get_bone_name_idx(armature_obj, RIGID_NAME)
data['rigid_idxs'] = rigid_idxs

np.savez(PATH + OUT_FILENAME, **data) 
print(f">> Kintree {kintree} is saved.")
print(f">> Joints of shape {J.shape} is saved.")
print(f">> Weights of shape {W.shape} saved. Are root weights set to zero? {RESET_ROOT_WEIGHTS}.")

if save_txt_weights:
    np.savetxt(weight_path, W)
    print(f">>> INFO: Saved weights at {weight_path}")
    
#if save_regular_tgf:
#    write_regular_tgf(J, kintree, regular_tgf_path)
#    print(f">>> INFO: Saved a regular .tgf at {regular_tgf_path}")