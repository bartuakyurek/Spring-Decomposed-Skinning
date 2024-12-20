"""
This file is intended to extract the armature (as point handles) in .tgf format.

"""


import os
import bpy
import mathutils

import igl
import numpy as np


# ===========================================================
# Editable variables
# ===========================================================
modelname = "spot_helpers"
armature_name = "armature"

path_to_data = f"/Users/bartu/Documents/Github/Spring-Decomp/data/{modelname}"
tgf_path = os.path.join(path_to_data, f"{modelname}.tgf")


# ===========================================================
# Functions
# ===========================================================
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
            return o
    return False

def get_point_bones(armatureObj :  bpy.types.Object,
                    get_tails = False):
    
    bpy.context.view_layer.objects.active = armatureObj                  
    bpy.ops.object.mode_set(mode='EDIT')

    amt = armatureObj.data # bpy.types.Armature 
    #[delete_object(eb) for eb in amt.edit_bones] # Clear cache
    
    J = np.empty((len(amt.edit_bones),3))
    for i,bone in enumerate(amt.edit_bones):
        
        pt = bone.head
        if get_tails:
            pt = bone.tail
        
        J[i] = pt
        
    print(">>> INFO: Point bones: ",J.shape)
    return J

def write_point_tgf(J, tgf_path):
    """
    This function is intended to create .tgf file that
    is expected in Wu et al.'s source code.
    
    WARNING: It does not correspond to original .tgf in libigl's docs.
    In libigl's write_tgf() tab is used as separator but Wu et al.
    reads .tgf with space separators, and omits parent-child relationships.
    
    J: Point locations for point handles (n_handles, 3)
    tgf_path: path to .tgf file to be written
    
    """
    with open(tgf_path, "w") as file:
        # Iterate over joints and write to tgf lines
        for i in range(len(J)):
            pt = J[i]
            line = f"{i} {pt[0]} {pt[1]} {pt[2]}\n"
            file.write(line)
        file.write("#")
        
    return True

# ===========================================================
# M A I N
# ===========================================================
if not os.path.exists(path_to_data):
    print(f">>> ERROR: Data path {path_to_data} is not found.")

# Check if armature to be saved exists
armatureObj = object_exists(armature_name, use_basename=True)
if armatureObj:    
    print(f">>> INFO: Saving armature as point handles in {tgf_path}...")
    
    J = get_point_bones(armatureObj)
    if write_point_tgf(J, tgf_path):
        print(f">>> Saved point handles at {tgf_path}")     
else:
    print(f">>> ERROR: Armature not found! No .tgf saved!")
