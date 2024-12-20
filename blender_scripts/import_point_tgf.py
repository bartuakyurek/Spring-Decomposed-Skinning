"""
Blender version 3.8

This file is intended to create an armature given a .tgf file.
Note that it assumes the .tgf has point handles as the paper
we compare against has that data structure. 

"""

import os
import bpy
import mathutils

import igl
import meshio
import numpy as np

# ===========================================================
# Editable variables
# ===========================================================
modelname = "spot"
path_to_data = f"/Users/bartu/Documents/Github/Spring-Decomp/data/{modelname}"

armature_name = "armature"
bone_basename = "Rigid"

bonescale = 0.2

armature_path = os.path.join(path_to_data, f"{modelname}.tgf")
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

# DISCLAIMER: This functions is adoptec from
# https://b3d.interplanety.org/en/how-to-delete-object-from-scene-through-the-blender-python-api/
def delete_object(obj):
    if bpy.context.object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    # deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    # select the object
    obj.select_set(True)
    # delete all selected objects
    bpy.ops.object.delete()
    return

# DISCLAIMER: This function is adopted from
# https://wiki.blender.jp/Dev:Py/Scripts/Cookbook/Code_snippets/Three_ways_to_create_objects
def create_armature(name, origin=(0,0,0)):
    # Create armature and object
    amt = bpy.data.armatures.new(name)
    ob = bpy.data.objects.new(name, amt)
    ob.location = origin
    ob.show_name = True

    # Link object to scene and make active
    bpy.context.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    ob.show_in_front = True
    return ob

def add_point_bones(armatureObj :  bpy.types.Object, 
                    points, 
                    bone_basename : str):
    
    bpy.context.view_layer.objects.active = armatureObj                  
    bpy.ops.object.mode_set(mode='EDIT')

    amt = armatureObj.data # bpy.types.Armature 
    #[delete_object(eb) for eb in amt.edit_bones] # Clear cache
    
    for i,pt in enumerate(points):
        print(f">>> Adding bone {i}...")
        bonename = bone_basename + f".{i}"
        bone = amt.edit_bones.new(bonename)
        
        bone.head = (0,0,0) #np.array([0,0,0])
        bone.tail = (0,1*bonescale,0)
        bone.translate(mathutils.Vector(pt)) 
        
        #bpy.context.scene.cursor.location = pt
        #bpy.ops.armature.bone_primitive_add(name=bonename)
        
    print(">>> WARNING: Cached armatures: ", len(bpy.data.armatures))
    print(">>> INFO: Armature edit bones: ", len(amt.edit_bones))
    print(">>> INFO: Armature bones: ", len(amt.bones))
     
    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode='OBJECT')
    return

# ===========================================================
# M A I N
# ===========================================================
if not os.path.exists(path_to_data):
    print(f">>> ERROR: Data path {path_to_data} is not found.")


# Check if .tgf is already loaded, otherwise load it
obj = object_exists(armature_name, use_basename=True)
if obj:
    print(f">>> INFO: {armature_name} already exists. Deleting the previous armature...")
    delete_object(obj)

print(f">>> INFO: Loading armature from {armature_path}...")
    
V, _, _, _, _, _ = igl.read_tgf(armature_path)
print(f">>> INFO: Loaded point handles {V.shape} at \n", V)
print(">>> WARNING: This script assumes bones are point handles. Edges and kintree are omitted.")
    
# Add an armature
print(">>> INFO: Creating an armature...")
amt = create_armature(armature_name)
    
# Add point bones to the armature (no parent-child relationship)
print(">>> INFO: Adding point bones to the armature...")
add_point_bones(amt, V, bone_basename)


