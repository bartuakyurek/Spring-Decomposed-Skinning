import bpy
import numpy as np

# Thanks to 
https://blenderartists.org/t/procedural-weight-painting/596228/4
# Also see 
https://blender.stackexchange.com/questions/3249/show-mesh-vertices-id 
# to toggle indices 

# ensure we are in edit mode
obj = bpy.context.active_object
bpy.ops.object.mode_set(mode="EDIT")

vertexGroupName = 'Weights'  
mesh = obj.data

selVerts = [v for v in mesh.vertices]
indexVal = obj.vertex_groups[vertexGroupName].index

weights = []
idxs = []
for i, v in enumerate(selVerts): #loop over all verts
    for n in v.groups: #loop over all groups with each vert
        if n.group == indexVal: #check if the group val is the same as the 
index value of the required vertex group
            weights.append(n.weight)
            idxs.append(i)
            

# (vert_id, weight)
weights = np.array(weights)
idxs = np.array(idxs, dtype=int)
print(f"Weights {weights.shape} saved...")  
print(f"Indices {idxs.shape} saved...")
  
PATH = "/Users/bartu/Documents/Github/Spring-Decomp/data/"
np.savez(PATH + "single-bone-weights-0.npz", weights, idxs)       

bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
