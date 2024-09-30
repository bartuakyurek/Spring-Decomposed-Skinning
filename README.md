# Spring Decomposed Dynamic Skinning
 This repository demonstrates a dynamic deformation achieved by adding spring helper bones to an 
existing geometric skinning pipeline. Note that it's a work in progress.



### TO-DO

- Integrate an animation visualizer (either by a  library or completing visualizer.py files) that renders 
every frame of animation given a sequence of meshes for every frame (note that it's not a real-time rendering)

- Construct a mass-spring lattice 

- Bind a mass-spring latice to parent bones to inherit rigid bone transformations

- Implement FK/IK required to compute transforms for each bone (use optimal rigid motion for springs) 

- Bind vertices to bones for skinning weights 
