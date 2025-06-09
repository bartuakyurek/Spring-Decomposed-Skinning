# Spring Decomposed Skinning (SDS)

> [!IMPORTANT]  
> For further information, please refer to our paper [SDS webpage](https://bartuakyurek.github.io/publications/SDS/)

### Summary of Features

- **Dynamic Motion with a simple framework:** We implement simple Hookean spring forces directly on the rig bones with Position-Based Dynamics (PBD) to emulate secondary dynamics caused by the skeletal motion. 
- **No Tetrahedralization Required:** Unlike most physical simulation methods, we avoid computationally expensive tetrahedralization while still achieving realistic dynamics.
- **Scalability:** We avoid any simulation on the surface vertices which allows us to scale to higher resolution meshes just like the traditional skinning pipelines.
- **User-Controlled Dynamics:** We provide an intuitive control over the global and local deformation dynamics. This is achieved by introducing spring forces on both primary bones and helper bones.


 > [!WARNING]
 > It might take some time to load all the GIFs on this page.

## Visual Demo
Our method effectively captures secondary motion dynamics that would otherwise require manual bone positioning for each keyframe.

![Monstera plant in a pot is shaked](./assets/monstera_lq.gif)
**Figure 2.1:** A plant pot is shaken right to left. Traditional skinning methods cannot automatically capture the secondary dynamics.
![A rubber duck shaked from left to right](./assets/duck_lq.gif)
**Figure 2.2:** A rubber duck is rotated back and forth. Our deformation introduces dynamic effects as if the duck is floating on the water without any complex simulation over the surface.

In the figures above, blue bones are simulated within our framework. We automatically compute the dynamic motion, which would be cumbersome to replicate using traditional skinning method as the 3D artist would have to manually position the bones at each keyframe.  


## Installation Steps

**Step 1** It's recommended to first create a virtual environment:
```
$ python -m venv venv
$ source venv/bin/activate
```
**Step 2** Inside the project directory, install the dependencies:
```
$ pip install -r requirements.txt
```

**Step 3** To run a demo, change directory to ``/demo`` and run the relevant script:
```
$ cd demo
$ python spot_demo.py
```

> [!TIP]
> To deactivate the virtual environment, use:
>
> `` $ deactivate ``

**Download Data**
 
For DFAUST demos, the relevant dataset should be downloaded from (link here). This step is not required for other demos as their relevant data is included in the ``/data`` directory.

> [!WARNING]
>  Note that this repo is a work in progress and might have been missing the latest updates. 
> Currently you can run either the files under ``./tests/`` or ``./demo/`` directories. (You may need to change directories to run them)


## References
[1] Wu and N. Umetani, “Two-way coupling of skinning transformations and
position based dynamics,” Proc. ACM Comput. Graph. Interact. Tech., vol. 6, Aug. 2023.

[2] Bogo, J. Romero, G. Pons-Moll, and M. J. Black, “Dynamic FAUST: Regis-
tering human bodies in motion,” in IEEE Conf. on Computer Vision and Pattern
Recognition (CVPR), July 2017.

## Acknowledgements
Thanks to
* [@yoharol](https://github.com/yoharol/Controllable_PBD_3D) for making their Controllable PBD research open source.
* [printable_models](https://free3d.com/3d-model/rubber-duck-v1--164824.html) for their rubber duck model.
* [PyVista](https://docs.pyvista.org/) for their visualization tools.

## Contact
You can contact me at bartu.akyurek@metu.edu.tr
