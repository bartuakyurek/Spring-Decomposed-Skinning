# Spring Decomposed Skinning
![Placeholder figure](./assets/placeholder_figure.png)

 This repository demonstrates a dynamic skinning deformation achieved by spring bones. Traditional skinning methods lack secondary motion such as jiggling. Our aim is to introduce dynamic motion into existing skinning pipelines through physically simulating the bones. Our work unifies the accuracy benefits of physical simulation with the computational cost and intuitive deformation benefits of the traditional skinning framework.
 
 We use Hookean spring forces simulated with Position Based Dynamics to achieve jiggling motion on the bones. Unlike the majority of physical simulation approaches, we do not require any tetrahedralization to achieve dynamics in skinning deformation.
 > [!WARNING]
 > It might take some time to load the GIFs on this page.

![Monstera plant in a pot is shaked](./assets/monstera.gif)

Our method can easily capture secondary motion dynamics in various subjects. Here we automatically compute the dynamic motion with spring forces, which would be cumbersome to replicate using traditional skinning method as the 3D artist would have to manually position the bones at each keyframe.  
![A rubber duck shaked from left to right](./assets/duck.gif)

Here we emulate dynamics for a piece of cloth using a limited set of spring bone chains. Traditional garment simulations often require a mass-spring system at every edge of the mesh, which makes the simulation dependent on the surface resolution. Our method can scale to various resolutions as we utilize skinning bones to decompose a dynamic motion.
![A piece of paper is moved back and forth](./assets/cloth.gif)

We compare our results with another method that unifies physical simulation with skinning framework by Wu and Umetani's study in 2023 [1]. 
![Comparison with Wu et al.](./assets/spot_comparison.gif)

Here the yellow dots represent the point handles that are transformed by the user. The same handles are transformed for both Wu et al. and our work. The red dots represent the point handles that are fixed in Wu et al.'s controllable PBD work and green ones are dynamic handles that are moved by controllable PBD framework. For the details of their implementation, please refer to [their webpage](https://yoharol.github.io/pages/control_pbd/). In our work, blue handles represent the spring bones that are simulated in our pipeline. We can achieve both global secondary dynamics that jiggles major body parts of the mesh, and local secondary dynamics for soft tissues.  The point handles on the above figure are bound to larger areas of the mesh, hence they can produce global dynamics.

![Helper bone chains simulate local dynamics](./assets/spot_helpers.gif)

To introduce local dynamics, we utilize helper bones that are additional bones often places perpendicular to the rigid bones. In the figure above, we place the helper chains that are bound to smaller local areas on a mesh to emulate jigglings of the soft tissues. 
The yellow handles are moved by the user that creates the rigid motion, then the blue helper spring chains are simulated via PBD, that produces the jiggling of the soft tissues.


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
>  Note that this repo is a work in progress. Currently you can run either the files under ``./tests/`` or ``./demo/`` directories. (You may need to change directories to run them)

## References
[1] . Wu and N. Umetani, “Two-way coupling of skinning transformations and
position based dynamics,” Proc. ACM Comput. Graph. Interact. Tech., vol. 6, Aug. 2023.


## TODO
[ ] Allow running from the directory ./test/test_file.py (change step 3)

[ ] Add instructions for setting up DFAUST data

[ ] Add references to PyVista and other source code