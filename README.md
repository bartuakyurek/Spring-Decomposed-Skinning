# Spring Decomposed Dynamic Skinning
 This repository demonstrates a dynamic deformation achieved by adding spring helper bones to an 
existing geometric skinning pipeline. Note that it's a work in progress. Currently you can run either 
the files under ./tests/ or ./demo/ directories, e.g. ``python ./tests/spring_bone_test.py``.

![A chain of skeletal bones jiggles through a rigid motion.](./assets/jiggle-chain.gif)
![Comparison of DFAUST vs. SMPL vs. Spring Decomposed Skinning.](./assets/dfaust_comparison_50004_jiggle_on_toes.gif)



### TO-DO

[ ] Adjust plotter camera zoom

[ ] Add demo gifs

[ ] Color code evaluation

[ ] Update instructions to use this repo


It's recommended to first create a virtual environment. Assuming you're inside the source directory, run:
```
$ python -m venv venv
$ source venv/bin/activate
```

Then, to install the dependencies, run:
```
$ pip install -r requirements.txt
```

To deactivate the virtual environment, run:
```
$ deactivate
```