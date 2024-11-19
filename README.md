# Spring Decomposed Skinning
 This repository demonstrates a dynamic skinning deformation achieved by spring bones. Note that it might take some time to load of the GIFs in this page.

![Monstera plant in a pot is shaked](./assets/monstera.gif)

![A rubber duck shaked from left to right](./assets/duck.gif)

![A piece of paper is moved back and forth](./assets/cloth.gif)


## Installation Steps

**Step 1** It's recommended to first create a virtual environment. Assuming you're inside the source directory, run:
```
$ python -m venv venv
$ source venv/bin/activate
```
**Step 2** To install the dependencies, run:
```
$ pip install -r requirements.txt
```

**Step 3** To run a demo, change directory to ``/demo`` and run the relevant script:
```
$ cd demo
$ python spot_demo.py
```

> [!TIP]
> To deactivate the virtual environment, run:
>
> `` $ deactivate ``

**Download Data**
 
For DFAUST demos, the relevant dataset should be downloaded from (link here). This step is not required for other demos as their relevant data is included in ``/data`` directory.

> [!WARNING]
>  Note that this repo is a work in progress. Currently you can run either the files under ``./tests/`` or ``./demo/`` directories. (You may need to change directories to run them)


## TODO
[ ] Allow running from the directory ./test/test_file.py (change step 3)

[ ] Add instructions for setting up DFAUST data