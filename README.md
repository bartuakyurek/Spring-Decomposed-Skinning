# Spring Decomposed Skinning
 This repository demonstrates a dynamic deformation achieved by adding spring helper bones to an 
existing geometric skinning pipeline. 

![Monstera plant in a pot is shaked](./assets/monstera.gif)

![A rubber duck shaked from left to right](./assets/duck.gif)


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

> [!WARNING]
>  Note that this repo is a work in progress. Currently you can run either the files under ``./tests/`` or ``./demo/`` directories. (You may need to change directories to run them)


## TODO
[ ] Allow running from the directory ./test/test_file.py (change step 3)