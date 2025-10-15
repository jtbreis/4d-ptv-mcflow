4D-PTV
=======

4D-PTV is a package to do 4D Particle Tracking Velocimetry. This package is composed of all functions required to find particles trajectories and to treat them doing pair dispersion...

Installation
------------

The last version of the code can be obtained from the gitlab repository:

```
git clone https://github.com/turbulencelyon/4d-ptv
```
To compile all executable files:

```
cd Documentation/
make
cd ../Matching/STMCpp/
make
```

Please refer to the [online documentation](https://4d-ptv.readthedocs.io/en/latest/) for more information.

### Run as devcontainer

'''matlab-proxy-app'''
You need to keep the terminal open in the background


## Run docker container
`docker run -d -v /mnt/ssd:/workspaces/4d-ptv-mcflow/data --mount type=bind,source=/mnt/raid/,target=/workspaces/4d-ptv-mcflow/raw_data,readonly particle_tracking --case_name=CASE_NAME `

Run after cloning!
`git submodule init`