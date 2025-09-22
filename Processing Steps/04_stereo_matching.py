import subprocess

filename = 'data/julian/PTV_center/TTI_aligned_with_gravity/Run1/rays.h5'
output = 'data/julian/PTV_center/TTI_aligned_with_gravity/Run1'
frames = 10
mincameras = 3
maxdistance = 1
multiplematchesperraydistance = 1
maxmatchesperray = 2
# Number of Voxels in direction
nx = 400
ny = 400
nz = 200
# Bounding Box
minX = -30
maxX = 30
minY = -30
maxY = 30
minZ = -20
maxZ = 20

##########

# TODO make the Stereomatching an actual part of the python package
run_command = f'./Matching/STMCpp/STM -i {filename} -o {output} -f {frames} -c {mincameras} -d {maxdistance} -s {multiplematchesperraydistance} -m {maxmatchesperray} -x {nx} -y {ny} -z {nz} -b {minX} {maxX} {minY} {maxY} {minZ} {maxZ} --hdf5'

# Launch the process
proc = subprocess.Popen(
    # replace with your command
    run_command.split(),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,           # automatically decode bytes to string
    bufsize=1            # line-buffered
)

# Continuously read lines as they are printed
for line in proc.stdout:
    print(line, end="")  # already has newline

# Wait for the process to finish
proc.wait()
