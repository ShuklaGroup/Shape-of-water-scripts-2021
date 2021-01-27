### Script for calculate the x,y,z positional coordinate of water molecules within zeolite framework from MD simulations
### Author: Matthew Chan

import mdtraj as md
import numpy as np
import glob
import os


for file in sorted(glob.glob('../*center.xtc')):
	#Load the trajectory and parameter file
  t = md.load(file, top='../BEA-SIOH-water.prmtop')
	
  water = t.topology.select("water and name O")  ##indexes of water
	nFrames = t.n_frames
	water_xyz = []

	for frame in range(nFrames):
          #Define the boundaries of the zeolite and only select water molecules within the planes of the zeolite
        	waters_in_zeo = [water_atom for water_atom in water if t.xyz[frame][water_atom][0] > 1.7 and t.xyz[frame][water_atom][0] < 5.0 and t.xyz[frame][water_atom][1] >1.7 and t.xyz[frame][water_atom][1] < 5.0 and t.xyz[frame][water_atom][2] >1.2 and t.xyz[frame][water_atom][2] < 6.5]

        	for j in waters_in_zeo:
                	water_xyz.append(t.xyz[frame][j])

  ##Save the array of x,y,z positions of water molecules in the zeolite.
	dir=os.path.dirname(file)
	filename=file.replace(dir+'/','',1)
	np.save(filename+'-water_xyz.npy', water_xyz)
	print(filename)

