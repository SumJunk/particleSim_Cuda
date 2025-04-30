# Particle

# Overview  
Simulates particle movement under gravitational forces using Gravitational Force and Equations of Motion to update the position of a particle over time. You may add the number of particles, time step per update, number of simulation steps, and the output intervals to modify your experience. "output.log" is then created to check each interval. To improve performance, the simulation uses OpenMP for parallel computation of gravitational forces and particle updates.
 

# Features  
- Random particle generation, calculation of gravitational forces between each pair of particles.
- Velocity updates based on forces. 
- Uses GPU acceleration via CUDA 12.4.
- Compiles with nvcc, nvcc -arch=sm_75 -O2 -o nbody nbody.cu
 
 
# Clone the Repository  
To download and use the project on Centaurus ensure you are on a computing node, run:  
```bash
git clone https://github.com/SumJunk/particleSim_Cuda.git
cd particleSim_Cuda
git checkout master
g++ -O3 nbody.cpp -o nbody
./nbody <nbpart> <dt> <nbstep> <printevery> <threads>