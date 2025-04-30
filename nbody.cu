#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


double G = 6.67430e-11;
double dt = 0.01;
size_t nbstep = 1000;
size_t printevery = 100;

struct State {
    std::vector<double> mass, x, y, z;
    std::vector<double> vx, vy, vz;
    std::vector<double> fx, fy, fz;
    int nbpart;
};

void dump_state(const State& s) {
    std::cout << "Dumping state for " << s.nbpart << " particles:\n";
    for (int i = 0; i < s.nbpart; ++i) {
        std::cout << "Particle " << i << ": "
                  << "x=" << s.x[i] << ", y=" << s.y[i] << ", z=" << s.z[i]
                  << ", vx=" << s.vx[i] << ", vy=" << s.vy[i] << ", vz=" << s.vz[i]
                  << ", fx=" << s.fx[i] << ", fy=" << s.fy[i] << ", fz=" << s.fz[i]
                  << "\n";
    }
}

__global__ void compute_forces(int nbpart, double G, double softening,
  const double* mass, const double* x, const double* y, const double* z,
  double* fx, double* fy, double* fz) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nbpart) return;

  double fx_i = 0.0, fy_i = 0.0, fz_i = 0.0;
  double xi = x[i], yi = y[i], zi = z[i];

  for (int j = 0; j < nbpart; ++j) {
      if (i == j) continue;

      double dx = x[j] - xi;
      double dy = y[j] - yi;
      double dz = z[j] - zi;
      double distSqr = dx * dx + dy * dy + dz * dz + softening;
      double distSixth = distSqr * sqrt(distSqr);
      double F = G * mass[i] * mass[j] / distSixth;

      fx_i += dx * F;
      fy_i += dy * F;
      fz_i += dz * F;
  }

  fx[i] = fx_i;
  fy[i] = fy_i;
  fz[i] = fz_i;
}

__global__ void update_particles(int nbpart, double dt,
  double* x, double* y, double* z,
  double* vx, double* vy, double* vz,
  const double* fx, const double* fy, const double* fz,
  const double* mass) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nbpart) return;

  vx[i] += fx[i] / mass[i] * dt;
  vy[i] += fy[i] / mass[i] * dt;
  vz[i] += fz[i] / mass[i] * dt;

  x[i] += vx[i] * dt;
  y[i] += vy[i] * dt;
  z[i] += vz[i] * dt;
}

void run_simulation(State& s, size_t nbstep, double dt, double G, size_t printevery, int threads){

  int nbpart = s.nbpart;
  size_t bytes = nbpart * sizeof(double);

  double *d_mass, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_fx, *d_fy, *d_fz;
  cudaMalloc(&d_mass, bytes);
  cudaMalloc(&d_x, bytes);  cudaMalloc(&d_y, bytes);  cudaMalloc(&d_z, bytes);
  cudaMalloc(&d_vx, bytes); cudaMalloc(&d_vy, bytes); cudaMalloc(&d_vz, bytes);
  cudaMalloc(&d_fx, bytes); cudaMalloc(&d_fy, bytes); cudaMalloc(&d_fz, bytes);

  cudaMemcpy(d_mass, s.mass.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, s.x.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, s.y.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, s.z.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vx, s.vx.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vy, s.vy.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vz, s.vz.data(), bytes, cudaMemcpyHostToDevice);

  int blocks = (nbpart + threads - 1) / threads;

  for (size_t step = 0; step < nbstep; ++step) {
      compute_forces<<<blocks, threads>>>(nbpart, G, 0.1, d_mass, d_x, d_y, d_z, d_fx, d_fy, d_fz);

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
          std::cerr << "Error after compute_forces: " << cudaGetErrorString(err) << std::endl;
          return;
      }

      update_particles<<<blocks, threads>>>(nbpart, dt, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass);

      err = cudaGetLastError();
      if (err != cudaSuccess) {
          std::cerr << "Error after update_particles: " << cudaGetErrorString(err) << std::endl;
          return;
      }

      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
          std::cerr << "Error after cudaDeviceSynchronize: " << cudaGetErrorString(err) << std::endl;
          return;
      }

      if (step % printevery == 0) {
          cudaMemcpy(s.x.data(), d_x, bytes, cudaMemcpyDeviceToHost);
          cudaMemcpy(s.y.data(), d_y, bytes, cudaMemcpyDeviceToHost);
          cudaMemcpy(s.z.data(), d_z, bytes, cudaMemcpyDeviceToHost);
          cudaMemcpy(s.vx.data(), d_vx, bytes, cudaMemcpyDeviceToHost);
          cudaMemcpy(s.vy.data(), d_vy, bytes, cudaMemcpyDeviceToHost);
          cudaMemcpy(s.vz.data(), d_vz, bytes, cudaMemcpyDeviceToHost);
          cudaMemcpy(s.fx.data(), d_fx, bytes, cudaMemcpyDeviceToHost);
          cudaMemcpy(s.fy.data(), d_fy, bytes, cudaMemcpyDeviceToHost);
          cudaMemcpy(s.fz.data(), d_fz, bytes, cudaMemcpyDeviceToHost);

          dump_state(s);
      }
  }

  cudaFree(d_mass);
  cudaFree(d_x);  cudaFree(d_y);  cudaFree(d_z);
  cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
  cudaFree(d_fx); cudaFree(d_fy); cudaFree(d_fz);
}


int main(int argc, char* argv[]) {
  if (argc != 6) {
      std::cerr << "Usage: " << argv[0]
                << " <num_particles> <dt> <nbstep> <printevery> <threads_per_block>\n";
      return 1;
  }

  int nbpart = std::stoi(argv[1]);
  double dt = std::stod(argv[2]);
  size_t nbstep = std::stoul(argv[3]);
  size_t printevery = std::stoul(argv[4]);
  int threads = std::stoi(argv[5]);

  State s;
  s.nbpart = nbpart;
  s.mass.resize(nbpart);
  s.x.resize(nbpart); s.y.resize(nbpart); s.z.resize(nbpart);
  s.vx.resize(nbpart); s.vy.resize(nbpart); s.vz.resize(nbpart);
  s.fx.resize(nbpart); s.fy.resize(nbpart); s.fz.resize(nbpart);

  std::mt19937 gen(42);
  std::uniform_real_distribution<> pos(-1.0, 1.0);
  std::uniform_real_distribution<> vel(-0.01, 0.01);
  std::uniform_real_distribution<> massdist(1e20, 1e25);

  for (int i = 0; i < nbpart; ++i) {
      s.x[i] = pos(gen);
      s.y[i] = pos(gen);
      s.z[i] = pos(gen);
      s.vx[i] = vel(gen);
      s.vy[i] = vel(gen);
      s.vz[i] = vel(gen);
      s.mass[i] = massdist(gen);
  }

  run_simulation(s, nbstep, dt, G, printevery, threads);


  return 0;
}