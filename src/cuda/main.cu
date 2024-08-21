#include <xfdtd/boundary/pml.h>
#include <xfdtd/monitor/field_monitor.h>
#include <xfdtd/monitor/movie_monitor.h>
#include <xfdtd/nffft/nffft_frequency_domain.h>
#include <xfdtd/simulation/simulation.h>
#include <xfdtd/waveform_source/tfsf_3d.h>

#include <filesystem>
#include <xfdtd_cuda/simulation/simulation_hd.cuh>
#include <xtensor/xnpy.hpp>

#include "argparse.hpp"
#include "flow_field.h"

int main(int argc, char** argv) {
  auto start_time = std::chrono::high_resolution_clock::now();

  xfdtd::MpiSupport::setMpiParallelDim(1, 2, 2);
  xfdtd::MpiSupport::instance(argc, argv);
  constexpr auto data_path_str = "./tmp/data/flow_field_cuda";
  const auto data_path = std::filesystem::path{data_path_str};

  auto program = argparse::ArgumentParser("flow_field_cuda");
  program.add_argument("-f_p", "--flow_field_path")
      .help("flow field data path")
      .required();
  program.add_argument("-g", "--cuda_grid_dim")
      .help("cuda grid dim")
      .default_value(std::vector<unsigned int>{128, 128, 2})
      .nargs(3)
      .action([](const std::string& value) { return std::stoi(value); });
  program.add_argument("-b", "--cuda_block_dim")
      .help("cuda block dim")
      .default_value(std::vector<unsigned int>{2, 2, 64})
      .nargs(3)
      .action([](const std::string& value) { return std::stoi(value); });
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    exit(0);
  }

  auto flow_field = std::make_shared<xfdtd::FlowField>(
      "flow_field", program.get<std::string>("--flow_field_path"));
  auto vector_to_dim = [](const auto& vec) {
    dim3 dim;
    dim.x = vec[0];
    dim.y = vec[1];
    dim.z = vec[2];
    return dim;
  };
  auto grid_dim =
      vector_to_dim(program.get<std::vector<int>>("--cuda_grid_dim"));
  auto block_dim =
      vector_to_dim(program.get<std::vector<int>>("--cuda_block_dim"));

  auto&& shape = flow_field->flowFieldShape();
  auto&& cube = shape.wrappedCube();

  constexpr xfdtd::Real dl{20e-3};

  auto domain_shape = xfdtd::Cube{
      xfdtd::Vector{-10 * dl + cube->originX(), -10 * dl + cube->originY(),
                    -10 * dl + cube->originZ()},
      xfdtd::Vector{20 * dl + cube->sizeX(), 20 * dl + cube->sizeY(),
                    20 * dl + cube->sizeZ()}};

  auto domain = std::make_shared<xfdtd::Object>(
      "domain", std::make_unique<xfdtd::Cube>(domain_shape),
      xfdtd::Material::createAir());

  auto s = xfdtd::Simulation{dl, dl, dl, 0.9, xfdtd::ThreadConfig{1, 1, 1}};
  s.addObject(domain);
  s.addObject(flow_field);

  constexpr auto l_min = dl * 20;
  constexpr auto f_max = 3e8 / l_min;
  constexpr auto tau = l_min / 6e8;
  constexpr auto t_0 = 4.5 * tau;
  constexpr xfdtd::Index tfsf_start = 13;
  auto tfsf{std::make_shared<xfdtd::TFSF3D>(
      tfsf_start, tfsf_start, tfsf_start, xfdtd::constant::PI / 2, 0, 1,
      xfdtd::Waveform::gaussian(tau, t_0))};

  s.addWaveformSource(tfsf);

  auto movie_ez_xy{std::make_shared<xfdtd::MovieMonitor>(
      std::make_unique<xfdtd::FieldMonitor>(
          std::make_unique<xfdtd::Cube>(
              xfdtd::Vector{domain_shape.originX(), domain_shape.originY(), 0},
              xfdtd::Vector{domain_shape.sizeX(), domain_shape.sizeY(), dl}),
          xfdtd::EMF::Field::EZ, "", ""),
      20, "movie_ez_xy", (data_path / "movie_ez_xy").string())};

  s.addMonitor(movie_ez_xy);

  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::XN));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::XP));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::YN));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::YP));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::ZN));
  s.addBoundary(std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::ZP));

  auto nf2ff_fd = std::make_shared<xfdtd::NFFFTFrequencyDomain>(
      11, 11, 11, xfdtd::Array<xfdtd::Real>{0.8 * f_max});
  s.addNF2FF(nf2ff_fd);

  auto s_hd = xfdtd::cuda::SimulationHD{&s};
  s_hd.setGridDim(grid_dim);
  s_hd.setBlockDim(block_dim);
  s_hd.run(1000);

  nf2ff_fd->setOutputDir((data_path / "fd").string());
  nf2ff_fd->processFarField(
      xfdtd::constant::PI * 0.5,
      xt::linspace<double>(-xfdtd::constant::PI, xfdtd::constant::PI, 360),
      "xy");

  auto time = tfsf->waveform()->time();
  auto incident_wave_data = tfsf->waveform()->value();
  if (!xfdtd::MpiSupport::instance().isRoot()) {
    return 0;
  }
  xt::dump_npy((data_path / "time.npy").string(), time);
  xt::dump_npy((data_path / "incident_wave.npy").string(), incident_wave_data);

  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - start_time)
                         .count();

  std::stringstream ss;
  ss << "Elapsed time: " << duration_ms << " ms" << " " << duration_ms / 1000.0
     << " s" << " " << duration_ms / 60000.0 << " min";

  std::cout << ss.str() << "\n";

  return 0;
}