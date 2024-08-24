#include <xfdtd/boundary/pml.h>
#include <xfdtd/monitor/field_monitor.h>
#include <xfdtd/monitor/movie_monitor.h>
#include <xfdtd/nffft/nffft_frequency_domain.h>
#include <xfdtd/simulation/simulation.h>
#include <xfdtd/waveform_source/tfsf_3d.h>

#include <ase_reader/ase_reader.hpp>
#include <filesystem>
#include <xfdtd_cuda/simulation/simulation_hd.cuh>
#include <xfdtd_model/grid_model.hpp>
#include <xfdtd_model/model_object.hpp>
#include <xfdtd_model/model_shape.hpp>
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
  program.add_argument("-ase").help("ASE file path").required();
  program.add_argument("-f_p", "--flow_field_path")
      .help("flow field data path")
      .required();
  program.add_argument("-r", "--resolution")
      .help("resolution")
      .default_value(20e-3)
      .action([](const std::string& value) { return std::stod(value); });
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

  const auto dl = program.get<xfdtd::Real>("--resolution");

  std::cout << "Resolution: " << dl << "\n";

  auto metal_info_ss = std::stringstream{};
  auto metal_vertices_ss = std::stringstream{};
  auto metal_elements_ss = std::stringstream{};

  {
    auto ase_path_str = program.get<std::string>("-ase");
    auto ase_path = std::filesystem::path{ase_path_str};
    if (!std::filesystem::exists(ase_path)) {
      std::cerr << "ASE file not found: " << ase_path_str << std::endl;
      exit(1);
    }

    auto ase_reader = ase_reader::ASEReader{};
    ase_reader.read(ase_path.string());
    ase_reader.setPrecision(8);

    {
      constexpr auto unit = xfdtd::unit::Length::Millimeter;
      const auto delta_l = xfdtd::model::ModelShape<unit>::standardToUnit(dl);
      for (const auto& o : ase_reader.objects()) {
        auto info_ss = std::stringstream{};
        auto vertices_ss = std::stringstream{};
        auto elements_ss = std::stringstream{};
        auto grid_model = xfdtd::model::GridModel{};
        o.write(info_ss, vertices_ss, elements_ss);
        grid_model.read(info_ss, vertices_ss, elements_ss);

        std::cout << "Object: " << o.name() << "\n";
        std::cout << "Region: " << "\n";
        std::cout << "  Origin: " << grid_model.triangularModelInfo().minX()
                  << " " << grid_model.triangularModelInfo().minY() << " "
                  << grid_model.triangularModelInfo().minZ() << "\n";
        std::cout << "  Size: " << grid_model.triangularModelInfo().sizeX()
                  << " " << grid_model.triangularModelInfo().sizeY() << " "
                  << grid_model.triangularModelInfo().sizeZ() << "\n";
        std::cout << " End: " << grid_model.triangularModelInfo().maxX() << " "
                  << grid_model.triangularModelInfo().maxY() << " "
                  << grid_model.triangularModelInfo().maxZ() << "\n";

        if (o.name() == "metal") {
          metal_info_ss << info_ss.str();
          metal_vertices_ss << vertices_ss.str();
          metal_elements_ss << elements_ss.str();
        }
      }

      auto info_ss = std::stringstream{};
      auto vertices_ss = std::stringstream{};
      auto elements_ss = std::stringstream{};
      ase_reader.write(info_ss, vertices_ss, elements_ss);
      auto grid_model = xfdtd::model::GridModel{};
      grid_model.read(info_ss, vertices_ss, elements_ss);
    }
  }

  auto model_shape = std::make_unique<
      xfdtd::model::ModelShape<xfdtd::unit::Length::Millimeter>>(
      metal_info_ss, metal_vertices_ss, metal_elements_ss);
  std::cout << "Model shape Wrapping: "
            << model_shape->wrappedCube()->toString() << "\n";

  auto model_object = std::make_shared<xfdtd::model::ModelObject>(
      "metal", std::move(model_shape), xfdtd::Material::createPec());

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
  s.addObject(model_object);

  constexpr auto l_min = 20e-3 * 20;
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