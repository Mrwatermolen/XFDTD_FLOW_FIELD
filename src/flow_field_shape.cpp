#include <algorithm>
#include <filesystem>
#include <fstream>
#include <istream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "flow_field.h"

namespace xfdtd {

FlowFieldShape::FlowFieldShape(std::string_view flow_field_model_info_path)
    : _flow_field_model_info_path{flow_field_model_info_path},
      _flow_field_entries{readEntry(flow_field_model_info_path)},
      _cube{generateCube(_flow_field_entries)} {}

FlowFieldShape::~FlowFieldShape() = default;

auto FlowFieldShape::wrappedCube() const -> std::unique_ptr<Cube> {
  return std::make_unique<Cube>(_cube->origin(), _cube->size());
}

static auto& operator>>(std::istream& is, FlowFieldEntry& entry) {
  Real x{};
  Real y{};
  Real z{};
  Real omega_p{};
  Real gamma{};
  is >> x >> y >> z >> omega_p >> gamma;
  entry = FlowFieldEntry{Vector{x, y, z}, omega_p, gamma};
  return is;
}

auto FlowFieldShape::readEntry(std::string_view path) const
    -> std::vector<FlowFieldEntry> {
  auto file = std::ifstream{std::filesystem::path{path}};
  if (!file.is_open()) {
    std::stringstream ss;
    ss << "FlowFieldShape: readEntry(): can't open file: " << path;
    throw XFDTDFlowFieldException{ss.str()};
  }

  std::vector<FlowFieldEntry> entries{};

  // skip 4 line
  {
    std::string buffer;
    for (int i = 0; i < 4 && !file.eof(); ++i) {
      std::getline(file, buffer);
    }
  }

  while (!file.eof()) {
    FlowFieldEntry entry{};
    file >> entry;
    entries.emplace_back(entry);
  }

  return entries;
}

auto FlowFieldShape::generateCube(
    const std::vector<FlowFieldEntry>& flow_field_entries) const
    -> std::unique_ptr<Cube> {
  if (flow_field_entries.empty()) {
    std::stringstream ss;
    ss << "FlowFieldShape: generateCube(): data is empty";
    throw XFDTDFlowFieldException{ss.str()};
  }

  auto start_x = std::numeric_limits<Real>::max();
  auto start_y = std::numeric_limits<Real>::max();
  auto start_z = std::numeric_limits<Real>::max();
  auto end_x = std::numeric_limits<Real>::lowest();
  auto end_y = std::numeric_limits<Real>::lowest();
  auto end_z = std::numeric_limits<Real>::lowest();

  std::for_each(flow_field_entries.begin(), flow_field_entries.end(),
                [&](const auto& entry) {
                  const auto& p = entry.position();
                  // get start
                  start_x = std::min(start_x, p.x());
                  start_y = std::min(start_y, p.y());
                  start_z = std::min(start_z, p.z());
                  end_x = std::max(end_x, p.x());
                  end_y = std::max(end_y, p.y());
                  end_z = std::max(end_z, p.z());
                });

  return std::make_unique<Cube>(
      Vector{start_x, start_y, start_z},
      Vector{end_x - start_x, end_y - start_y, end_z - start_z});
}

}  // namespace xfdtd