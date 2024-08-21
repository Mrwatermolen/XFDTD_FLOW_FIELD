#include "flow_field.h"

#include <xfdtd/material/ade_method/ade_method.h>
#include <xfdtd/material/dispersive_material.h>
#include <xfdtd/material/dispersive_material_equation/dispersive_material_equation.h>

#include <memory>
#include <random>

namespace xfdtd {

FlowField::FlowField(std::string_view name,
                     std::string_view flow_field_model_info_path)
    : Object{std::string(name),
             std::make_unique<FlowFieldShape>(flow_field_model_info_path),
             DrudeMedium::makeDrudeMedium("null", 1, Array1D<Real>{0},
                                          Array1D<Real>{0})},
      _flow_field_shape{dynamic_cast<FlowFieldShape*>(shapePtr())} {
  if (_flow_field_shape == nullptr) {
    throw XFDTDFlowFieldException{"FlowFieldShape is not FlowFieldShape"};
  }
}

FlowField::~FlowField() = default;

auto FlowField::correctMaterialSpace(Index index) -> void {
  setMaterialIndex(index);

  auto g_variety = std::const_pointer_cast<GridSpace>(gridSpace());

  auto nx = g_variety->sizeX();
  auto ny = g_variety->sizeY();
  auto nz = g_variety->sizeZ();

  const auto& grid_space = gridSpace();
  const auto& region =
      Cube{Vector{grid_space->eNodeX().front(), grid_space->eNodeY().front(),
                  grid_space->eNodeZ().front()},
           Vector{grid_space->eNodeX().back() - grid_space->eNodeX().front(),
                  grid_space->eNodeY().back() - grid_space->eNodeY().front(),
                  grid_space->eNodeZ().back() - grid_space->eNodeZ().front()}};

  const auto& flow_field_entries = _flow_field_shape->flowFieldEntries();

  for (const auto& e : flow_field_entries) {
    auto vec = e.position();
    if (!region.isInside(vec, 1e-6)) {
      continue;
    }

    auto g = grid_space->getGrid(vec);
    auto i = g.i();
    auto j = g.j();
    auto k = g.k();

    g_variety->gridWithMaterial()(i, j, k).setMaterialIndex(index);
  }
}

auto FlowField::handleDispersion(
    std::shared_ptr<ADEMethodStorage> ade_method_storage) -> void {
  const auto& grid_space = gridSpace();

  // random
  auto min_real = 8e7;
  auto max_real = 5e8;
  auto random_engine = std::mt19937{std::random_device{}()};
  auto dis = std::normal_distribution<Real>{};
  auto dice = [&dis, &random_engine, &min_real, &max_real]() {
    return std::clamp(dis(random_engine), min_real, max_real);
  };

  if (grid_space->type() != GridSpace::Type::UNIFORM) {
    throw XFDTDFlowFieldException{
        "handleDispersion(): Non-uniform grid space is not supported yet"};
  }

  const auto& flow_field_entries = _flow_field_shape->flowFieldEntries();
  const auto& region =
      Cube{Vector{grid_space->eNodeX().front(), grid_space->eNodeY().front(),
                  grid_space->eNodeZ().front()},
           Vector{grid_space->eNodeX().back() - grid_space->eNodeX().front(),
                  grid_space->eNodeY().back() - grid_space->eNodeY().front(),
                  grid_space->eNodeZ().back() - grid_space->eNodeZ().front()}};

  for (const auto& e : flow_field_entries) {
    auto vec = e.position();
    if (!region.isInside(vec, 1e-6)) {
      continue;
    }

    auto g = grid_space->getGrid(vec);
    auto i = g.i();
    auto j = g.j();
    auto k = g.k();
    if (grid_space->gridWithMaterial()(i, j, k).materialIndex() !=
        materialIndex()) {
      continue;
    }

    // why dose it divide by 6 and 8?
    // Because my memory is not enough to simulate the more accurate result.
    // auto omega_p = e.omegaP() / 6;
    // auto gamma = e.gamma() / 8;

    auto omega_p = dice();
    auto gamma = dice();

    auto dispersive_model = DrudeMedium::makeDrudeMedium(
        "null", 1, Array1D<Real>{omega_p}, Array1D<Real>{gamma});
    ade_method_storage->correctCoeff(i, j, k, *dispersive_model, gridSpace(),
                                     calculationParam());
  }
}

}  // namespace xfdtd