#ifndef __FLOW_FIELD_H__
#define __FLOW_FIELD_H__

#include <xfdtd/material/dispersive_material.h>
#include <xfdtd/object/object.h>

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace xfdtd {

class ModelShape;

class XFDTDFlowFieldException : public xfdtd::XFDTDException {
 public:
  explicit XFDTDFlowFieldException(
      std::string message = "XFDTD Flow Field Exception")
      : XFDTDException{std::move(message)} {}
};

class FlowFieldEntry {
 public:
  FlowFieldEntry() = default;
  FlowFieldEntry(xfdtd::Vector position, xfdtd::Real omega_p, xfdtd::Real gamma)
      : _position{std::move(position)}, _omega_p{omega_p}, _gamma{gamma} {}

  auto position() const { return _position; }

  auto omegaP() const { return _omega_p; }

  auto gamma() const { return _gamma; }

 private:
  xfdtd::Vector _position;
  xfdtd::Real _omega_p{}, _gamma{};
};

class FlowFieldShape : public xfdtd::Shape {
 public:
  explicit FlowFieldShape(std::string_view flow_field_model_info_path);

  ~FlowFieldShape() override;

  auto clone() const -> std::unique_ptr<Shape> override { return {}; }

  auto isInside(xfdtd::Real x, xfdtd::Real y, xfdtd::Real z,
                xfdtd::Real eps) const -> bool override {
    return false;
  }

  auto isInside(const Vector& vector, Real eps) const -> bool override {
    return false;
  }

  auto wrappedCube() const -> std::unique_ptr<Cube> override;

  auto& flowFieldEntries() const { return _flow_field_entries; }

  auto& flowFieldEntries() { return _flow_field_entries; }

 private:
  std::string _flow_field_model_info_path;
  std::vector<FlowFieldEntry> _flow_field_entries;
  std::unique_ptr<Cube> _cube;

  auto readEntry(std::string_view path) const -> std::vector<FlowFieldEntry>;

  auto generateCube(const std::vector<FlowFieldEntry>& flow_field_entries) const
      -> std::unique_ptr<Cube>;
};

/**
 * @brief Read flow field
 * The format of the flow field is as follows:
 *  TITLE = "TEC3DS from TOPL3D at NT=10000, TAU=0.00000"
 *  VARIABLES = "X", "Y", "Z", "Fp", "Muc"
 *  ZONE T="Zone_1", I=70, J=108, K=60,
 *  F=POINT
 *    2.003403  0.2990603  0  4.773956e+10  8.218315e+09
 *    ...
 *
 */
class FlowField : public Object {
 public:
  FlowField(std::string_view name, std::string_view flow_field_model_info_path);

  ~FlowField() override;

  auto correctMaterialSpace(Index index) -> void override;

  auto handleDispersion(std::shared_ptr<ADEMethodStorage> ade_method_storage)
      -> void override;

  auto& flowFieldShape() const { return *_flow_field_shape; }

  auto& flowFieldShape() { return *_flow_field_shape; }

 private:
  FlowFieldShape* _flow_field_shape{};
};

}  // namespace xfdtd

#endif  // __FLOW_FIELD_H__
