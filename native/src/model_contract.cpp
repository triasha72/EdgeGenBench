#include "edgegenbench/runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace edgegenbench {
namespace {
constexpr std::array<float, 6> kFeatureMean{64.9352417F, 950.402283F, 534.764587F,
                                            526.701416F, 0.574177563F, 0.243009508F};
constexpr std::array<float, 6> kFeatureScale{14.4065495F, 318.028931F, 66.4537277F,
                                             130.141174F, 0.072084941F, 0.215282857F};
constexpr std::array<float, 6> kTargetMean{29978.1504F, 13540.7588F, 0.22503832F,
                                           2065.57275F, 2130.65796F, 85.2433472F};
constexpr std::array<float, 6> kTargetScale{7466.43115F, 5245.73193F, 0.0403823331F,
                                            1457.17334F, 697.039856F, 3.20977306F};
constexpr std::array<const char*, 4> kCategories{
    "conventional_turboprop", "fuel_cell_electric", "parallel_hybrid", "series_hybrid"};
constexpr std::array<const char*, 10> kInputs{
    "passenger_capacity", "design_range_km", "cruise_speed_kmh",
    "battery_specific_energy_wh_per_kg", "hydrogen_storage_efficiency",
    "hybridization_ratio", "propulsion_architecture=conventional_turboprop",
    "propulsion_architecture=fuel_cell_electric", "propulsion_architecture=parallel_hybrid",
    "propulsion_architecture=series_hybrid"};
constexpr std::array<const char*, 6> kOutputs{
    "estimated_takeoff_mass_kg", "mission_energy_kwh", "energy_per_passenger_km_kwh",
    "lifecycle_emissions_proxy_kgco2e", "operating_cost_proxy_usd", "noise_proxy_db"};
}  // namespace

const std::array<const char*, 10>& model_input_names() { return kInputs; }
const std::array<const char*, 6>& model_output_names() { return kOutputs; }

Tensor encode_aircraft_designs(const std::vector<AircraftDesign>& designs) {
  if (designs.empty()) throw RuntimeError("at least one aircraft design is required");
  if (designs.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    throw RuntimeError("aircraft design batch is too large");
  Tensor result{{static_cast<std::int64_t>(designs.size()), 10},
                std::vector<float>(designs.size() * 10, 0.0F)};
  for (std::size_t row = 0; row < designs.size(); ++row) {
    for (std::size_t column = 0; column < 6; ++column) {
      const float value = designs[row].numeric[column];
      if (!std::isfinite(value)) throw RuntimeError("aircraft design contains a non-finite value");
      result.data[row * 10 + column] = (value - kFeatureMean[column]) / kFeatureScale[column];
    }
    const auto category = std::find(kCategories.begin(), kCategories.end(),
                                    designs[row].propulsion_architecture);
    if (category == kCategories.end())
      throw RuntimeError("unknown propulsion architecture: " + designs[row].propulsion_architecture);
    result.data[row * 10 + 6 + static_cast<std::size_t>(category - kCategories.begin())] = 1.0F;
  }
  return result;
}

Tensor denormalize_predictions(const Tensor& normalized) {
  normalized.validate();
  if (normalized.shape.size() != 2 || normalized.shape[1] != 6)
    throw RuntimeError("normalized predictions must have shape [batch,6]");
  Tensor result{normalized.shape, std::vector<float>(normalized.data.size())};
  for (std::size_t index = 0; index < normalized.data.size(); ++index) {
    const auto column = index % 6;
    result.data[index] = normalized.data[index] * kTargetScale[column] + kTargetMean[column];
  }
  return result;
}
}  // namespace edgegenbench
