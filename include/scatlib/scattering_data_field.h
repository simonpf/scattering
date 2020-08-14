/** \file scattering_data.h
 *
 * Represents scalar data that is defined on the product
 * space of two solid angles.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATLIB_SCATTERING_DATA_FIELD__
#define __SCATLIB_SCATTERING_DATA_FIELD__

#include <scatlib/eigen.h>
#include <scatlib/sht.h>
#include <scatlib/interpolation.h>
#include <memory>
#include <cassert>

namespace scatlib {
namespace scattering_data {

enum class DataFormat { Gridded, Spectral, FullySpectral };
enum class DataType { Spherical, TotallyRandom, AzimuthallyRandom, General };


class ScatteringDataFieldBase {
  static DataType determine_type(size_t n_lon_inc,
                                 size_t n_lat_inc,
                                 size_t n_lon_scat,
                                 size_t n_lat_scat) {
    if ((n_lon_inc == 1) && (n_lat_inc == 1) &&
        (n_lon_scat == 1) && (n_lat_scat == 1)) {
      return DataType::Spherical;
    }
    if ((n_lon_inc == 1) && (n_lat_inc == 1) &&
        (n_lon_scat == 1)) {
      return DataType::TotallyRandom;
    }
    if (n_lon_inc == 1) {
      return DataType::AzimuthallyRandom;
    }
    return DataType::General;
  }

public:
  ScatteringDataFieldBase(size_t n_lon_inc,
                          size_t n_lat_inc,
                          size_t n_lon_scat,
                          size_t n_lat_scat)
      : n_lon_inc_(n_lon_inc),
        n_lat_inc_(n_lat_inc),
        n_lon_scat_(n_lon_scat),
        n_lat_scat_(n_lat_scat),
        type_(determine_type(n_lon_inc, n_lat_inc, n_lon_scat, n_lat_scat)) {}

  DataType get_type() const { return type_; }

 protected:
  size_t n_lon_inc_;
  size_t n_lat_inc_;
  size_t n_lon_scat_;
  size_t n_lat_scat_;
  DataType type_;
};

template <typename Scalar, DataFormat type>
    class ScatteringDataField;

// pxx :: export
// pxx :: instance("ScatteringDataFieldGridded", ["double"])
template <typename Scalar>
    class ScatteringDataField<Scalar, DataFormat::Gridded> : public ScatteringDataFieldBase {

public:

    using ScatteringDataFieldBase::n_lon_inc_;
    using ScatteringDataFieldBase::n_lat_inc_;
    using ScatteringDataFieldBase::n_lon_scat_;
    using ScatteringDataFieldBase::n_lat_scat_;
    using ScatteringDataFieldBase::type_;
    using ScatteringDataFieldBase::get_type;

    using Vector = eigen::Vector<Scalar>;
    using VectorMap = eigen::VectorMap<Scalar>;
    using ConstVectorMap = eigen::ConstVectorMap<Scalar>;
    using Matrix = eigen::Matrix<Scalar>;
    using MatrixMap = eigen::MatrixMap<Scalar>;
    using ConstMatrixMap = eigen::ConstMatrixMap<Scalar>;
    using OneAngle = eigen::MatrixFixedRows<Scalar, 1>;
    using ThreeAngles = eigen::MatrixFixedRows<Scalar, 3>;
    using FourAngles = eigen::MatrixFixedRows<Scalar, 4>;

    template <eigen::Index rank>
    using Tensor = eigen::Tensor<Scalar, rank>;
    template <eigen::Index rank>
    using TensorMap = eigen::TensorMap<Scalar, rank>;
    template <eigen::Index rank>
      using ConstTensorMap = eigen::ConstTensorMap<Scalar, rank>;
    using DataTensor = eigen::Tensor<Scalar, 5>;

    // pxx :: hide
    ScatteringDataField(std::shared_ptr<Vector> lon_inc,
                        std::shared_ptr<Vector> lat_inc,
                        std::shared_ptr<Vector> lon_scat,
                        std::shared_ptr<Vector> lat_scat,
                        std::shared_ptr<Tensor<5>> data)
        : ScatteringDataFieldBase(lon_inc->size(),
                                  lat_inc->size(),
                                  lon_scat->size(),
                                  lat_scat->size()),
        lon_inc_(lon_inc),
        lat_inc_(lat_inc),
        lon_scat_(lon_scat),
        lat_scat_(lat_scat),
        lon_inc_map_(lon_inc->data(), n_lon_inc_),
        lat_inc_map_(lat_inc->data(), n_lat_inc_),
        lon_scat_map_(lon_scat->data(), n_lon_scat_),
        lat_scat_map_(lat_scat->data(), n_lat_scat_),
        data_(data)
    {
          }

      ScatteringDataField(const Vector &lon_inc,
                          const Vector &lat_inc,
                          const Vector &lon_scat,
                          const Vector &lat_scat,
                          const Tensor<5> &data)
          : ScatteringDataFieldBase(lon_inc.size(),
                                    lat_inc.size(),
                                    lon_scat.size(),
                                    lat_scat.size()),
      lon_inc_(std::make_shared<Vector>(lon_inc)),
      lat_inc_(std::make_shared<Vector>(lat_inc)),
      lon_scat_(std::make_shared<Vector>(lon_scat)),
      lat_scat_(std::make_shared<Vector>(lat_scat)),
      data_(std::make_shared<DataTensor>(data)),
      lon_inc_map_(lon_inc_->data(), n_lon_inc_),
      lat_inc_map_(lat_inc_->data(), n_lat_inc_),
      lon_scat_map_(lon_scat_->data(), n_lon_scat_),
      lat_scat_map_(lat_scat_->data(), n_lat_scat_) {

    }

      Scalar interpolate() const {
        assert(get_type() == DataType::Spherical);
        return (*data_)({0, 0, 0, 0, 0});
}

    Matrix interpolate(OneAngle &angles) const {
        assert(get_type() == DataType::TotallyRandom);

        Tensor<2> result{data_->dimension(0), angles.rows()};
        using Interpolator = RegularGridInterpolator<typename eigen::IndexResult<const DataTensor, 4>::type, 1, ConstVectorMap>;
        Interpolator interpolator({lat_scat_map_});
        auto weights = interpolator.calculate_weights(angles);
        auto interpolate = [&interpolator, &weights](const TensorMap<1> &out, const TensorMap<4> &in) {
            return interpolator.interpolate(out,
                                            eigen::tensor_index<3>(in, {0, 0, 0}),
                                            weights);
        };
        eigen::map_over_dimensions<1>(result, *data_, interpolate);
        return eigen::to_matrix_map(result).transpose();
    }

    Matrix interpolate(Vector &angles) const {
        return interpolate(angles);
    }

    Matrix interpolate(ThreeAngles &angles) const {
      assert(get_type() == DataType::AzimuthallyRandom);

      Tensor<2> result{data_->dimension(0), angles.rows()};
      using Interpolator = RegularGridInterpolator<
          typename eigen::IndexResult<const DataTensor, 2>::type,
          3,
          ConstVectorMap>;
      Interpolator interpolator({lat_inc_map_, lon_scat_map_, lat_scat_map_});
      auto weights = interpolator.calculate_weights(angles);
      auto interpolate = [&interpolator, &weights](const TensorMap<1> &out,
                                                   const TensorMap<4> &in) {
        return interpolator.interpolate(out,
                                        eigen::tensor_index<1>(in, {0}),
                                        weights);
      };
      eigen::map_over_dimensions<1>(result, *data_, interpolate);
      return eigen::to_matrix_map(result).transpose();
    }

    Matrix interpolate(FourAngles &angles) const {
        assert(get_type() == DataType::General);

        Tensor<2> result{data_->dimension(0), angles.rows()};
        using Interpolator = RegularGridInterpolator<
            typename eigen::IndexResult<DataTensor, 1>::type,
            4,
            ConstVectorMap>;
        Interpolator interpolator(
            {lon_inc_map_, lat_inc_map_, lon_scat_map_, lat_scat_map_});
        auto weights = interpolator.calculate_weights(angles);
        auto interpolate = [&interpolator, &weights](const TensorMap<1> &out,
                                                     const TensorMap<4> &in) {
            return interpolator.interpolate(out,
                                            in,
                                            weights);
        };
        eigen::map_over_dimensions<1>(result, *data_, interpolate);
        return eigen::to_matrix_map(result).transpose();
    }

protected:

    std::shared_ptr<Vector> lon_inc_;
    std::shared_ptr<Vector> lat_inc_;
    std::shared_ptr<Vector> lon_scat_;
    std::shared_ptr<Vector> lat_scat_;

    ConstVectorMap lon_inc_map_;
    ConstVectorMap lat_inc_map_;
    ConstVectorMap lon_scat_map_;
    ConstVectorMap lat_scat_map_;

    std::shared_ptr<Tensor<5>> data_;

};

}  // namespace scattering_data
}  // namespace scatlib

#endif
