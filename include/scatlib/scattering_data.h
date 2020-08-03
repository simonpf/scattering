/** \file scattering_data.h
 *
 * Provides the ScatteringData class providing methods for the manipulation
 * of scattering data.
 *
 * @author Simon Pfreundschuh, 2020
 */
#ifndef __SCATLIB_SCATTERING_DATA__
#define __SCATLIB_SCATTERING_DATA__

#include <scatlib/eigen.h>

template <Scalar, size_t rank>
using SpectralCoeffTensor = eigen::Tensor<std::complex<Scalar>, rank>;
template <Scalar, size_t rank>
using GriddedCoeffTensor = eigen::Tensor<Scalar>, rank>;


namespace scatlib {


enum class DataFormat { Gridded, Spectral };

namespace detail {

    class SHTProvider {
    public:
        using SHTParams = std::array<size_t, 4>;

        SHTProvider();

        SHT& get_sht_instance(SHTParams params) {
            if (sht_instances_.count(params) == 0) {
                sht_instances[params] = SHT(params[0], params[1], params[2], params[3]);
            }
            return sht_instances[params];
        }

    protected:
        std::map<SHTParams, SHT> sht_instances_;
    };

    template <typename Scalar, size_t rank>
    class ScatteringToSpectralTransformer {
      ScatteringToSpectralTransformer(const GriddedCoeffTensor<Scalar, rank> &t,
                                      SHT &sht)
          : in_(t),
            sht_(sht),
            n_lat_(t.dimension(-1)),
            n_lon_(t.dimension(-2)),
            nlm_scat_(sht.get_number_of_spectral_coefficients()) {}

      size_t get_major_stride_in() { return n_lat_ * n_lon_; }

      size_t get_major_stride_out() { return nlm_scat_; }

      eigen::MatrixMap<Scalar> get_input_sequence(size_t major_index) {
          size_t start = major_index * get_major_stride_in();
        return MatrixMap<Scalar>(in_.data, n_lon_, n_lat_);
      }

      eigen::VectorMap<std::complex<Scalar>> get_output_sequence(
          size_t major_index,
          SpectralCoeffTensor<Scalar, rank - 1> &out) {
          size_t start = major_index * get_major_stride_out();
        size_t stride = get_lm_stride_out();
        return VectorMap<Scalar>(out.data, nlm_inc_, {stride});
      }

      std::array<size_t, rank - 1> get_output_dimensions() {
        auto input_size = in.dimensions();
        std::array<size_t, rank - 1> output_dimensions;
        std::copy(input_dimensions.begin(), input_dimensions.end() - 1,
                  output_dimensions.begin());
        output_dimensions[rank - 1] = nlm;
        return output_dimensions;
      }

      SpectralCoeffTensor<Scalar, rank - 1> transform() {
        SpectralCoeffTensor<Scalar, rank - 1> out(get_output_dimensions());
        for (size_t i = 0; i < out.size() / get_major_stride(); ++i) {
            get_output_sequence(i, out) = get_input_sequence(i, 0);
        }
        return out;
      }

     protected:
      SpectralCoeffTensor<Scalar, rank> &in;
      SHT &sht;
      size_t n_lat_, n_lon_, nlm_scat_;
    };

    class IncomingToSpectralTransformer {
      IncomingToSpectralTransformer(const SpectralCoeffTensor<Scalar, rank> &t,
                                    SHT &sht)
          : in_(t),
            sht_(sht),
            n_lat_(t.dimension(-2)),
            n_lon_(t.dimension(-3)),
            nlm_scat_(t.dimension(-1)) {}

      size_t get_major_stride_in() { return n_lat_ * n_lon_ * nlm_scat_ * 2; }

      size_t get_major_stride_out() { return 2 * nlm_scat_ * nlm_inc_; }

      size_t get_lm_stride_in() { return nlm_scat_ * 2; }

      size_t get_lm_stride_out() { return nlm_scat_; }

      eigen::MatrixMap<Scalar> get_input_sequence(size_t major_index,
                                                  size_t lm_index,
                                                  size_t complex_index) {
        size_t start = major_index * get_major_stride_in() +
                       lm_index * get_lm_stride_in() + complex_index;
        size_t col_stride = 2 * get_lm_stride_in();
        size_t row_stride = 2 * get_lm_stride_in() * n_lat_;
        return MatrixMap<Scalar>(in_.data, n_lon_, n_lat_,
                                 {row_stride, col_stride});
      }

      eigen::VectorMap<std::complex<Scalar>> get_output_sequence(
          size_t major_index, size_t lm_index, size_t complex_index,
          SpectralCoeffTensor<Scalar, rank> &out) {
        size_t start = major_index * get_major_stride_out() +
                       complex_index * get_major_stride_out() / 2 +
                       lm_index * get_lm_stride_out();
        size_t stride = get_lm_stride_out();
        return VectorMap<Scalar>(out.data, nlm_inc_, {stride});
      }

      std::array<size_t, rank> get_output_dimensions() {
        auto input_size = in.dimensions();
        std::array<size_t, rank> output_dimensions;
        std::copy(input_dimensions.begin(), input_dimensions.end(),
                  output_dimensions.begin());
        output_dimensions[rank - 3] = nlm;
        output_dimensions[rank - 4] = 2;
        return output_dimensions;
      }

      SpectralCoeffTensor<Scalar, rank> transform() {
        SpectralCoeffTensor<Scalar, rank> out(get_output_dimensions());
        for (size_t i = 0; i < out.size() / get_major_stride(); ++i) {
          for (size_t j = 0; j < nlm_scat_; ++j) {
            get_output_sequence(i, j, 0, out) = get_input_sequence(i, j, 0);
            get_output_sequence(i, j, 1, out) = get_input_sequence(i, j, 1);
          }
        }
        return out;
      }

     protected:
      SpectralCoeffTensor &in;
      SHT &sht;
      size_t n_lat_, n_lon_, nlm_scat_;
    };
}

template <typename Scalar>
class ScatteringData<DataFormat::Gridded, DataFormat::Gridded> {
 public:
    using CoeffType = Scalar;

 protected:

    eigen::Tensor<CoeffType, 7> phase_matrix_data; // f x T x elements x (inc. angles) x (scat. angles)
    eigen::Tensor<CoeffType, 7> extinction_matrix_data; // f x T x elements x (inc. angles) x (scat. angles)
    eigen::Tensor<CoeffType, 7> absorption_vector_data; // f x T x elements x (inc. angles) x (scat. angles)
    eigen::Tensor<CoeffType, 6> backscattering_coefficient; // f x T x elements x (inc. angles) x (scat. angles)
    eigen::Tensor<CoeffType, 6> forwardscattering_coefficient; // f x T x elements x (inc. angles) x (scat. angles)

};

template <typename Scalar>
    class ScatteringData<DataFormat::Gridded, DataFormat::Spectral> {
public:
    using CoeffType = std::complex<Scalar>;

protected:

    eigen::Tensor<CoeffType, 6> phase_matrix_data; // f x T x elements x (inc. angles) x nlm
    eigen::Tensor<CoeffType, 6> extinction_matrix_data; // f x T x elements x (inc. angles) x nlm
    eigen::Tensor<CoeffType, 6> absorption_vector_data; // f x T x elements x (inc. angles) x nlm
    eigen::Tensor<CoeffType, 5> backscattering_coefficient; // f x T x elements x (inc. angles) x nlm
    eigen::Tensor<CoeffType, 5> forwardscattering_coefficient; // f x T x elements x (inc. angles) x nlm

};

template <typename Scalar>
    class ScatteringData<DataFormat::Spectral, DataFormat::Spectral> {
public:
    using CoeffType = std::complex<Scalar>;

protected:

    eigen::Tensor<CoeffType, 7> phase_matrix_data; // f x T x elements x (nlm_inc) x nlm x 2
    eigen::Tensor<CoeffType, 7> extinction_matrix_data; // f x T x elements x (nlm_inc) x nlm x 2
    eigen::Tensor<CoeffType, 7> absorption_vector_data; // f x T x elements x (nlm_inc) x nlm x 2
    eigen::Tensor<CoeffType, 6> backscattering_coefficient; // f x T x elements x (nlm_inc) x nlm x 2
    eigen::Tensor<CoeffType, 6> forwardscattering_coefficient; // f x T x elements x (nlm_inc) x nlm x 2

};

}  // namespace scatlib

#endif
