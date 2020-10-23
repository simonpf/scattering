#include <scatlib/sht.h>

namespace scatlib {
namespace sht {

/** Deleter functional for smart pointers. */
struct FFTWDeleter {
  template <typename T>
  void operator()(T *t) {
    if (t) {
      fftw_free(t);
    }
  }
};

    template <typename Numeric>
    FFTWArray<Numeric>::FFTWArray(Index n)
    : ptr_(reinterpret_cast<Numeric *>(fftw_malloc(n * sizeof(Numeric))),
           FFTWDeleter()) {}

shtns_cfg ShtnsHandle::get(Index l_max,
                           Index m_max,
                           Index n_lon,
                           Index n_lat) {
  std::array<Index, 4> config = {l_max, m_max, n_lon, n_lat};
  if (config == current_config_) {
    return shtns_;
  } else {
    shtns_reset();
    shtns_ = shtns_init(sht_reg_fast, l_max, m_max, 1, n_lat, n_lon);
    current_config_ = config;
  }
  return shtns_;
}

shtns_cfg ShtnsHandle::shtns_ = nullptr;
std::array<Index, 4> ShtnsHandle::current_config_ = {-1, -1, -1, -1};

////////////////////////////////////////////////////////////////////////////////
// SHT
////////////////////////////////////////////////////////////////////////////////

SpectralCoeffs SHT::add_coeffs(const SHT &sht_l,
                               SpectralCoeffsRef v,
                               const SHT &sht_r,
                               SpectralCoeffsRef w) {
  auto result = SpectralCoeffs(v);

  if (sht_r.is_trivial_) {
    result[0] += w[0];
    return result;
  }

  Index m_max_min = std::min(sht_l.m_max_, sht_r.m_max_);
  Index l_max_min = std::min(sht_l.l_max_, sht_r.l_max_);
  for (Index m = 0; m <= m_max_min; ++m) {
    Index index_r = m * (sht_r.l_max_ + 1) - (m * (m - 1)) / 2;
    Index index_l = m * (sht_l.l_max_ + 1) - (m * (m - 1)) / 2;
    for (Index l = m; l <= l_max_min; ++l) {
      result[index_l] += w[index_r];
      ++index_r;
      ++index_l;
    }
  }
  return result;
}

SpectralCoeffMatrix SHT::add_coeffs(const SHT &sht_inc_l,
                                    const SHT &sht_scat_l,
                                    SpectralCoeffMatrixRef v,
                                    const SHT &sht_inc_r,
                                    const SHT &sht_scat_r,
                                    SpectralCoeffMatrixRef w) {
  Index nlm_inc = sht_inc_l.get_n_spectral_coeffs_cmplx();
  Index nlm_scat = sht_scat_l.get_n_spectral_coeffs();
  auto result = SpectralCoeffMatrix(nlm_inc, nlm_scat);

  Index index_l = 0;
  for (int l = 0; l <= (int)sht_inc_l.l_max_; ++l) {
    int m_max = (l <= (int)sht_inc_l.m_max_) ? l : sht_inc_l.m_max_;
    for (int m = -m_max; m <= m_max; ++m) {
      if ((l > sht_inc_r.l_max_) || (std::abs(m) > sht_inc_r.m_max_)) {
        result.row(index_l) = v.row(index_l);
      } else {
        int h = std::min<int>(sht_inc_r.m_max_, l);
        int index_r = l * (2 * h + 1) - h * h + m;

        auto r =
            add_coeffs(sht_scat_l, v.row(index_l), sht_scat_r, w.row(index_r));
        result.row(index_l) = r;
      }
      ++index_l;
    }
  }
  return result;
}

SHT::Vector SHT::get_latitude_grid(Index n_lat) {
  Vector result(n_lat);
  for (Index i = 0; i < n_lat; ++i) {
    result[i] = M_PI / n_lat * (i + 0.5);
  }
  return result;
}

SHT::Vector SHT::get_longitude_grid(Index n_lon) {
  if (n_lon == 1) {
    return Vector::Constant(1, M_PI);
  }
  Vector result(n_lon);
  double dx = 2.0 * M_PI / n_lon;
  for (Index i = 0; i < n_lon; ++i) {
    result[i] = dx * i;
  }
  return result;
}

std::array<Index, 4> SHT::get_params(Index n_lon, Index n_lat) {
  n_lon -= n_lon % 2;
  n_lat -= n_lat % 2;

  Index l_max = (n_lat > 2) ? (n_lat / 2) - 1 : 0;
  Index m_max = (n_lon > 2) ? (n_lon / 2) - 1 : 0;
  m_max = std::min(l_max, m_max);
  return {l_max, m_max, n_lat, n_lon};
}

SHT::SHT(Index l_max, Index m_max, Index n_lon, Index n_lat)
    : l_max_(l_max), m_max_(m_max), n_lon_(n_lon), n_lat_(n_lat) {
  if (l_max == 0) {
    is_trivial_ = true;
    n_spectral_coeffs_ = 1;
    n_spectral_coeffs_cmplx_ = 1;
  } else {
    is_trivial_ = false;
    shtns_verbose(1);
    shtns_use_threads(0);
    n_spectral_coeffs_ = calc_n_spectral_coeffs(l_max, m_max);
    n_spectral_coeffs_cmplx_ = calc_n_spectral_coeffs_cmplx(l_max, m_max);
    spectral_coeffs_ = sht::FFTWArray<std::complex<double>>(n_spectral_coeffs_);
    spectral_coeffs_cmplx_ =
        sht::FFTWArray<std::complex<double>>(n_spectral_coeffs_cmplx_);
    spatial_coeffs_ = sht::FFTWArray<double>(n_lon * n_lat);
    cmplx_spatial_coeffs_ = sht::FFTWArray<std::complex<double>>(n_lon * n_lat);
  }
}

SHT::Vector SHT::get_latitude_grid() {
  if (is_trivial_) {
    return Vector::Constant(1, M_PI / 2.0);
  }
  auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lon_, n_lat_);
  return ConstVectorMap(shtns->ct, n_lat_).array().acos();
}

SHT::Vector SHT::get_colatitude_grid() {
  if (is_trivial_) {
    return Vector::Constant(1, M_PI / 2.0);
  }
  auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lon_, n_lat_);
  return ConstVectorMap(shtns->ct, n_lat_);
}

SHT::Vector SHT::get_longitude_grid() {
  if (is_trivial_) {
    return Vector::Constant(1, M_PI);
  }
  Vector v{n_lon_};
  double dx = 2.0 * M_PI / n_lon_;
  for (Index i = 0; i < n_lon_; ++i) {
    v[i] = dx * i;
  }
  return v;
}

SHT::IndexVector SHT::get_l_indices() {
  if (is_trivial_) {
    return IndexVector::Constant(1, 0);
  }
  auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lon_, n_lat_);
  IndexVector result(n_spectral_coeffs_);
  for (Index i = 0; i < n_spectral_coeffs_; ++i) {
    result[i] = shtns->li[i];
  }
  return result;
}

SHT::IndexVector SHT::get_m_indices() {
  if (is_trivial_) {
    return IndexVector::Constant(1, 0);
  }
  auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lon_, n_lat_);
  IndexVector result(n_spectral_coeffs_);
  for (Index i = 0; i < n_spectral_coeffs_; ++i) {
    result[i] = shtns->mi[i];
  }
  return result;
}

void SHT::set_spatial_coeffs(const GridCoeffsRef &m) const {
  Index index = 0;
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      spatial_coeffs_[index] = m(i, j);
      ++index;
    }
  }
}

void SHT::set_spatial_coeffs(const CmplxGridCoeffsRef &m) const {
  Index index = 0;
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      cmplx_spatial_coeffs_[index] = m(i, j);
      ++index;
    }
  }
}

void SHT::set_spectral_coeffs(const SpectralCoeffsRef &m) const {
  Index index = 0;
  for (auto &x : m) {
    spectral_coeffs_[index] = x;
    ++index;
  }
}

void SHT::set_spectral_coeffs_cmplx(const SpectralCoeffsRef &m) const {
  Index index = 0;
  for (auto &x : m) {
    spectral_coeffs_cmplx_[index] = x;
    ++index;
  }
}

GridCoeffs SHT::get_spatial_coeffs() const {
  GridCoeffs result(n_lon_, n_lat_);
  Index index = 0;
  for (int i = 0; i < result.rows(); ++i) {
    for (int j = 0; j < result.cols(); ++j) {
      result(i, j) = spatial_coeffs_[index];
      ++index;
    }
  }
  return result;
}

CmplxGridCoeffs SHT::get_cmplx_spatial_coeffs() const {
  CmplxGridCoeffs result(n_lon_, n_lat_);
  Index index = 0;
  for (int i = 0; i < result.rows(); ++i) {
    for (int j = 0; j < result.cols(); ++j) {
      result(i, j) = cmplx_spatial_coeffs_[index];
      ++index;
    }
  }
  return result;
}

SpectralCoeffs SHT::get_spectral_coeffs() const {
  SpectralCoeffs result(n_spectral_coeffs_);
  Index index = 0;
  for (auto &x : result) {
    x = spectral_coeffs_[index];
    ++index;
  }
  return result;
}

SpectralCoeffs SHT::get_spectral_coeffs_cmplx() const {
  SpectralCoeffs result(n_spectral_coeffs_cmplx_);
  Index index = 0;
  for (auto &x : result) {
    x = spectral_coeffs_cmplx_[index];
    ++index;
  }
  return result;
}

SpectralCoeffs SHT::transform(const GridCoeffsRef &m) {
  if (is_trivial_) {
    return SpectralCoeffs::Constant(1, m(0, 0));
  }
  set_spatial_coeffs(m);
  auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lon_, n_lat_);
  spat_to_SH(shtns, spatial_coeffs_, spectral_coeffs_);
  return get_spectral_coeffs();
}

SpectralCoeffs SHT::transform_cmplx(const CmplxGridCoeffsRef &m) {
  if (is_trivial_) {
    return SpectralCoeffs::Constant(1, m(0, 0));
  }
  set_spatial_coeffs(m);
  auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lon_, n_lat_);
  spat_cplx_to_SH(shtns, cmplx_spatial_coeffs_, spectral_coeffs_cmplx_);
  return get_spectral_coeffs_cmplx();
}

GridCoeffs SHT::synthesize(const SpectralCoeffsRef &m) {
  if (is_trivial_) {
    return GridCoeffs::Constant(1, 1, m(0, 0).real());
  }
  set_spectral_coeffs(m);
  auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lon_, n_lat_);
  SH_to_spat(shtns, spectral_coeffs_, spatial_coeffs_);
  return get_spatial_coeffs();
}

CmplxGridCoeffs SHT::synthesize_cmplx(const SpectralCoeffsRef &m) {
  if (is_trivial_) {
    return CmplxGridCoeffs::Constant(1, 1, m(0, 0).real());
  }
  set_spectral_coeffs_cmplx(m);
  auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lon_, n_lat_);
  SH_to_spat_cplx(shtns, spectral_coeffs_cmplx_, cmplx_spatial_coeffs_);
  return get_cmplx_spatial_coeffs();
}

eigen::Vector<double> SHT::evaluate(
    const SpectralCoeffsRef &m,
    const eigen::MatrixFixedRows<double, 2> &points) {
  if (is_trivial_) {
    return eigen::Vector<double>::Constant(1, 1, m(0, 0).real());
  }
  set_spectral_coeffs(m);
  int n_points = points.rows();
  eigen::Vector<double> result(n_points);
  auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lon_, n_lat_);
  for (int i = 0; i < n_points; ++i) {
    result[i] =
        SH_to_point(shtns, spectral_coeffs_, cos(points(i, 1)), points(i, 0));
  }
  return result;
}

eigen::Vector<double> SHT::evaluate(const SpectralCoeffsRef &m,
                                    const eigen::Vector<double> &thetas) {
  if (is_trivial_) {
    return eigen::Vector<double>::Constant(1, 1, m(0, 0).real());
  }
  assert(m_max_ == 0);
  set_spectral_coeffs(m);
  int n_points = thetas.size();
  eigen::Vector<double> result(n_points);
  auto shtns = ShtnsHandle::get(l_max_, m_max_, n_lon_, n_lat_);
  for (int i = 0; i < n_points; ++i) {
    result[i] = SH_to_point(shtns, spectral_coeffs_, cos(thetas[i]), 0.0);
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// SHTProvider
////////////////////////////////////////////////////////////////////////////////

SHTProvider::SHTProvider(){};

SHT &SHTProvider::get_sht_instance(SHTProvider::SHTParams params) {
  if (sht_instances_.count(params) == 0) {
    sht_instances_[params] =
        std::make_unique<SHT>(params[0], params[1], params[2], params[3]);
  }
  return *sht_instances_[params];
}

}  // namespace sht
}  // namespace scatlib
