#ifndef __SCATLIB_SHT__
#define __SCATLIB_SHT__

#include <fftw3.h>
#include <shtns.h>

namespace scatlib {

class SHT {
  STH(size_t l_max, size_t m_max, size_t n_lat, size_t n_phi, size_t m_res = 1)
      : l_max_(l_max),
        m_max_(m_max),
        n_lat_(n_lat),
        n_phi_(n_phi),
        m_res_(m_res) {
    // Nothing to do here.
  }

 private:
  size_t l_max_, m_max_, n_lat_, n_phi_, m_res_;
  shtns_cfg shtns_;
};

}  // namespace scatlib
