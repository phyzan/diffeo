#pragma once

#include "diffeo/einsum/tensors.hpp"
#include "diffeo/linalg/linalg.hpp"
#include <diffeo/coordinates/coordinates.hpp>

namespace geom{

template<typename T, typename CoordSys>
class MetricTensor;

template<typename T, typename CoordSys>
class InverseMetricTensor;

///@brief g_mu_nu
template<typename T, typename CoordSys>
class MetricTensor : Tensor<T, Ax<Loc::Dn, CoordSys::N>, Ax<Loc::Dn, CoordSys::N>>{
public:
    static constexpr size_t N = CoordSys::N;
    using Base = Tensor<T, Ax<Loc::Dn, CoordSys::N>, Ax<Loc::Dn, CoordSys::N>>;
    MetricTensor() = default;
    using Base::Base;

    using Base::operator();
    using Base::fastor;

    void get_inverse(InverseMetricTensor<T, CoordSys>& out) const{
        linalg::detail::invert_rowmajor<T, N>(out.fastor().data(), this->fastor().data());
    }
};

///@brief g^mu^nu
template<typename T, typename CoordSys>
class InverseMetricTensor : Tensor<T, Ax<Loc::Up, CoordSys::N>, Ax<Loc::Up, CoordSys::N>>{
public:
    static constexpr size_t N = CoordSys::N;
    using Base = Tensor<T, Ax<Loc::Up, CoordSys::N>, Ax<Loc::Up, CoordSys::N>>;
    InverseMetricTensor() = default;
    using Base::Base;

    using Base::operator();
    using Base::fastor;

    void get_inverse(MetricTensor<T, CoordSys>& out) const{
        linalg::detail::invert_rowmajor<T, N>(out.fastor().data(), this->fastor().data());
    }
};

///@brief Generator for MetricTensor, providing analytical expressions
template<typename Derived, typename T, typename CoordSystem>
struct BaseMetric{

    using CoordSys = CoordSystem;

    using Coords = Coordinates<CoordSys, T>;

    static constexpr size_t N = CoordSys::N;

    // =========== MUST OVERRIDDE ===========

    /// @brief Get g_ij(x)
    void fill_gcov(MetricTensor<T, CoordSys>& out, const Coords& x) const {
        static_assert(sizeof(T)==0, "Must override fill_gcov");
    }

    /*
    A derived fill_gcov hides this class's overload of g_ij(x'), so the derived class must also state

    using BaseMetric<Derived, CoordSys>::fill_gcov;
    */
    // ======================================

    // ==== MUST OVERRIDE, only if called ====

    /**
    @brief out_i = (d_i g_jk) u^j u^k at x, for a vector u that is held fixed.
    In the equations of motion of the covariant u_i, this gives the force of the spatial metric without
    the rank-3 d_i g_jk, as -(1/2) u_j u_k d_i g^jk = (1/2) u^j u^k d_i g_jk.
    @note Not a tensor, as d_i g_jk is not
    */
    void ddx_gcov_contracted(CoVector<T, N>& /*out*/, const Coords& /*x*/, const Vector<T, N>& /*u*/) const {
        static_assert(sizeof(T)==0, "Must override ddx_gcov_contracted");
    }
    // ======================================

    void fill_gcont(InverseMetricTensor<T, CoordSys>& out, const Coords& x) const {
        MetricTensor<T, CoordSys> gcov;
        static_cast<const Derived*>(this)->fill_gcov(gcov, x);
        linalg::detail::invert_rowmajor<T, N>(out.fastor().data(), gcov.fastor().data());
    }

    /// @brief The signed value of the determinant
    T gcov_det(const Coords& x) const {
        MetricTensor<T, CoordSys> gcov;
        static_cast<const Derived*>(this)->fill_gcov(gcov, x);
        return linalg::detail::det_rowmajor<T, N>(gcov.fastor().data());
    }

    /// @brief sqrt(|g|)
    T sqrt_gdet(const Coords& x) const {
        return sqrt(abs(static_cast<const Derived*>(this)->gcov_det(x)));
    }

    /**
    @brief Get g_ij(x').
    @param out Matrix to fill with g_ij in the new coordinates
    @param dxdX The jacobian dx/dx' at the same point
    @param x The coordinates to compute g_ij(x) at, in the original coordinate system
    */
    template<typename NewCoordSys>
    void fill_gcov(
        MetricTensor<T, NewCoordSys>& out,
        const Jacobian<T, CoordSys, NewCoordSys>& dxdX,
        const Coordinates<CoordSys, T>& x) const {

        MetricTensor<T, CoordSys> gcov;
        static_cast<const Derived*>(this)->fill_gcov(gcov, x);

        const Idx<0> mu; const Idx<1> nu; const Idx<2> i; const Idx<3> j;

        // Using the dxdX_T is not necessary, but helps Fastor
        // vectorize the loops
        Tensor<T, Ax<Loc::Dn, N>, Ax<Loc::Up, N>> dxdX_T;
        dxdX_T(i, j) = dxdX(j, i);
        out(mu, nu) = dxdX_T(mu, i) * gcov(i, j) * dxdX(j, nu);
    }

};

} // namespace geom