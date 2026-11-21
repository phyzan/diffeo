#pragma once


/**
 @file Contains implementations for the spatial part of black hole metrics in spherical coordinates.

 The metrics are implemented in 3D and not 4D, even though it would be equivalent, because all operations
 in the library will generally use contractions of spatial-only (tangent to the foliation) vectors,
 and the spatial part of the metric is required for that.

 To convert to the full 4D metric, as well as the full 4-vectors, the lapse function `alpha` and
 the shift `beta^i` are required, which are components of the full 4D metric.
*/


#include "diffeo/einsum/tensors.hpp"
#include <diffeo/coordinates/common.hpp>
#include <diffeo/coordinates/KerrCoords.hpp>
#include <diffeo/metric/metric_tensor.hpp>


namespace geom::adm{

template<typename T>
T sin_sqr(const T x){
    const T sin_x = sin(x);
    return sin_x * sin_x;
}

// ------------- 3D Kerr metric ---------------



template<typename T>
struct KerrBL3D : BaseMetric<KerrBL3D<T>, T, Spherical>{
    /*
    Using units where rg = 1, so that -1 <= a = Jc/(GM^2) <= 1
    */
    T alpha;

    using Base = BaseMetric<KerrBL3D<T>, T, Spherical>;
    using Coords = Coordinates<Spherical, T>;

    /// @brief Get g_ij(x)
    void fill_gcov(MetricTensor<T, Spherical>& out, const Coords& x) const {
        out.fastor().fill(T(0));
        const T sin2_th = sin_sqr(x.theta());
        out(0, 0) = g_rr(x.r(), sin2_th);
        out(1, 1) = g_thth(x.r(), sin2_th);
        out(2, 2) = g_phph(x.r(), sin2_th);
    }

    template<typename NewCoordSys>
    void fill_gcov(
        MetricTensor<T, NewCoordSys>& out,
        const Jacobian<T, Spherical, NewCoordSys>& dxdX,
        const Coordinates<Spherical, T>& x) const {

        Base::fill_gcov(out, dxdX, x);
    }

    /// @brief g^ij(x): g_ij is diagonal, so this is diag(1/g_rr, 1/g_thth, 1/g_phph)
    void fill_gcont(InverseMetricTensor<T, Spherical>& out, const Coords& x) const {
        const T sin2_th = sin_sqr(x.theta());
        out.fastor().fill(T(0));
        out(0, 0) = 1/g_rr(x.r(), sin2_th);
        out(1, 1) = 1/g_thth(x.r(), sin2_th);
        out(2, 2) = 1/g_phph(x.r(), sin2_th);
    }

    T gcov_det(const Coords& x) const {
        const T sin2_th = sin_sqr(x.theta());
        /*
        For the full 4D metric, this would be
        xdiff::power<2>(sigma(x.r(), sin2_th))*sin2_th;
        */
        return g_rr(x.r(), sin2_th) * g_thth(x.r(), sin2_th) * g_phph(x.r(), sin2_th);
    }

    // ----------------- 3+1 quantities ------------------

    /**
    @brief The lapse alpha = sqrt(Delta Sigma / A).
    Only real outside the outer horizon, where Delta > 0:
    inside it, the BL slices are no longer spacelike, and this returns NaN
    */
    T lapse(const Coords& x) const {
        const T sin2_th = sin_sqr(x.theta());
        return sqrt(delta(x.r())*sigma(x.r(), sin2_th)/big_a(x.r(), sin2_th));
    }

    /// @brief d_i(alpha): A covariant 3-vector (transforms properly under spatial transformations)
    void ddx_lapse(CoVector<T, 3>& out, const Coords& x) const {
        /*
        alpha^2 = Delta Sigma / A, so d(alpha) = alpha/2 (dDelta/Delta + dSigma/Sigma - dA/A), with
        d_r Delta = 2(r - 1), d_r Sigma = 2r, d_th Delta = 0, d_th Sigma = -a^2 sin(2 theta).
        The Delta term of d_r is written as (r - 1) Sigma/(alpha A): +inf on the horizon instead of 0*inf = NaN
        */
        const T r = x.r();
        const T sin_th = sin(x.theta());
        const T sin2_th = sin_th*sin_th;
        const T sin_2th = 2*sin_th*cos(x.theta());
        const T sig = sigma(r, sin2_th);
        const T del = delta(r);
        const T big = big_a(r, sin2_th);
        const T lap = sqrt(del*sig/big);

        out(0) = lap/2 * (2*r/sig - ddr_big_a(r, sin2_th)/big) + (r - 1)*sig/(lap*big);
        out(1) = lap/2 * (-alpha*alpha*sin_2th/sig - ddth_big_a(r, sin_2th)/big);
        out(2) = 0;
    }

    /**
    @brief @brief The contravariant shift beta^i = (0, 0, -2 a r / A),
    where -beta^phi is the frame-dragging angular velocity
    @note It transforms as a tensor (in pure spatial transformations),
    so we can interpret is as a contravariant spatial 3-vector
    */
    void shift(Vector<T, 3>& out, const Coords& x) const {
        const T sin2_th = sin_sqr(x.theta());
        out(1) = out(0) = 0;
        out(2) = -2*alpha*x.r()/big_a(x.r(), sin2_th);
    }

    /**
    @brief out(i, j) = d_i(beta^j). Only d_r and d_theta of beta^phi are nonzero:
    d_r beta^phi = -2a (A - r d_r A)/A^2, d_theta beta^phi = 2 a r d_theta A/A^2
    @note Not a tensor: under x -> x', d_i'(beta'^j) also has a term with the second derivatives of x'(x)
    */
    void ddx_shift(Tensor<T, Ax<Loc::Dn, 3>, Ax<Loc::Up, 3>>& out, const Coords& x) const{
        const T r = x.r();
        const T sin_th = sin(x.theta());
        const T sin2_th = sin_th*sin_th;
        const T sin_2th = 2*sin_th*cos(x.theta());
        const T big = big_a(r, sin2_th);
        const T big2 = big*big;

        out.fastor().fill(T(0));
        out(0, 2) = -2*alpha*(big - r*ddr_big_a(r, sin2_th))/big2;
        out(1, 2) = 2*alpha*r*ddth_big_a(r, sin_2th)/big2;
    }

    // ----------- Kerr-related function ------------

    T sigma(const T r, const T sin2_th) const {
        return r*r + alpha*alpha*(1-sin2_th);
    }

    T delta(const T r) const {
        return r*r - 2*r + alpha*alpha;
    }

    /// @brief A = (r^2 + a^2)^2 - a^2 Delta sin^2(theta)
    T big_a(const T r, const T sin2_th) const {
        const T term1 = r*r + alpha*alpha;
        return term1 * term1 - alpha*alpha*delta(r)*sin2_th;
    }

    /// @brief d_r A = 4r (r^2 + a^2) - 2 a^2 (r - 1) sin^2(theta)
    T ddr_big_a(const T r, const T sin2_th) const {
        return 4*r*(r*r + alpha*alpha) - 2*alpha*alpha*(r - 1)*sin2_th;
    }

    /// @brief d_theta A = -a^2 Delta sin(2 theta)
    T ddth_big_a(const T r, const T sin_2th) const {
        return -alpha*alpha*delta(r)*sin_2th;
    }

    // ---- Not really used for the 3D metric, but for completeness ----
    T g_tt(const T r, const T sin2_th) const {
        return 2*r/sigma(r, sin2_th) - 1;
    }

    T g_tphi(const T r, const T sin2_th) const {
        return -2*alpha*r*sin2_th/sigma(r, sin2_th);
    }
    // -----------------------------------------------------------------

    T g_rr(const T r, const T sin2_th) const {
        return sigma(r, sin2_th)/delta(r);
    }

    T g_thth(const T r, const T sin2_th) const {
        return sigma(r, sin2_th);
    }

    T g_phph(const T r, const T sin2_th) const {
        const auto a_sq = alpha * alpha;
        return (r * r + a_sq + 2*r*a_sq*sin2_th/sigma(r, sin2_th))*sin2_th;
    }
};



// --------- 3D Kerr in Spherical KS ----------

// The 3D Kerr metric in BL (classical spherical) coordinates cannot
// transform to Spherical KS, as the transformation mixes the time
// components too, and the 4D metrics are required.
// Thus, it is simpler to implement the Kerr metric in Spherical KS
// coordinates in 3D, which also directly transforms to MKS coordinates
// with a purely spatial transformation.

/*
On the ingoing Kerr-Schild slices, with M = 1 and z = 2r/Sigma:

    g_rr = 1 + z,   g_rphi = -a (1 + z) sin^2(theta),   g_thth = Sigma,
    g_phph = (r^2 + a^2 + z a^2 sin^2(theta)) sin^2(theta)

This is regular across the horizon, and is the metric BHAC uses for coord=mks, before its relabeling of r and theta.
*/
template<typename T>
struct KerrSphericalKS : BaseMetric<KerrSphericalKS<T>, T, KerrSchildSpherical>{
    T alpha; // spin, with -1 <= alpha <= 1

    using Base = BaseMetric<KerrSphericalKS<T>, T, KerrSchildSpherical>;
    using Coords = Coordinates<KerrSchildSpherical, T>;

    using Base::fill_gcov;

    /// @brief Get g_ij(x)
    void fill_gcov(MetricTensor<T, KerrSchildSpherical>& out, const Coords& x) const {
        const T sin2_th = sin_sqr(x.theta());
        const T sig = sigma(x.r(), sin2_th);
        const T one_z = 1 + 2*x.r()/sig;

        out.fastor().fill(T(0));
        out(0, 0) = one_z;
        out(2, 0) = -alpha*one_z*sin2_th;
        out(0, 2) = out(2, 0);
        out(1, 1) = sig;
        out(2, 2) = (x.r()*x.r() + alpha*alpha + (one_z - 1)*alpha*alpha*sin2_th)*sin2_th;
    }

    /// @brief det(g_ij) = Sigma^2 sin^2(theta) (1 + z), which is never negative
    T gcov_det(const Coords& x) const {
        const T sin2_th = sin_sqr(x.theta());
        const T sig = sigma(x.r(), sin2_th);
        return sin2_th*sig*(sig + 2*x.r());
    }

    // ----------------- 3+1 quantities ------------------

    /// @brief The lapse alpha = 1/sqrt(1 + z)
    T lapse(const Coords& x) const {
        const T sin2_th = sin_sqr(x.theta());
        return 1/sqrt(1 + 2*x.r()/sigma(x.r(), sin2_th));
    }

    /// @brief d_i(alpha): A covariant 3-vector (transforms properly under spatial transformations)
    void ddx_lapse(CoVector<T, 3>& out, const Coords& x) const {
        // alpha = (1 + z)^(-1/2), so d(alpha) = -alpha^3/2 dz
        T z, dr_z, dth_z;
        z_and_ddx(z, dr_z, dth_z, x);
        const T lap = 1/sqrt(1 + z);
        const T c = -lap*lap*lap/2;
        out(0) = c*dr_z;
        out(1) = c*dth_z;
        out(2) = 0;
    }

    /**
    @brief The contravariant shift beta^i = (z/(1 + z), 0, 0)
    @note It transforms as a tensor (in pure spatial transformations),
    so we can interpret is as a contravariant spatial 3-vector
    */
    void shift(Vector<T, 3>& out, const Coords& x) const {
        const T sin2_th = sin_sqr(x.theta());
        const T z = 2*x.r()/sigma(x.r(), sin2_th);
        out(0) = z/(1 + z);
        out(2) = out(1) = 0;
    }

    /**
    @brief out(i, j) = d_i(beta^j). Only d_r and d_theta of beta^r = z/(1 + z) are nonzero: d(beta^r) = dz/(1 + z)^2
    @note Not a tensor: under x -> x', d_i'(beta'^j) also has a term with the second derivatives of x'(x)
    */
    void ddx_shift(Tensor<T, Ax<Loc::Dn, 3>, Ax<Loc::Up, 3>>& out, const Coords& x) const{
        T z, dr_z, dth_z;
        z_and_ddx(z, dr_z, dth_z, x);
        const T c = 1/((1 + z)*(1 + z));
        out.fastor().fill(T(0));
        out(0, 0) = c*dr_z;
        out(1, 0) = c*dth_z;
    }

    // ----------- Kerr-related functions ------------

    T sigma(const T r, const T sin2_th) const {
        return r*r + alpha*alpha*(1 - sin2_th);
    }

    /// @brief z = 2r/Sigma, d_r z = 2 (Sigma - 2r^2)/Sigma^2 and d_theta z = 2 a^2 r sin(2 theta)/Sigma^2
    void z_and_ddx(T& z, T& dr_z, T& dth_z, const Coords& x) const {
        const T r = x.r();
        const T sin_th = sin(x.theta());
        const T sig = sigma(r, sin_th*sin_th);
        const T sig2 = sig*sig;
        z = 2*r/sig;
        dr_z = 2*(sig - 2*r*r)/sig2;
        dth_z = 4*alpha*alpha*r*sin_th*cos(x.theta())/sig2;
    }
};

} // namespace geom::adm
