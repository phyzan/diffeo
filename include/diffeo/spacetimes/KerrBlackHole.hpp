#pragma once


/**
 @file Contains implementations for the spatial part of the Kerr metric, in Boyer-Lindquist coordinates
 and in the Kerr-Schild family of coordinates (SKS3D, MKS3D and CKS3D).

 The metrics are implemented in 3D and not 4D, even though it would be equivalent, because all operations
 in the library will generally use contractions of spatial-only (tangent to the foliation) vectors,
 and the spatial part of the metric is required for that.

 To convert to the full 4D metric, as well as the full 4-vectors, the lapse and the shift beta^i
 (lapse() and shift()) are required, which are components of the full 4D metric.

 Here alpha is the spin, in the code and in the comments, and the lapse is written as `lapse`.
*/

#include <array>
#include <diffeo/diffeo.hpp>


namespace geom::adm{

template<typename T>
T sin_sqr(const T x){
    const T sin_x = sin(x);
    return sin_x * sin_x;
}

/// @brief What the equations of motion of a particle need from a 3+1 metric, at its position. See ADMMetric::fill_geometry
template<typename T, typename CoordSys>
struct ADMGeometry{
    InverseMetricTensor<T, CoordSys> gcont;                     // g^ij
    Tensor<T, Ax<Loc::Dn, 3>, Ax<Loc::Up, 3>> ddx_shift;        // d_i(beta^j)
    Vector<T, 3> u;                                             // g^ij u_j
    Vector<T, 3> shift;                                         // beta^i
    CoVector<T, 3> ddx_lapse;                                   // d_i(lapse)
    CoVector<T, 3> ddx_gcov_contracted;                         // (d_i g_jk) u^j u^k
    T lapse;
    T sqrt_gdet;
};

/**
 * @brief 3 + 1 split: Interpretes the metric as the spatial part of a 4D spacetime
 */
template<typename Derived, typename CoordSys, typename T>
struct ADMMetric : BaseMetric<Derived, T, CoordSys>{

    using Coords = Coordinates<CoordSys, T>;

    T lapse(const Coords& x) const {
        static_assert(sizeof(T) == 0, "lapse(x) not implemented");
        return 0;
    }

    /// @brief d_i(lapse): A covariant 3-vector (transforms properly under spatial transformations)
    void ddx_lapse(CoVector<T, 3>& out, const Coords& x) const {
        static_assert(sizeof(T) == 0, "d_i lapse(x) not implemented");
    }


    void shift(Vector<T, 3>& out, const Coords& x) const {
        static_assert(sizeof(T) == 0, "shift^i(x) not implemented");
    }


    void ddx_shift(Tensor<T, Ax<Loc::Dn, 3>, Ax<Loc::Up, 3>>& out, const Coords& x) const{
        static_assert(sizeof(T) == 0, "d_i shift^j(x) not implemented");
    }

    /// @brief All of ADMGeometry at once, for a covariant vector u_i.
    /// Override to reduce expensive evaluations
    void fill_geometry(ADMGeometry<T, CoordSys>& out, const Coords& x, const CoVector<T, 3>& co_u) const {
        constexpr Idx<0> i; constexpr Idx<1> j;
        const Derived& metric = *static_cast<const Derived*>(this);
        metric.fill_gcont(out.gcont, x);
        out.u(i) = out.gcont(i, j) * co_u(j);
        metric.shift(out.shift, x);
        metric.ddx_shift(out.ddx_shift, x);
        metric.ddx_lapse(out.ddx_lapse, x);
        metric.ddx_gcov_contracted(out.ddx_gcov_contracted, x, out.u);
        out.lapse = metric.lapse(x);
        out.sqrt_gdet = metric.sqrt_gdet(x);
    }
};

// ------------- 3D Kerr metric ---------------



template<typename T>
struct KerrBL3D : ADMMetric<KerrBL3D<T>, Spherical, T>{
    /*
    Using units where rg = 1, so that -1 <= alpha = Jc/(GM^2) <= 1
    */
    T alpha;

    using Base = ADMMetric<KerrBL3D<T>, Spherical, T>;
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

    /**
    @brief out_i = (d_i g_jk) u^j u^k. g_ij is diagonal and independent of phi, so that
    out_i = d_i(g_rr) (u^r)^2 + d_i(g_thth) (u^theta)^2 + d_i(g_phph) (u^phi)^2, and out_phi = 0
    */
    void ddx_gcov_contracted(CoVector<T, 3>& out, const Coords& x, const Vector<T, 3>& u) const {
        /*
        g_rr = Sigma/Delta, g_thth = Sigma and g_phph = A sin^2(theta)/Sigma, with
        d_r Sigma = 2r, d_r Delta = 2(r - 1), d_th Sigma = -alpha^2 sin(2 theta) and d_th Delta = 0
        */
        const T r = x.r();
        const T sin_th = sin(x.theta());
        const T sin2_th = sin_th*sin_th;
        const T sin_2th = 2*sin_th*cos(x.theta());
        const T sig = sigma(r, sin2_th);
        const T del = delta(r);
        const T big = big_a(r, sin2_th);
        const T dth_sig = -alpha*alpha*sin_2th;

        const T dr_grr = 2*(r*del - (r - 1)*sig)/(del*del);
        const T dr_gphph = sin2_th*(ddr_big_a(r, sin2_th) - 2*r*big/sig)/sig;
        const T dth_grr = dth_sig/del;
        const T dth_gphph = (ddth_big_a(r, sin_2th)*sin2_th + big*sin_2th - big*sin2_th*dth_sig/sig)/sig;

        const T ur2 = u(0)*u(0);
        const T uth2 = u(1)*u(1);
        const T uph2 = u(2)*u(2);
        out(0) = dr_grr*ur2 + 2*r*uth2 + dr_gphph*uph2;
        out(1) = dth_grr*ur2 + dth_sig*uth2 + dth_gphph*uph2;
        out(2) = 0;
    }

    // ----------------- 3+1 quantities ------------------

    /**
    @brief The lapse = sqrt(Delta Sigma / A).
    Only real outside the outer horizon, where Delta > 0:
    inside it, the BL slices are no longer spacelike, and this returns NaN
    */
    T lapse(const Coords& x) const {
        const T sin2_th = sin_sqr(x.theta());
        return sqrt(delta(x.r())*sigma(x.r(), sin2_th)/big_a(x.r(), sin2_th));
    }

    /// @brief d_i(lapse): A covariant 3-vector (transforms properly under spatial transformations)
    void ddx_lapse(CoVector<T, 3>& out, const Coords& x) const {
        /*
        lapse^2 = Delta Sigma / A, so d(lapse) = lapse/2 (dDelta/Delta + dSigma/Sigma - dA/A), with
        d_r Delta = 2(r - 1), d_r Sigma = 2r, d_th Delta = 0, d_th Sigma = -alpha^2 sin(2 theta).
        The Delta term of d_r is written as (r - 1) Sigma/(lapse A): +inf on the horizon instead of 0*inf = NaN
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
    @brief @brief The contravariant shift beta^i = (0, 0, -2 alpha r / A),
    where -beta^phi is the frame-dragging angular velocity
    @note It transforms as a tensor (in pure spatial transformations),
    so we can interpret it as a contravariant spatial 3-vector
    */
    void shift(Vector<T, 3>& out, const Coords& x) const {
        const T sin2_th = sin_sqr(x.theta());
        out(1) = out(0) = 0;
        out(2) = -2*alpha*x.r()/big_a(x.r(), sin2_th);
    }

    /**
    @brief out(i, j) = d_i(beta^j). Only d_r and d_theta of beta^phi are nonzero:
    d_r beta^phi = -2 alpha (A - r d_r A)/A^2, d_theta beta^phi = 2 alpha r d_theta A/A^2
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

    /// @brief A = (r^2 + alpha^2)^2 - alpha^2 Delta sin^2(theta)
    T big_a(const T r, const T sin2_th) const {
        const T term1 = r*r + alpha*alpha;
        return term1 * term1 - alpha*alpha*delta(r)*sin2_th;
    }

    /// @brief d_r A = 4r (r^2 + alpha^2) - 2 alpha^2 (r - 1) sin^2(theta)
    T ddr_big_a(const T r, const T sin2_th) const {
        return 4*r*(r*r + alpha*alpha) - 2*alpha*alpha*(r - 1)*sin2_th;
    }

    /// @brief d_theta A = -alpha^2 Delta sin(2 theta)
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
        const auto alpha_sq = alpha * alpha;
        return (r * r + alpha_sq + 2*r*alpha_sq*sin2_th/sigma(r, sin2_th))*sin2_th;
    }
};



// --------- 3D Kerr in Spherical KS ----------

// The 3D Kerr metric in BL (classical spherical) coordinates cannot
// transform to Spherical KS, as the transformation mixes the time
// components too, and the 4D metrics are required.
// Thus, it is simpler to implement the Kerr metric in Spherical KS
// coordinates in 3D, which also directly transforms to MKS and CKS coordinates
// with a purely spatial transformation.

/*
On the ingoing Kerr-Schild slices, with M = 1 and z = 2r/Sigma:

    g_rr = 1 + z,   g_rphi = -alpha (1 + z) sin^2(theta),   g_thth = Sigma,
    g_phph = (r^2 + alpha^2 + z alpha^2 sin^2(theta)) sin^2(theta)

This is regular across the horizon, and is the metric BHAC uses for coord=mks, before its relabeling of r and theta.
*/
template<typename T>
struct KerrSKS3D : ADMMetric<KerrSKS3D<T>, SKS3D, T>{
    T alpha; // spin, with -1 <= alpha <= 1

    using Base = ADMMetric<KerrSKS3D<T>, SKS3D, T>;
    using Coords = Coordinates<SKS3D, T>;

    using Base::fill_gcov;

    /// @brief Get g_ij(x)
    void fill_gcov(MetricTensor<T, SKS3D>& out, const Coords& x) const {
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

    /**
    @brief out_i = (d_i g_jk) u^j u^k. g_ij is independent of phi, so that out_phi = 0 and
    out_i = d_i(g_rr) (u^r)^2 + 2 d_i(g_rphi) u^r u^phi + d_i(g_thth) (u^theta)^2 + d_i(g_phph) (u^phi)^2
    */
    void ddx_gcov_contracted(CoVector<T, 3>& out, const Coords& x, const Vector<T, 3>& u) const {
        /*
        g_rr = 1 + z, g_rphi = -alpha (1 + z) sin^2(theta), g_thth = Sigma and
        g_phph = (r^2 + alpha^2) sin^2(theta) + z alpha^2 sin^4(theta), with d_th Sigma = -alpha^2 sin(2 theta)
        */
        T z, dr_z, dth_z;
        z_and_ddx(z, dr_z, dth_z, x);
        const T r = x.r();
        const T sin_th = sin(x.theta());
        const T sin2_th = sin_th*sin_th;
        const T sin_2th = 2*sin_th*cos(x.theta());
        const T alpha2 = alpha*alpha;

        const T ur2 = u(0)*u(0);
        const T ur_uph = u(0)*u(2);
        const T uth2 = u(1)*u(1);
        const T uph2 = u(2)*u(2);
        out(0) = dr_z*ur2 - 2*alpha*dr_z*sin2_th*ur_uph + 2*r*uth2 + (2*r + dr_z*alpha2*sin2_th)*sin2_th*uph2;
        out(1) = dth_z*ur2 - 2*alpha*(dth_z*sin2_th + (1 + z)*sin_2th)*ur_uph - alpha2*sin_2th*uth2
               + ((r*r + alpha2)*sin_2th + alpha2*(dth_z*sin2_th + 2*z*sin_2th)*sin2_th)*uph2;
        out(2) = 0;
    }

    // ----------------- 3+1 quantities ------------------

    /// @brief The lapse = 1/sqrt(1 + z)
    T lapse(const Coords& x) const {
        const T sin2_th = sin_sqr(x.theta());
        return 1/sqrt(1 + 2*x.r()/sigma(x.r(), sin2_th));
    }

    /// @brief d_i(lapse): A covariant 3-vector (transforms properly under spatial transformations)
    void ddx_lapse(CoVector<T, 3>& out, const Coords& x) const {
        // lapse = (1 + z)^(-1/2), so d(lapse) = -lapse^3/2 dz
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
    so we can interpret it as a contravariant spatial 3-vector
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

    /// @brief z = 2r/Sigma, d_r z = 2 (Sigma - 2r^2)/Sigma^2 and d_theta z = 2 alpha^2 r sin(2 theta)/Sigma^2
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



// ------------ 3D Kerr in MKS ---------------

/*
KerrSKS3D in BHAC's MKS coordinates. The Jacobian J = dx_sks/dx_mks = diag(e^s, 1 + h cos(2 vartheta), 1)
is diagonal, so that with no summation over i, j:

    g_ij = J_i J_j g^{SKS}_ij,   beta^i = beta_SKS^i / J_i,   d_i(lapse) = J_i d_i(lapse_SKS)

The lapse is the same scalar, as the slices are the same.
*/
template<typename T>
struct KerrMKS3D : ADMMetric<KerrMKS3D<T>, MKS3D, T>{
    T alpha; // spin, with -1 <= alpha <= 1
    T h;     // BHAC's coordpar(h_), with |h| < 1
    T r0;    // BHAC's coordpar(R0_)

    using Base = ADMMetric<KerrMKS3D<T>, MKS3D, T>;
    using Coords = Coordinates<MKS3D, T>;

    using Base::fill_gcov;

    /// @brief Get g_ij(x)
    void fill_gcov(MetricTensor<T, MKS3D>& out, const Coords& x) const {
        Coordinates<SKS3D, T> x_sks;
        Jacobian<T, SKS3D, MKS3D> jac;
        to_sks(x_sks, jac, x);
        MetricTensor<T, SKS3D> g;
        sks().fill_gcov(g, x_sks);

        // Like g^{SKS}_ij, g_ij has no (s, vartheta) and (vartheta, phi) components
        out.fastor().fill(T(0));
        out(0, 0) = jac(0, 0)*jac(0, 0)*g(0, 0);
        out(0, 2) = jac(0, 0)*g(0, 2);
        out(2, 0) = out(0, 2);
        out(1, 1) = jac(1, 1)*jac(1, 1)*g(1, 1);
        out(2, 2) = g(2, 2);
    }

    /// @brief det(g_ij) = (J_s J_vartheta)^2 det(g^{SKS}_ij)
    T gcov_det(const Coords& x) const {
        Coordinates<SKS3D, T> x_sks;
        Jacobian<T, SKS3D, MKS3D> jac;
        to_sks(x_sks, jac, x);
        const T det_jac = jac(0, 0)*jac(1, 1);
        return det_jac*det_jac*sks().gcov_det(x_sks);
    }

    /**
    @brief out_i = (d_i g_jk) u^j u^k. With U^j = J_j u^j the same vector in spherical KS, g_jk u^j u^k = g^{SKS}_jk U^j U^k,
    and U^j depends on x^j through J_j, so that with no summation over i:

        out_i = J_i [(d_i g^{SKS}_jk) U^j U^k] + 2 d_i(J_i) u^i (g^{SKS}_ik U^k),

    with d_s(J_s) = e^s = J_s, d_vartheta(J_vartheta) = -2h sin(2 vartheta) and J_phi = 1
    */
    void ddx_gcov_contracted(CoVector<T, 3>& out, const Coords& x, const Vector<T, 3>& u) const {
        Coordinates<SKS3D, T> x_sks;
        Jacobian<T, SKS3D, MKS3D> jac;
        to_sks(x_sks, jac, x);
        const KerrSKS3D<T> g_sks = sks();

        Vector<T, 3> u_sks;
        for (int i = 0; i < 3; ++i){
            u_sks(i) = jac(i, i)*u(i);
        }
        g_sks.ddx_gcov_contracted(out, x_sks, u_sks); // out_phi = 0

        // g^{SKS}_ik U^k for i = r, theta: g^{SKS} has no (r, theta) and (theta, phi) components
        MetricTensor<T, SKS3D> g;
        g_sks.fill_gcov(g, x_sks);
        const T gu_r = g(0, 0)*u_sks(0) + g(0, 2)*u_sks(2);
        const T gu_th = g(1, 1)*u_sks(1);

        const T ds_js = jac(0, 0);
        const T dvth_jvth = -2*h*sin(2*x.vartheta());
        out(0) = jac(0, 0)*out(0) + 2*ds_js*u(0)*gu_r;
        out(1) = jac(1, 1)*out(1) + 2*dvth_jvth*u(1)*gu_th;
    }

    // ----------------- 3+1 quantities ------------------

    /// @brief The lapse = 1/sqrt(1 + 2r/Sigma), as in spherical KS
    T lapse(const Coords& x) const {
        Coordinates<SKS3D, T> x_sks;
        get_coords(x_sks, x, h, r0);
        return sks().lapse(x_sks);
    }

    /// @brief d_i(lapse): A covariant 3-vector (transforms properly under spatial transformations)
    void ddx_lapse(CoVector<T, 3>& out, const Coords& x) const {
        Coordinates<SKS3D, T> x_sks;
        Jacobian<T, SKS3D, MKS3D> jac;
        to_sks(x_sks, jac, x);
        sks().ddx_lapse(out, x_sks);
        out(0) *= jac(0, 0);
        out(1) *= jac(1, 1);
    }

    /**
    @brief The contravariant shift beta^i = (beta_SKS^r e^{-s}, 0, 0)
    @note It transforms as a tensor (in pure spatial transformations),
    so we can interpret it as a contravariant spatial 3-vector
    */
    void shift(Vector<T, 3>& out, const Coords& x) const {
        Coordinates<SKS3D, T> x_sks;
        Jacobian<T, SKS3D, MKS3D> jac;
        to_sks(x_sks, jac, x);
        sks().shift(out, x_sks);
        out(0) /= jac(0, 0);
    }

    /**
    @brief out(i, j) = d_i(beta^j). Only d_s and d_vartheta of beta^s = beta^r/J_s are nonzero. As J_s = e^s = d_s(J_s),
    d_s(beta^s) = d_r(beta^r) - beta^s and d_vartheta(beta^s) = (J_vartheta/J_s) d_theta(beta^r), with beta^r in spherical KS
    @note Not a tensor: under x -> x', d_i'(beta'^j) also has a term with the second derivatives of x'(x)
    */
    void ddx_shift(Tensor<T, Ax<Loc::Dn, 3>, Ax<Loc::Up, 3>>& out, const Coords& x) const{
        Coordinates<SKS3D, T> x_sks;
        Jacobian<T, SKS3D, MKS3D> jac;
        to_sks(x_sks, jac, x);
        const KerrSKS3D<T> g_sks = sks();
        Vector<T, 3> beta_sks;
        g_sks.shift(beta_sks, x_sks);

        g_sks.ddx_shift(out, x_sks); // only out(0, 0) = d_r(beta^r) and out(1, 0) = d_theta(beta^r) are nonzero
        out(0, 0) -= beta_sks(0)/jac(0, 0);
        out(1, 0) *= jac(1, 1)/jac(0, 0);
    }

    // ----------- Kerr-related functions ------------

    /// @brief The same metric, in spherical KS
    KerrSKS3D<T> sks() const {
        return KerrSKS3D<T>{{}, alpha};
    }

    /// @brief The spherical KS point x_sks(x), and the diagonal J = dx_sks/dx
    void to_sks(Coordinates<SKS3D, T>& x_sks, Jacobian<T, SKS3D, MKS3D>& jac, const Coords& x) const {
        get_coords(x_sks, x, h, r0);
        get_dxdX(jac, x, h, r0);
    }
};



// --------- 3D Kerr in Cartesian KS ----------

/*
With M = 1, the Kerr radius r (see kerr_radius) and

    f = 2 r^3/(r^4 + alpha^2 z^2) = 2r/Sigma,   l = ((r x + alpha y)/(r^2 + alpha^2), (r y - alpha x)/(r^2 + alpha^2), z/r),

the 4D metric is g_mu_nu = eta_mu_nu + f l_mu l_nu with l_t = 1, so that

    g_ij = delta_ij + f l_i l_j,   beta_i = g_ti = f l_i,   lapse = 1/sqrt(1 + f).

l is a unit vector in the flat metric, which gives the closed forms below. This is KerrSKS3D in the coordinates
x + i y = (r + i alpha) sin(theta) e^{i phi_ks}, z = r cos(theta), where l_i dx^i = dr - alpha sin^2(theta) dphi_ks.
Unlike the spherical charts, it is regular on the axis, but it is singular at r = 0, inside the inner horizon.
*/
template<typename T>
struct KerrCKS3D : ADMMetric<KerrCKS3D<T>, CKS3D, T>{
    T alpha; // spin, with -1 <= alpha <= 1

    using Base = ADMMetric<KerrCKS3D<T>, CKS3D, T>;
    using Coords = Coordinates<CKS3D, T>;

    using Base::fill_gcov;

    /// @brief Get g_ij(x) = delta_ij + f l_i l_j
    void fill_gcov(MetricTensor<T, CKS3D>& out, const Coords& x) const {
        T r, f;
        std::array<T, 3> l;
        f_and_l(r, f, l, x);
        for (int i = 0; i < 3; ++i){
            for (int j = 0; j < 3; ++j){
                out(i, j) = f*l[i]*l[j];
            }
            out(i, i) += 1;
        }
    }

    /// @brief g^ij(x) = delta_ij - f/(1 + f) l_i l_j, the Sherman-Morrison inverse, as l_i l_i = 1
    void fill_gcont(InverseMetricTensor<T, CKS3D>& out, const Coords& x) const {
        T r, f;
        std::array<T, 3> l;
        f_and_l(r, f, l, x);
        const T c = f/(1 + f);
        for (int i = 0; i < 3; ++i){
            for (int j = 0; j < 3; ++j){
                out(i, j) = -c*l[i]*l[j];
            }
            out(i, i) += 1;
        }
    }

    /// @brief det(g_ij) = 1 + f
    T gcov_det(const Coords& x) const {
        return 1 + f_of(x);
    }

    /// @brief out_i = (d_i g_jk) u^j u^k = d_i(f) (l_j u^j)^2 + 2 f (l_j u^j) d_i(l_k) u^k, from g_jk = delta_jk + f l_j l_k
    void ddx_gcov_contracted(CoVector<T, 3>& out, const Coords& x, const Vector<T, 3>& u) const {
        T r, f;
        std::array<T, 3> l, dr, df;
        f_and_l(r, f, l, x);
        ddx_r_and_f(dr, df, r, f, x);
        std::array<std::array<T, 3>, 3> dl;
        ddx_l(dl, r, l, dr, x);

        const T lu = l[0]*u(0) + l[1]*u(1) + l[2]*u(2);
        for (int i = 0; i < 3; ++i){
            const T dl_u = dl[i][0]*u(0) + dl[i][1]*u(1) + dl[i][2]*u(2);
            out(i) = df[i]*lu*lu + 2*f*lu*dl_u;
        }
    }

    // ----------------- 3+1 quantities ------------------

    /// @brief The lapse = 1/sqrt(1 + f)
    T lapse(const Coords& x) const {
        return 1/sqrt(1 + f_of(x));
    }

    /// @brief d_i(lapse): A covariant 3-vector (transforms properly under spatial transformations)
    void ddx_lapse(CoVector<T, 3>& out, const Coords& x) const {
        // lapse = (1 + f)^(-1/2), so d(lapse) = -lapse^3/2 df
        T r, f;
        std::array<T, 3> l, dr, df;
        f_and_l(r, f, l, x);
        ddx_r_and_f(dr, df, r, f, x);
        const T lap = 1/sqrt(1 + f);
        const T c = -lap*lap*lap/2;
        for (int i = 0; i < 3; ++i){
            out(i) = c*df[i];
        }
    }

    /**
    @brief The contravariant shift beta^i = f/(1 + f) l_i, which is beta_i = f l_i raised with g^ij
    @note It transforms as a tensor (in pure spatial transformations),
    so we can interpret it as a contravariant spatial 3-vector
    */
    void shift(Vector<T, 3>& out, const Coords& x) const {
        T r, f;
        std::array<T, 3> l;
        f_and_l(r, f, l, x);
        const T c = f/(1 + f);
        for (int i = 0; i < 3; ++i){
            out(i) = c*l[i];
        }
    }

    /**
    @brief out(i, j) = d_i(beta^j) = d_i(f) l_j/(1 + f)^2 + f/(1 + f) d_i(l_j)
    @note Not a tensor: under x -> x', d_i'(beta'^j) also has a term with the second derivatives of x'(x)
    */
    void ddx_shift(Tensor<T, Ax<Loc::Dn, 3>, Ax<Loc::Up, 3>>& out, const Coords& x) const{
        T r, f;
        std::array<T, 3> l, dr, df;
        f_and_l(r, f, l, x);
        ddx_r_and_f(dr, df, r, f, x);
        std::array<std::array<T, 3>, 3> dl; // dl[i][j] = d_i(l_j)
        ddx_l(dl, r, l, dr, x);

        const T one_f = 1 + f;
        const T c = f/one_f;
        for (int i = 0; i < 3; ++i){
            const T dc = df[i]/(one_f*one_f); // d_i(f/(1 + f))
            for (int j = 0; j < 3; ++j){
                out(i, j) = dc*l[j] + c*dl[i][j];
            }
        }
    }

    /// @brief Same as the functions above, but r, f, l and their derivatives are computed only once
    void fill_geometry(ADMGeometry<T, CKS3D>& out, const Coords& x, const CoVector<T, 3>& co_u) const {
        T r, f;
        std::array<T, 3> l, dr, df;
        f_and_l(r, f, l, x);
        ddx_r_and_f(dr, df, r, f, x);
        std::array<std::array<T, 3>, 3> dl; // dl[i][j] = d_i(l_j)
        ddx_l(dl, r, l, dr, x);

        const T one_f = 1 + f;
        const T c = f/one_f;
        out.sqrt_gdet = sqrt(one_f);
        out.lapse = 1/out.sqrt_gdet;

        const T l_co_u = l[0]*co_u(0) + l[1]*co_u(1) + l[2]*co_u(2);
        for (int i = 0; i < 3; ++i){
            out.u(i) = co_u(i) - c*l_co_u*l[i];
        }
        const T lu = l[0]*out.u(0) + l[1]*out.u(1) + l[2]*out.u(2);

        const T dlapse_df = -out.lapse/(2*one_f);
        const T dc_df = 1/(one_f*one_f);
        for (int i = 0; i < 3; ++i){
            out.shift(i) = c*l[i];
            out.ddx_lapse(i) = dlapse_df*df[i];
            const T dl_u = dl[i][0]*out.u(0) + dl[i][1]*out.u(1) + dl[i][2]*out.u(2);
            out.ddx_gcov_contracted(i) = df[i]*lu*lu + 2*f*lu*dl_u;
            for (int j = 0; j < 3; ++j){
                out.gcont(i, j) = -c*l[i]*l[j];
                out.ddx_shift(i, j) = dc_df*df[i]*l[j] + c*dl[i][j];
            }
            out.gcont(i, i) += 1;
        }
    }

    // ----------- Kerr-related functions ------------

    /// @brief f = 2 r^3/(r^4 + alpha^2 z^2)
    T f_of(const Coords& x) const {
        const T r = kerr_radius(x, alpha);
        return 2*r*r*r/(r*r*r*r + alpha*alpha*x.z()*x.z());
    }

    /// @brief The Kerr radius r, f and l_i at x
    void f_and_l(T& r, T& f, std::array<T, 3>& l, const Coords& x) const {
        r = kerr_radius(x, alpha);
        const T r2_alpha2 = r*r + alpha*alpha;
        f = 2*r*r*r/(r*r*r*r + alpha*alpha*x.z()*x.z());
        l[0] = (r*x.x() + alpha*x.y())/r2_alpha2;
        l[1] = (r*x.y() - alpha*x.x())/r2_alpha2;
        l[2] = x.z()/r;
    }

    /**
    @brief dr[i] = d_i(r) and df[i] = d_i(f), given the Kerr radius r and f at x.
    With D = r^4 + alpha^2 z^2, differentiating r^4 - (R^2 - alpha^2) r^2 - alpha^2 z^2 = 0 gives
    d(r) = [r^3 (x dx + y dy) + r (r^2 + alpha^2) z dz]/D, and f = 2 r^3/D gives d(f) = f (3 d(r)/r - d(D)/D)
    */
    void ddx_r_and_f(std::array<T, 3>& dr, std::array<T, 3>& df, const T r, const T f, const Coords& x) const {
        const T alpha2 = alpha*alpha;
        const T z = x.z();
        const T r3 = r*r*r;
        const T big_d = r3*r + alpha2*z*z;

        dr[0] = r3*x.x()/big_d;
        dr[1] = r3*x.y()/big_d;
        dr[2] = r*(r*r + alpha2)*z/big_d;
        for (int i = 0; i < 3; ++i){
            df[i] = f*(3/r - 4*r3/big_d)*dr[i]; // d(D) = 4 r^3 d(r) + 2 alpha^2 z dz
        }
        df[2] -= 2*f*alpha2*z/big_d;
    }

    /**
    @brief dl[i][j] = d_i(l_j), given the Kerr radius r, l and dr[i] = d_i(r) at x.
    Differentiating l_x (r^2 + alpha^2) = r x + alpha y, l_y (r^2 + alpha^2) = r y - alpha x and l_z r = z:

        d_i(l_x) = [(x - 2 r l_x) d_i(r) + r delta_ix + alpha delta_iy]/(r^2 + alpha^2),
        d_i(l_y) = [(y - 2 r l_y) d_i(r) + r delta_iy - alpha delta_ix]/(r^2 + alpha^2),
        d_i(l_z) = [delta_iz - l_z d_i(r)]/r
    */
    void ddx_l(std::array<std::array<T, 3>, 3>& dl, const T r, const std::array<T, 3>& l, const std::array<T, 3>& dr, const Coords& x) const {
        const T r2_alpha2 = r*r + alpha*alpha;
        for (int i = 0; i < 3; ++i){
            dl[i][0] = (x.x() - 2*r*l[0])*dr[i]/r2_alpha2;
            dl[i][1] = (x.y() - 2*r*l[1])*dr[i]/r2_alpha2;
            dl[i][2] = -l[2]*dr[i]/r;
        }
        dl[0][0] += r/r2_alpha2;
        dl[1][0] += alpha/r2_alpha2;
        dl[1][1] += r/r2_alpha2;
        dl[0][1] -= alpha/r2_alpha2;
        dl[2][2] += 1/r;
    }
};

} // namespace geom::adm
