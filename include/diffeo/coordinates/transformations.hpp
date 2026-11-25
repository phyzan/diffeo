#pragma once


#include <cmath>
#include <limits>
#include <numbers>

#include "common.hpp"
#include "KerrCoords.hpp"


namespace geom{


// ============= Cartesian3D <-> Cylindrical3D =============

template<typename T>
void get_coords(Coordinates<Cartesian3D, T>& xA, const Coordinates<Cylindrical3D, T>& xB){
    xA.x() = xB.rho() * cos(xB.phi());
    xA.y() = xB.rho() * sin(xB.phi());
    xA.z() = xB.z();
}

template<typename T>
void get_coords(Coordinates<Cylindrical3D, T>& xA, const Coordinates<Cartesian3D, T>& xB){
    /*
    rho = sqrt(x^2 + y^2)
    phi = atan2(y, x), in [0, 2pi)
    z = z
    */
    xA.rho() = sqrt(xB.x()*xB.x() + xB.y()*xB.y());
    xA.phi() = atan2(xB.y(), xB.x()); // (-pi, pi]
    if (xA.phi() < 0){
        auto pi = std::numbers::pi_v<T>;
        xA.phi() += 2*pi;
        if (xA.phi() >= 2*pi){ // within ~1e-16 below the x axis, 2pi + phi rounds to 2pi
            xA.phi() -= 2*pi;
        }
    }
    xA.z() = xB.z();
}


// ============= Cartesian3D <-> Spherical =============
template<typename T>
void get_coords(Coordinates<Cartesian3D, T>& xA, const Coordinates<Spherical, T>& xB){
    xA.x() = xB.r() * sin(xB.theta()) * cos(xB.phi());
    xA.y() = xB.r() * sin(xB.theta()) * sin(xB.phi());
    xA.z() = xB.r() * cos(xB.theta());
}

template<typename T>
void get_coords(Coordinates<Spherical, T>& xA, const Coordinates<Cartesian3D, T>& xB){
    /*
    r = sqrt(x^2 + y^2 + z^2)
    theta = atan2(sqrt(x^2 + y^2), z), in [0, pi]
    phi = atan2(y, x), in [0, 2pi)
    */
    const T rho = sqrt(xB.x()*xB.x() + xB.y()*xB.y());
    xA.r() = sqrt(xB.x()*xB.x() + xB.y()*xB.y() + xB.z()*xB.z());
    xA.theta() = atan2(rho, xB.z()); // Unlike acos(z/r), this is accurate near the poles
    xA.phi() = atan2(xB.y(), xB.x()); // (-pi, pi]
    if (xA.phi() < 0){
        auto pi = std::numbers::pi_v<T>;
        xA.phi() += 2*pi;
        if (xA.phi() >= 2*pi){ // within ~1e-16 below the x axis, 2pi + phi rounds to 2pi
            xA.phi() -= 2*pi;
        }
    }
}





// ================== Kerr - Schild family ====================
/*
SKS3D, MKS3D and CKS3D are charts on the same Kerr-Schild slices (see KerrCoords.hpp), so the transformations
between them are purely spatial. Their extra parameters are BHAC's h and r0 for SKS3D <-> MKS3D, the spin alpha
for SKS3D <-> CKS3D, and (alpha, h, r0) for MKS3D <-> CKS3D, which goes through SKS3D.
*/

namespace detail{

/// @brief atan2(y, x), in [0, 2pi) instead of (-pi, pi]
template<typename T>
T azimuth(const T y, const T x){
    T phi = atan2(y, x);
    if (phi < 0){
        const auto pi = std::numbers::pi_v<T>;
        phi += 2*pi;
        if (phi >= 2*pi){ // within ~1e-16 below the x axis, 2pi + phi rounds to 2pi
            phi -= 2*pi;
        }
    }
    return phi;
}

/// @brief (sin(x), cos(x)) -> (sin(x + d), cos(x + d)), for |d| <= 1
template<typename T>
void rotate(T& sin_x, T& cos_x, const T& d){
    const T tol = std::numeric_limits<T>::epsilon();
    T sin_d = d;
    T cos_d = 1;
    T term = d;
    for (int n = 2; abs(term) > tol && n < 10000; n += 2){
        term *= -d/n;
        cos_d += term;
        term *= d/(n + 1);
        sin_d += term;
    }
    const T tmp = sin_x*cos_d + cos_x*sin_d;
    cos_x = cos_x*cos_d - sin_x*sin_d;
    sin_x = tmp;
}

/// @brief Solves theta = vartheta + (h/2) sin(2 vartheta) for |h| < 1, with Newton's method on delta = 2 (vartheta - theta).
/// Always converges: limiting the first step to |h| puts it before the root, where the rest approach it monotonically
template<typename T>
void solve_vartheta(T& vartheta, T& dtheta, const T& theta, const T& sin_2theta, const T& cos_2theta, const auto h){
    const T tol = 8 * std::numeric_limits<T>::epsilon();
    T sin_x = sin_2theta;
    T cos_x = cos_2theta;

    T step = -h*sin_x/(1 + h*cos_x);
    if (step > abs(h)){
        step = abs(h);
    }
    else if (step < -abs(h)){
        step = -abs(h);
    }
    T delta = step;
    rotate(sin_x, cos_x, step);

    int direction = 0;
    for (int i = 0; i < 100; ++i){
        step = -(delta + h*sin_x)/(1 + h*cos_x);
        delta += step;
        rotate(sin_x, cos_x, step);
        const int sign = (step > 0) ? 1 : -1;
        if (!(abs(step) > tol) || sign == -direction){ // converged, NaN, or rounding errors
            break;
        }
        direction = sign;
    }
    vartheta = theta + delta/2;
    dtheta = 1 + h*cos_x;
}

template<typename T>
T vartheta_of(const T theta, const auto h){
    T vartheta, dtheta;
    solve_vartheta<T>(vartheta, dtheta, theta, sin(2*theta), cos(2*theta), h);
    return vartheta;
}
} // namespace geom::detail


// ============= SKS3D <-> MKS3D =============

/// @brief x_sks(x_mks): r = r0 + e^s, theta = vartheta + (h/2) sin(2 vartheta)
template<typename T, typename U>
void get_coords(Coordinates<SKS3D, T>& x_sks, const Coordinates<MKS3D, T>& x_mks, const U h, const U r0){
    x_sks.r() = r0 + exp(x_mks.s());
    x_sks.theta() = x_mks.vartheta() + h/2 * sin(2*x_mks.vartheta());
    x_sks.phi_ks() = x_mks.phi_ks();
}


/// @brief x_mks(x_sks). vartheta(theta) has no closed form, and is found with Newton's method
template<typename T, typename U>
void get_coords(Coordinates<MKS3D, T>& x_mks, const Coordinates<SKS3D, T>& x_sks, const U h, const U r0){
    x_mks.s() = log(x_sks.r() - r0);
    x_mks.vartheta() = detail::vartheta_of(x_sks.theta(), h);
    x_mks.phi_ks() = x_sks.phi_ks();
}

/// @brief J = dx_mks/dx_sks at x_sks = diag(1/(r - r0), 1/(1 + h cos(2 vartheta)), 1)
template<typename T, typename U>
void get_dxdX(Jacobian<T, MKS3D, SKS3D>& jac, const Coordinates<SKS3D, T>& x_sks, const U h, const U r0){
    const T vartheta = detail::vartheta_of(x_sks.theta(), h);
    jac.fastor().fill(T(0));

    jac(0, 0) = 1/(x_sks.r() - r0); // same as exp(-s), but without invoking exp(...)
    jac(1, 1) = 1/(1 + h*cos(2*vartheta));
    jac(2, 2) = 1;
}

/// @brief J = dx_sks/dx_mks at x_mks = diag(e^s, 1 + h cos(2 vartheta), 1). Unlike the inverse, needs no root-finding
template<typename T, typename U>
void get_dxdX(Jacobian<T, SKS3D, MKS3D>& jac, const Coordinates<MKS3D, T>& x_mks, const U h, const U /*r0*/){
    jac.fastor().fill(T(0));
    jac(0, 0) = exp(x_mks.s());
    jac(1, 1) = 1 + h*cos(2*x_mks.vartheta());
    jac(2, 2) = 1;
}


// ============= SKS3D <-> CKS3D =============

/**
 @brief The Kerr radius r >= 0 of a CKS point: the root of r^4 - (R^2 - alpha^2) r^2 - alpha^2 z^2 = 0, with R^2 = x^2 + y^2 + z^2
 @note r = 0 on the disk z = 0, x^2 + y^2 <= alpha^2 inside the inner horizon, where (r, theta) are not unique
*/
template<typename T, typename U>
T kerr_radius(const Coordinates<CKS3D, T>& x_cks, const U alpha){
    const T z2 = x_cks.z()*x_cks.z();
    const T b = x_cks.x()*x_cks.x() + x_cks.y()*x_cks.y() + z2 - alpha*alpha; // R^2 - alpha^2
    const T d = sqrt(b*b + 4*alpha*alpha*z2);
    // r^2 = (b + d)/2. When b < 0, it is written as 2 alpha^2 z^2/(d - b), which avoids the cancellation in b + d
    const T r2 = (b >= 0) ? (b + d)/2 : 2*alpha*alpha*z2/(d - b);
    return sqrt(r2);
}

/// @brief x_cks(x_sks): x + i y = (r + i alpha) sin(theta) e^{i phi_ks}, z = r cos(theta)
template<typename T, typename U>
void get_coords(Coordinates<CKS3D, T>& x_cks, const Coordinates<SKS3D, T>& x_sks, const U alpha){
    const T sin_th = sin(x_sks.theta());
    const T cos_ph = cos(x_sks.phi_ks());
    const T sin_ph = sin(x_sks.phi_ks());
    x_cks.x() = (x_sks.r()*cos_ph - alpha*sin_ph)*sin_th;
    x_cks.y() = (x_sks.r()*sin_ph + alpha*cos_ph)*sin_th;
    x_cks.z() = x_sks.r()*cos(x_sks.theta());
}

/**
 @brief x_sks(x_cks), with theta in [0, pi] and phi_ks in [0, 2pi).
 phi_ks = atan2(y, x) - atan(alpha/r) is not the Cartesian azimuth, and is set to 0 on the axis x = y = 0
*/
template<typename T, typename U>
void get_coords(Coordinates<SKS3D, T>& x_sks, const Coordinates<CKS3D, T>& x_cks, const U alpha){
    const T x = x_cks.x();
    const T y = x_cks.y();
    const T r = kerr_radius(x_cks, alpha);
    const T rho = sqrt(x*x + y*y); // = sqrt(r^2 + alpha^2) sin(theta)

    x_sks.r() = r;
    // sin(theta) = rho/sqrt(r^2 + alpha^2) and cos(theta) = z/r. Unlike acos(z/r), atan2 is accurate near the poles
    x_sks.theta() = atan2(r*rho, sqrt(r*r + alpha*alpha)*x_cks.z());
    // (x + i y)(r - i alpha) = (r^2 + alpha^2) sin(theta) e^{i phi_ks}
    x_sks.phi_ks() = detail::azimuth(r*y - alpha*x, r*x + alpha*y);
}

/// @brief J = dx_cks/dx_sks at x_sks. The phi_ks column is (-y, x, 0), a rotation about the z axis
template<typename T, typename U>
void get_dxdX(Jacobian<T, CKS3D, SKS3D>& jac, const Coordinates<SKS3D, T>& x_sks, const U alpha){
    const T r = x_sks.r();
    const T sin_th = sin(x_sks.theta());
    const T cos_th = cos(x_sks.theta());
    const T sin_ph = sin(x_sks.phi_ks());
    const T cos_ph = cos(x_sks.phi_ks());
    const T u = r*cos_ph - alpha*sin_ph; // x = u sin(theta)
    const T v = r*sin_ph + alpha*cos_ph; // y = v sin(theta)

    // rows (x, y, z), columns (r, theta, phi_ks)
    jac(0, 0) = sin_th*cos_ph;
    jac(0, 1) = cos_th*u;
    jac(0, 2) = -sin_th*v;

    jac(1, 0) = sin_th*sin_ph;
    jac(1, 1) = cos_th*v;
    jac(1, 2) = sin_th*u;

    jac(2, 0) = cos_th;
    jac(2, 1) = -r*sin_th;
    jac(2, 2) = 0;
}

/**
 @brief J = dx_sks/dx_cks at x_cks. With Sigma = r^2 + alpha^2 cos^2(theta) and rho^2 = x^2 + y^2 = (r^2 + alpha^2) sin^2(theta):

    dr      = [r (x dx + y dy) + (r^2 + alpha^2) cos(theta) dz]/Sigma
    dtheta  = [cot(theta) (x dx + y dy) - r sin(theta) dz]/Sigma
    dphi_ks = (x dy - y dx)/rho^2 + alpha dr/(r^2 + alpha^2)

 @note The theta and phi_ks rows are undefined on the axis x = y = 0, where the spherical coordinates are singular
*/
template<typename T, typename U>
void get_dxdX(Jacobian<T, SKS3D, CKS3D>& jac, const Coordinates<CKS3D, T>& x_cks, const U alpha){
    const T x = x_cks.x();
    const T y = x_cks.y();
    const T r = kerr_radius(x_cks, alpha);
    const T r2_alpha2 = r*r + alpha*alpha;
    const T sqrt_r2_alpha2 = sqrt(r2_alpha2);
    const T cos_th = x_cks.z()/r;
    const T sig = r*r + alpha*alpha*cos_th*cos_th;
    const T rho2 = x*x + y*y;
    const T rho = sqrt(rho2);

    jac(0, 0) = r*x/sig;
    jac(0, 1) = r*y/sig;
    jac(0, 2) = r2_alpha2*cos_th/sig;

    const T cot_th = cos_th*sqrt_r2_alpha2/rho;
    jac(1, 0) = cot_th*x/sig;
    jac(1, 1) = cot_th*y/sig;
    jac(1, 2) = -r*rho/(sqrt_r2_alpha2*sig); // -r sin(theta)/Sigma

    const T c = alpha/r2_alpha2;
    jac(2, 0) = -y/rho2 + c*jac(0, 0);
    jac(2, 1) = x/rho2 + c*jac(0, 1);
    jac(2, 2) = c*jac(0, 2);
}


// ============= MKS3D <-> CKS3D =============
// Through SKS3D, with the parameters of both transformations, in the order (alpha, h, r0)

/// @brief x_cks(x_mks)
template<typename T, typename U>
void get_coords(Coordinates<CKS3D, T>& x_cks, const Coordinates<MKS3D, T>& x_mks, const U alpha, const U h, const U r0){
    Coordinates<SKS3D, T> x_sks;
    get_coords(x_sks, x_mks, h, r0);
    get_coords(x_cks, x_sks, alpha);
}

/// @brief x_mks(x_cks)
template<typename T, typename U>
void get_coords(Coordinates<MKS3D, T>& x_mks, const Coordinates<CKS3D, T>& x_cks, const U alpha, const U h, const U r0){
    Coordinates<SKS3D, T> x_sks;
    get_coords(x_sks, x_cks, alpha);
    get_coords(x_mks, x_sks, h, r0);
}

/// @brief J = dx_cks/dx_mks at x_mks = (dx_cks/dx_sks)(dx_sks/dx_mks). The second factor is diagonal, and scales the columns
template<typename T, typename U>
void get_dxdX(Jacobian<T, CKS3D, MKS3D>& jac, const Coordinates<MKS3D, T>& x_mks, const U alpha, const U h, const U r0){
    Coordinates<SKS3D, T> x_sks;
    get_coords(x_sks, x_mks, h, r0);
    Jacobian<T, CKS3D, SKS3D> dcks_dsks;
    get_dxdX(dcks_dsks, x_sks, alpha);
    Jacobian<T, SKS3D, MKS3D> dsks_dmks;
    get_dxdX(dsks_dmks, x_mks, h, r0);

    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 3; ++j){
            jac(i, j) = dcks_dsks(i, j)*dsks_dmks(j, j);
        }
    }
}

/// @brief J = dx_mks/dx_cks at x_cks = (dx_mks/dx_sks)(dx_sks/dx_cks). The first factor is diagonal, and scales the rows
template<typename T, typename U>
void get_dxdX(Jacobian<T, MKS3D, CKS3D>& jac, const Coordinates<CKS3D, T>& x_cks, const U alpha, const U h, const U r0){
    Coordinates<SKS3D, T> x_sks;
    get_coords(x_sks, x_cks, alpha);
    Jacobian<T, MKS3D, SKS3D> dmks_dsks;
    get_dxdX(dmks_dsks, x_sks, h, r0);
    Jacobian<T, SKS3D, CKS3D> dsks_dcks;
    get_dxdX(dsks_dcks, x_cks, alpha);

    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 3; ++j){
            jac(i, j) = dmks_dsks(i, i)*dsks_dcks(i, j);
        }
    }
}

/// @brief x_mks(x_cks) and J = dx_cks/dx_mks, cheaper than separately. Requires r > 0
template<typename T, typename U>
void get_coords_and_dxdX(
    Coordinates<MKS3D, T>& x_mks,
    Jacobian<T, CKS3D, MKS3D>& jac,
    const Coordinates<CKS3D, T>& x_cks,
    const U alpha, const U h, const U r0){

    const T x = x_cks.x();
    const T y = x_cks.y();
    const T r = kerr_radius(x_cks, alpha);
    const T r2_alpha2 = r*r + alpha*alpha;
    const T sqrt_r2_alpha2 = sqrt(r2_alpha2);
    const T rho = sqrt(x*x + y*y);
    const T sin_th = rho/sqrt_r2_alpha2;
    const T cos_th = x_cks.z()/r;
    const T theta = atan2(r*rho, sqrt_r2_alpha2*x_cks.z());

    T vartheta, dtheta;
    detail::solve_vartheta<T>(vartheta, dtheta, theta, 2*sin_th*cos_th, (cos_th - sin_th)*(cos_th + sin_th), h);

    const T exp_s = r - r0;
    x_mks.s() = log(exp_s);
    x_mks.vartheta() = vartheta;
    x_mks.phi_ks() = detail::azimuth(r*y - alpha*x, r*x + alpha*y);

    // (x, y)/sin(theta), with phi_ks = 0 on the axis
    T u = r;
    T v = alpha;
    if (rho > 0){
        u = sqrt_r2_alpha2*(x/rho);
        v = sqrt_r2_alpha2*(y/rho);
    }

    // rows (x, y, z), columns (s, vartheta, phi_ks)
    jac(0, 0) = (r*x + alpha*y)/r2_alpha2*exp_s;
    jac(0, 1) = cos_th*u*dtheta;
    jac(0, 2) = -y;

    jac(1, 0) = (r*y - alpha*x)/r2_alpha2*exp_s;
    jac(1, 1) = cos_th*v*dtheta;
    jac(1, 2) = x;

    jac(2, 0) = cos_th*exp_s;
    jac(2, 1) = -r*sin_th*dtheta;
    jac(2, 2) = 0;
}

} // namespace geom
