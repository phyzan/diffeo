#pragma once


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










// ============= Spherical KS <-> MKS =============

namespace detail{
/*
Solves theta = vartheta + (h/2) sin(2 vartheta) for vartheta. With x = 2 vartheta and
epsilon = 2 theta, this is epsilon = x + h sin(x), whose right side is increasing for |h| < 1.
The root is within |h| of epsilon, because x - epsilon = -h sin(x), so any theta can be given.
*/
template<typename T>
T vartheta_of(const T theta, const auto h){
    const T tol = 8 * std::numeric_limits<T>::epsilon();
    const T epsilon = 2*theta;
    T lo = epsilon - abs(h);
    T hi = epsilon + abs(h);
    for (int i = 0; i < 100 && hi - lo > tol; ++i){
        const T x = (lo + hi)/2;
        if (epsilon - x - h*sin(x) > 0){
            lo = x;
        }
        else{
            hi = x;
        }
    }

    // Performing one Newton step in case Dual numbers are used in the
    // future for automatic differentiation.
    T vartheta = (lo + hi)/4;
    vartheta -= (vartheta + h/2 * sin(2*vartheta) - theta) / (1 + h * cos(2*vartheta));
    return vartheta;
}
} // namespace geom::detail

/// @brief Compute x(x')
template<typename T, typename U>
void get_coords(Coordinates<KerrSchildSpherical, T>& xA, const Coordinates<ModifiedKerrSchild, T>& xB, const U h, const U r0){
    xA.r() = r0 + exp(xB.s());
    xA.theta() = xB.vartheta() + h/2 * sin(2*xB.vartheta());
    xA.phi_ks() = xB.phi_ks();
}


///@brief Compute x'(x)
template<typename T, typename U>
void get_coords(Coordinates<ModifiedKerrSchild, T>& xA, const Coordinates<KerrSchildSpherical, T>& xB, const U h, const U r0){
    xA.s() = log(xB.r() - r0);
    xA.vartheta() = detail::vartheta_of(xB.theta(), h);
    xA.phi_ks() = xB.phi_ks();
}

/// @brief Compute J = dx{mks}/dx'{sph}. Faster than the Dual overload, which would pass Duals through get_coords
template<typename T, typename U>
void get_dxdX(Jacobian<T, ModifiedKerrSchild, KerrSchildSpherical>& jac, const Coordinates<KerrSchildSpherical, T>& x_kss, const U h, const U r0){
    const T vartheta = detail::vartheta_of(x_kss.theta(), h);
    jac.fastor().fill(T(0));

    jac(0, 0) = 1/(x_kss.r() - r0); // same as exp(-x_mks.s()), but without invoking exp(...)
    jac(1, 1) = 1/(1 + h*cos(2*vartheta));
    jac(2, 2) = 1;
}

/// @brief Compute J = dx{sph}/dx'{mks} = diag(e^s, 1 + h cos(2 vartheta), 1). Unlike the inverse, needs no root-finding
template<typename T, typename U>
void get_dxdX(Jacobian<T, KerrSchildSpherical, ModifiedKerrSchild>& jac, const Coordinates<ModifiedKerrSchild, T>& x_mks, const U h, const U /*r0*/){
    jac.fastor().fill(T(0));
    jac(0, 0) = exp(x_mks.s());
    jac(1, 1) = 1 + h*cos(2*x_mks.vartheta());
    jac(2, 2) = 1;
}

} // namespace geom
