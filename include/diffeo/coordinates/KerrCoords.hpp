#pragma once

#include "coordinates.hpp"


namespace geom{

/*
The Kerr-Schild family: three charts on the same (ingoing) Kerr-Schild slices, so that the
transformations between them are purely spatial (see transformations.hpp). With M = 1 and spin alpha:

    SKS3D: spherical KS (r, theta, phi_ks). r and theta are the Boyer-Lindquist ones, phi_ks is the KS azimuth
    MKS3D: BHAC's modified KS (s, vartheta, phi_ks), with r = r0 + e^s and theta = vartheta + (h/2) sin(2 vartheta)
    CKS3D: Cartesian KS (x, y, z), with x + i y = (r + i alpha) sin(theta) e^{i phi_ks} and z = r cos(theta)
*/

// ================= SKS3D ==================

struct SKS3D : BaseCoordSys<3> {};


template<typename T>
struct Coordinates<SKS3D, T> : BaseCoordinates<SKS3D, T>{

    using Base = BaseCoordinates<SKS3D, T>;

    T& r() {return this->data[0];}
    T& theta() {return this->data[1];}
    T& phi_ks() {return this->data[2];}

    const T& r() const {return this->data[0];}
    const T& theta() const {return this->data[1];}
    const T& phi_ks() const {return this->data[2];}

};
// =========================================

// ================= MKS3D =================

struct MKS3D : BaseCoordSys<3> {};

template<typename T>
struct Coordinates<MKS3D, T> : BaseCoordinates<MKS3D, T>{

    using Base = BaseCoordinates<MKS3D, T>;

    T& s() {return this->data[0];}
    T& vartheta() {return this->data[1];}
    T& phi_ks() {return this->data[2];}

    const T& s() const {return this->data[0];}
    const T& vartheta() const {return this->data[1];}
    const T& phi_ks() const {return this->data[2];}

};
// =========================================

// ================= CKS3D =================

struct CKS3D : BaseCoordSys<3> {};

template<typename T>
struct Coordinates<CKS3D, T> : BaseCoordinates<CKS3D, T>{

    using Base = BaseCoordinates<CKS3D, T>;

    T& x() {return this->data[0];}
    T& y() {return this->data[1];}
    T& z() {return this->data[2];}

    const T& x() const {return this->data[0];}
    const T& y() const {return this->data[1];}
    const T& z() const {return this->data[2];}

};
// =========================================

} // namespace geom
