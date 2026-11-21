#pragma once

#include "coordinates.hpp"

namespace geom{


// =========== Cartesian3D ============
struct Cartesian3D : BaseCoordSys<3> {};

template<typename T>
struct Coordinates<Cartesian3D, T> : BaseCoordinates<Cartesian3D, T>{
    T& x() {return this->data[0];}
    T& y() {return this->data[1];}
    T& z() {return this->data[2];}

    const T& x() const {return this->data[0];}
    const T& y() const {return this->data[1];}
    const T& z() const {return this->data[2];}
};
// ====================================

// =========== Cylindrical3D ==========
struct Cylindrical3D : BaseCoordSys<3> {};

template<typename T>
struct Coordinates<Cylindrical3D, T> : BaseCoordinates<Cylindrical3D, T>{
    T& rho() {return this->data[0];}
    T& phi() {return this->data[1];}
    T& z() {return this->data[2];}

    const T& rho() const {return this->data[0];}
    const T& phi() const {return this->data[1];}
    const T& z() const {return this->data[2];}
};
// ====================================


// ============ Spherical =============
struct Spherical : BaseCoordSys<3> {};

template<typename T>
struct Coordinates<Spherical, T> : BaseCoordinates<Spherical, T>{
    T& r() {return this->data[0];}
    T& theta() {return this->data[1];}
    T& phi() {return this->data[2];}

    const T& r() const {return this->data[0];}
    const T& theta() const {return this->data[1];}
    const T& phi() const {return this->data[2];}
};
// ====================================

} // namespace geom