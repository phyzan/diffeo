#pragma once

#include "coordinates.hpp"


namespace geom{


// ========== KerrSchildSpherical ===========

struct KerrSchildSpherical : BaseCoordSys<3> {};


template<typename T>
struct Coordinates<KerrSchildSpherical, T> : BaseCoordinates<KerrSchildSpherical, T>{

    using Base = BaseCoordinates<KerrSchildSpherical, T>;

    T& r() {return this->data[0];}
    T& theta() {return this->data[1];}
    T& phi_ks() {return this->data[2];}

    const T& r() const {return this->data[0];}
    const T& theta() const {return this->data[1];}
    const T& phi_ks() const {return this->data[2];}

};
// =========================================

// ========== ModifiedKerrSchild ===========

struct ModifiedKerrSchild : BaseCoordSys<3> {};

template<typename T>
struct Coordinates<ModifiedKerrSchild, T> : BaseCoordinates<ModifiedKerrSchild, T>{

    using Base = BaseCoordinates<ModifiedKerrSchild, T>;

    T& s() {return this->data[0];}
    T& vartheta() {return this->data[1];}
    T& phi_ks() {return this->data[2];}

    const T& s() const {return this->data[0];}
    const T& vartheta() const {return this->data[1];}
    const T& phi_ks() const {return this->data[2];}

};
// =========================================

} // namespace geom