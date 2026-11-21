#pragma once

#include <diffeo/linalg/linalg.hpp>
#include <diffeo/einsum/tensors.hpp>

namespace geom{


/*
Derive from this struct for various types of coordinate systems,
with an empty body. For example:


struct Cartesian3D : BaseCoordSys<3> {};

This seems redundant, but the coordinate system struct name
serves as an identifier of the coordinate system,
and can be passed around as a template parameter in the metric
or coordinate transformations. The coordinates themselves, for a scalar
type T, are Coordinates<CoordSys, T>, which can be specialized to add
named accessors (see coordsys.hpp).
*/
template<size_t NDIM>
struct BaseCoordSys{
    static constexpr size_t N = NDIM;
};

template<typename CoordSys, typename T>
struct Coordinates;

template<typename T, typename CoordSys, typename NewCoordSys>
class Jacobian;

/*
Should not be Coords get_coords(const NewCoords& xB);
because Coords& allows for `out` to be initialized with extra parameters
and the get_coords implementation fills `out` using those parameters.
Initializing `out` inside `get_coords` cannot produce `out` with these
requested parameters.
*/
template<typename T, typename CoordSys, typename NewCoordSys>
void get_coords(Coordinates<CoordSys, T>& xA, const Coordinates<NewCoordSys, T>& xB){
    static_assert(sizeof(T) == 0, "get_coords not implemented for this coordinate transformation");
}

/// @brief Get J = dxA/dxB
template<typename T, typename CoordSys, typename NewCoordSys, typename... Args>
void get_dxdX(
    Jacobian<T, CoordSys, NewCoordSys>& out,
    const Coordinates<NewCoordSys, T>& xB,
    Args&&... args);

template<typename CoordSys, typename T>
struct BaseCoordinates{

    using CoordinateSystem = CoordSys;
    using Derived = Coordinates<CoordSys, T>;

    static constexpr size_t N = CoordSys::N;

    template<std::integral Int>
    inline const T& operator[](Int i) const {
        assert(i >= 0 && i < CoordSys::N && "Index out of bounds");
        return data[i];
    }

    template<std::integral Int>
    inline T& operator[](Int i) {
        assert(i >= 0 && i < CoordSys::N && "Index out of bounds");
        return data[i];
    }

    /// @brief Compute J = dx'/dx at the current coordinates
    template<typename NewCoordSys, typename... Args>
    Jacobian<T, NewCoordSys, CoordSys> dXdx(Args&&... args) const{
        
        Jacobian<T, NewCoordSys, CoordSys> out;
        get_dxdX(out, *static_cast<const Derived*>(this), std::forward<Args>(args)...);
        return out;
    }

    std::array<T, N> data;
};


/// @brief Default implementation. Template-specialize the `CoordSys` for custom implementation.
template<typename CoordSys, typename T>
struct Coordinates : BaseCoordinates<CoordSys, T> {};


/**
 @brief J = dx/dx', x = CoordSys, x' = NewCoordSys
 @note Not a tensor in the GR-sense, but still must obey the summation rules.
 TODO: Must distinguish tensors from geometrical objects that do not transform as tensors.
 A similar problem will arise with Christoffel symbols.
*/
template<typename T, typename CoordSys, typename NewCoordSys>
class Jacobian : public Operator<T, CoordSys::N> {
    static_assert(CoordSys::N == NewCoordSys::N, "Transformation unsupported");
public:
    static constexpr size_t N = CoordSys::N;
    
    using Base = Operator<T, CoordSys::N>;
    using Base::Base;
    using Base::operator();

    Jacobian<T, NewCoordSys, CoordSys> inv() const {
        Jacobian<T, NewCoordSys, CoordSys> out;
        linalg::detail::invert_rowmajor<T, N>(out.fastor().data(), this->fastor().data());
        return out;
    }
};

/** 
 @brief Compute J = dx/dx'
 @note Calls x(x') using get_coords(x, x');
*/
template<typename T, typename CoordSys, typename NewCoordSys, typename... Args>
void get_dxdX(
    Jacobian<T, CoordSys, NewCoordSys>& out,
    const Coordinates<NewCoordSys, T>& xB,
    Args&&... args){
    
    static_assert(sizeof(T) == 0, "The Jacobian for this coordinate transformation has not been defined");
}



} // namespace geom