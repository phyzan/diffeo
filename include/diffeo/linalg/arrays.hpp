#pragma once

#include "Fastor/config/config.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>


namespace geom{

// Which index runs fastest in memory: Left = column-major, Right = row-major
enum class Layout : uint8_t{
    Left, Right,
};

namespace detail{

template<typename T, size_t Depth>
struct NestedList{
    using type = std::initializer_list<typename NestedList<T, Depth-1>::type>;
};

template<typename T>
struct NestedList<T, 0>{
    using type = T;
};

} // namespace geom::detail

// initializer_list nested Depth times: NestedList<T, 2> = std::initializer_list<std::initializer_list<T>>
template<typename T, size_t Depth>
using NestedList = typename detail::NestedList<T, Depth>::type;

template<typename T, Layout LY, size_t... Dims>
class Array{
public:
    static constexpr size_t Rank = sizeof...(Dims);
    static constexpr size_t Size = (Dims*...*1);
    static constexpr std::array<size_t, Rank> Shape = {Dims...};
    static_assert(sizeof...(Dims) > 0, "Rank must be known at compile time");
    static_assert(Size > 0, "All dims must be positive");

    static constexpr std::array<size_t, Rank> Strides = [](){
        std::array<size_t, Rank> s{};
        size_t stride = 1;
        if constexpr (LY == Layout::Left){
            for (size_t i = 0; i < Rank; ++i){
                s[i] = stride;
                stride *= Shape[i];
            }
        } else {
            for (size_t i = Rank; i-- > 0;){
                s[i] = stride;
                stride *= Shape[i];
            }
        }
        return s;
    }();

    // --------------- Constructors --------------
    constexpr Array() = default;

    // Nested as deep as Rank, in index order whatever the layout:
    // Array<double, Layout::Right, 2, 3> A{{1, 2, 3}, {4, 5, 6}};  // A(1, 0) == 4
    // Each list must match its axis length (asserted); in release builds, extra values
    // are ignored and missing ones are zero
    constexpr Array(NestedList<T, Rank> values) : flat_{} {
        assign_nested<0>(values, 0);
    }

    // --------------- Accessors -----------------
    constexpr const T* data() const {
        return flat_.data();
    }

    constexpr T* data() {
        return flat_.data();
    }

    constexpr size_t size() const {
        return Size;
    }

    constexpr size_t ndim() const {
        return Rank;
    }

    constexpr size_t shape(size_t i) const {
        assert(i < Rank && "Axis out of bounds");
        return Shape[i];
    }

    template<std::integral Int>
    constexpr const T& operator[](Int i) const {
        assert(in_bounds(i, Size) && "Index out of bounds");
        return flat_[static_cast<size_t>(i)];
    }

    template<std::integral Int>
    constexpr T& operator[](Int i) {
        assert(in_bounds(i, Size) && "Index out of bounds");
        return flat_[static_cast<size_t>(i)];
    }
    
    template<std::integral... Int>
    static constexpr size_t offset(const Int... idx) {
        static_assert(sizeof...(Int) == Rank, "Incorrect number of indices");
        return [&]<size_t... I>(std::index_sequence<I...> /**/){
            assert((in_bounds(idx, Shape[I]) && ...) && "Index out of bounds");
            return ((static_cast<size_t>(idx) * Strides[I]) + ...);
        }(std::make_index_sequence<Rank>{});
    }

    template<std::integral... Int>
    constexpr const T& operator()(const Int... idx) const {
        return flat_[offset(idx...)];
    }

    template<std::integral... Int>
    constexpr T& operator()(const Int... idx) {
        return flat_[offset(idx...)];
    }

private:

    // Signed-safe i in [0, n): no -Wsign-compare or -Wtype-limits for unsigned Int
    template<std::integral Int>
    static constexpr bool in_bounds(Int i, size_t n) {
        return std::cmp_greater_equal(i, 0) && std::cmp_less(i, n);
    }

    // values: the sub-list along Axis, base: flat offset of its first element
    template<size_t Axis>
    constexpr void assign_nested(const NestedList<T, Rank - Axis>& values, size_t base) {
        assert(values.size() == Shape[Axis] && "Initializer list does not match the shape");
        const size_t n = std::min(values.size(), Shape[Axis]);
        for (size_t i = 0; i < n; ++i){
            const size_t off = base + i*Strides[Axis];
            if constexpr (Axis + 1 == Rank){
                flat_[off] = values.begin()[i];
            } else {
                assign_nested<Axis + 1>(values.begin()[i], off);
            }
        }
    }

    std::array<T, Size> flat_;
};


namespace detail{

template<Layout LY, typename F, std::integral First, std::integral... Int>
FASTOR_INLINE constexpr void for_each_impl(F&& f, const First Ni, const Int... Nj){
    if constexpr (sizeof...(Int) == 0){
        for (size_t i = 0; i < Ni; ++i) {
            std::forward<F>(f)(i);
        }
    } else if constexpr (LY == Layout::Left) {
        for_each_impl<LY>([&](const auto... Nk) {
            for (size_t i = 0; i < Ni; ++i) {
                f(i, Nk...);
            }
        }, Nj...);
    } else {
        for (size_t i = 0; i < Ni; ++i) {
            for_each_impl<LY>([&](const auto... Nk) {
                f(i, Nk...);
            }, Nj...);
        }
    }
}

template<Layout LY, typename F, size_t Dim, size_t... I>
constexpr void make_for_each(F&& f, const std::array<size_t, Dim> shape, std::index_sequence<I...> /**/){
    for_each_impl<LY>(std::forward<F>(f), shape[I]...);
}

template<typename T, size_t Rank, Layout LY, size_t N>
struct HelperNdArray{
    // Only declared: its return type is Array<T, LY, N, ..., N>, with N repeated once per Is
    template<size_t... Is>
    static Array<T, LY, (static_cast<void>(Is), N)...> make(std::index_sequence<Is...>);

    using type = decltype(make(std::make_index_sequence<Rank>{}));
};

} // namespace geom::detail


// --------------- Multidim loops -----------------
// Calls f(i0, i1, ...) with each index in [0, shape[axis]). Fastest::Left: i0 changes fastest (column-major).
template<Layout LY, typename F, size_t Dim>
constexpr void for_each(F&& f, const std::array<size_t, Dim> shape){
    detail::make_for_each<LY>(std::forward<F>(f), shape, std::make_index_sequence<Dim>{});
}

template<Layout LY, typename F, std::integral... Int>
constexpr void for_each(F&& f, const Int... shape){
    detail::for_each_impl<LY>(std::forward<F>(f), shape...);
}
// ------------------------------------------------


// Array with `Rank` indices, of common shape `N`
template<typename T, size_t Rank, Layout LY, size_t N>
using ArrayND = typename detail::HelperNdArray<T, Rank, LY, N>::type;


// ======== Column-major aliases =========
// F-style array with `Rank` indices, of common shape `N`
template<typename T, size_t Rank, size_t N>
using FArrayND = ArrayND<T, Rank, Layout::Left, N>;

template<typename T, size_t M, size_t N>
using FArray2D = Array<T, Layout::Left, M, N>;

// ======== Row-major aliases =========
// C-style array with `Rank` indices, of common shape `N`
template<typename T, size_t Rank, size_t N>
using CArrayND = ArrayND<T, Rank, Layout::Right, N>;

template<typename T, size_t M, size_t N>
using CArray2D = Array<T, Layout::Right, M, N>;


} // namespace geom