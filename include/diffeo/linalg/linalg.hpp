#pragma once

/*
Linear algebra module, for matrices of small, trivially copyable/constructible matrices
*/


#include "arrays.hpp"


namespace geom::linalg{

template<size_t N>
constexpr int levi_civita(std::array<size_t, N> idx);

namespace detail{

// In-place LU decomposition with partial pivoting of a row-major N x N matrix: P*A = L*U.
// L is unit lower triangular (stored below the diagonal), U is upper triangular (diagonal and above).
// perm[i] is the original row that ended up in row i.
// Returns the sign of the permutation P, or 0 if A is singular.
template<typename T, size_t N>
int lu_decompose(std::array<T, N*N>& a, std::array<size_t, N>& perm){
    int sign = 1;
    for (size_t i = 0; i < N; ++i){
        perm[i] = i;
    }
    for (size_t k = 0; k < N; ++k){
        size_t p = k;
        T pmax = std::abs(a[k*N + k]);
        for (size_t i = k+1; i < N; ++i){
            const T v = std::abs(a[i*N + k]);
            if (v > pmax){
                pmax = v;
                p = i;
            }
        }
        if (pmax == T(0)){
            return 0;
        }
        if (p != k){
            for (size_t j = 0; j < N; ++j){
                std::swap(a[k*N + j], a[p*N + j]);
            }
            std::swap(perm[k], perm[p]);
            sign = -sign;
        }
        for (size_t i = k+1; i < N; ++i){
            const T l = (a[i*N + k] /= a[k*N + k]);
            for (size_t j = k+1; j < N; ++j){
                a[i*N + j] -= l * a[k*N + j];
            }
        }
    }
    return sign;
}

// out = inv(mat), both row-major. out may alias mat. A singular mat gives an all-NaN out.
template<typename T, size_t N>
void invert_rowmajor(T* out, const T* mat){
    std::array<T, N*N> lu;
    std::copy(mat, mat + N*N, lu.begin());
    std::array<size_t, N> perm;
    if (lu_decompose<T, N>(lu, perm) == 0){
        std::fill(out, out + N*N, std::numeric_limits<T>::quiet_NaN());
        return;
    }

    // column j of the inverse solves A*x = e_j, i.e. L*y = P*e_j, then U*x = y
    std::array<T, N> x;
    for (size_t j = 0; j < N; ++j){
        for (size_t i = 0; i < N; ++i){
            T s = (perm[i] == j) ? T(1) : T(0);
            for (size_t k = 0; k < i; ++k){
                s -= lu[i*N + k] * x[k];
            }
            x[i] = s;
        }
        for (size_t i = N; i-- > 0;){
            T s = x[i];
            for (size_t k = i+1; k < N; ++k){
                s -= lu[i*N + k] * x[k];
            }
            x[i] = s / lu[i*N + i];
        }
        for (size_t i = 0; i < N; ++i){
            out[i*N + j] = x[i];
        }
    }
}

template<typename T, size_t N>
T det_rowmajor(const T* mat){
    std::array<T, N*N> lu;
    std::copy(mat, mat + N*N, lu.begin());
    std::array<size_t, N> perm;
    const int sign = lu_decompose<T, N>(lu, perm);
    T det = T(sign);
    if (sign != 0){
        for (size_t i = 0; i < N; ++i){
            det *= lu[i*N + i];
        }
    }
    return det;
}

template<size_t NDIM, size_t Rank, typename ArrayType, size_t... Is>
constexpr void fill_levi_civita_impl(ArrayType& out, std::index_sequence<Is...>){
    for_each<Layout::Left>([&](const auto... i){
        out(i...) = levi_civita(std::array<size_t, Rank>{i...});
    }, (Is*0UL + NDIM)...);
}

} // namespace geom::linalg::detail

/*
The column-major buffer of a matrix A is the row-major buffer of A^T. Since inv(A^T) = inv(A)^T
and det(A^T) = det(A), the F-layout overloads reuse the row-major routines on the raw buffers.
*/

template<typename T, size_t N>
void invert_mat(CArray2D<T, N, N>& out, const CArray2D<T, N, N>& mat){
    static_assert(N > 0, "invert_mat requires a compile-time matrix size");
    detail::invert_rowmajor<T, N>(out.data(), mat.data());
}

template<typename T, size_t N>
void invert_mat(FArray2D<T, N, N>& out, const FArray2D<T, N, N>& mat){
    static_assert(N > 0, "invert_mat requires a compile-time matrix size");
    detail::invert_rowmajor<T, N>(out.data(), mat.data());
}


template<typename T, size_t N>
T detLU(const CArray2D<T, N, N>& mat){
    static_assert(N > 0, "detLU requires a compile-time matrix size");
    return detail::det_rowmajor<T, N>(mat.data());
}

template<typename T, size_t N>
T detLU( const FArray2D<T, N, N>& mat){
    static_assert(N > 0, "detLU requires a compile-time matrix size");
    return detail::det_rowmajor<T, N>(mat.data());
}


template<size_t N>
constexpr int levi_civita(const std::array<size_t, N> idx){
    // For all i in idx: 0 <= i < N
    int inversions = 0;

    // Check for repeated indices
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (idx[i] == idx[j]){
                return 0;
            }
            if (idx[i] > idx[j]){
                ++inversions;
            }
        }
    }

    return (inversions & 1) ? -1 : 1;
}


template<size_t NDIM, size_t Rank, typename ArrayType>
constexpr void fill_levi_civita(ArrayType& out){
    detail::fill_levi_civita_impl<NDIM, Rank>(out, std::make_index_sequence<Rank>{});
}

} // namespace geom::linalg