#pragma once


#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Fastor/Fastor.h>


namespace geom{

enum class Loc : uint8_t{
    Up, Dn,
};

// As template parameter on a tensor
template<Loc Type, size_t N>
struct Ax{
    static constexpr Loc idx_type = Type;
    static constexpr size_t Num = N;
};

// To name an index: Idx<'i'> i; Idx<'j'> j; A(i, j) = B(j, i);
template<char Label>
struct Idx{
    static constexpr char Name = Label;
};


template<typename T, typename... Axes>
class Tensor; // holds a fastor tensor


namespace detail{

struct NamedIdx{
    char name;
    Loc type;
    size_t range;

    friend constexpr bool operator==(const NamedIdx&, const NamedIdx&) = default;
};

template<size_t Rank>
using FreeIdxColl = std::array<NamedIdx, Rank>;

template<size_t Size>
using ContractedIdxColl = std::array<char, Size>;

// Created temporarily when doing SomeTensor(I, J, ...),
// from each Ax of the tensor and each name of I, J, ...
template<char Label, Loc Type, size_t N>
struct AxisProxy : Ax<Type, N>{
    static constexpr char Name = Label;
};

template<typename... AxProx>
struct IndexCollection{
    /*
    An index collection of a virtual tensor.
    This can be the indices of e.g. A(mu, nu), or of B(a, a)
    Or it can be from a product of tensors (the index collection
    of the resulting tensor)
    For example for A(a, b) * B(mu, b), this can be {a, b, mu, b}
    and the index collection of the free indices is {a, mu}, while
    that of the contracted ones is {b}
    */

    /*
    Compile time stuff:

    1) The FreeIdxColl of the free indices is made of the indices
    whose label appears only once, in their original order.
    2) The ContractedIdxColl is constructed from the pairs of
    repeating indices. If a label appears twice it is added on the list.
    If it appears more than twice, or if both instances have the same Loc,
    or different ranges, a compile time error is raised.
    */

    using Indices = IndexCollection<AxProx...>;

    static constexpr size_t Count = sizeof...(AxProx);
    static constexpr std::array<NamedIdx, Count> IdxList{
        NamedIdx{.name = AxProx::Name, .type = AxProx::idx_type, .range = AxProx::Num}...};

    static constexpr bool valid_einsum(){
        for (size_t i = 0; i + 1 < Count; ++i){
            size_t rep_count = 1;
            for (size_t j = i + 1; j < Count; ++j){
                if (IdxList[i].name == IdxList[j].name){
                    // Einstein summation rules
                    bool is_bad = (++rep_count == 3);
                    is_bad = is_bad || (IdxList[i].type == IdxList[j].type);
                    is_bad = is_bad || (IdxList[i].range != IdxList[j].range);
                    if (is_bad){
                        return false;
                    }
                }
            }
        }
        return true;
    }

    static_assert(valid_einsum(), "Operation does not satisfy the einstein summation rules: a label can appear at most twice, "
                                  "and a repeated label must be once Up and once Dn, with the same range");

    static constexpr size_t occurrences(char name){
        size_t res = 0;
        for (const NamedIdx& idx : IdxList){
            if (idx.name == name){
                ++res;
            }
        }
        return res;
    }

    // Whether IdxList[i] is the first of a pair of contracted indices
    static constexpr bool opens_pair(size_t i){
        if (occurrences(IdxList[i].name) != 2){
            return false;
        }
        for (size_t j = 0; j < i; ++j){
            if (IdxList[j].name == IdxList[i].name){
                return false;
            }
        }
        return true;
    }

    static constexpr size_t contraction_count(){
        size_t res = 0;
        for (size_t i = 0; i < Count; ++i){
            if (opens_pair(i)){
                ++res;
            }
        }
        return res;
    }

    // Counted directly (not as Count - 2*Ncontr), so that an invalid collection
    // only triggers the static_assert above instead of a cascade of errors
    static constexpr size_t free_count(){
        size_t res = 0;
        for (const NamedIdx& idx : IdxList){
            if (occurrences(idx.name) == 1){
                ++res;
            }
        }
        return res;
    }

    static constexpr size_t NFree = free_count();
    static constexpr size_t Ncontr = contraction_count();

    static constexpr ContractedIdxColl<Ncontr> get_contracted(){
        ContractedIdxColl<Ncontr> res{};
        size_t k = 0;
        for (size_t i = 0; i < Count; ++i){
            if (opens_pair(i)){
                res[k++] = IdxList[i].name;
            }
        }
        return res;
    }

    static constexpr FreeIdxColl<NFree> get_free(){
        FreeIdxColl<NFree> res{};
        size_t k = 0;
        for (const NamedIdx& idx : IdxList){
            if (occurrences(idx.name) == 1){
                res[k++] = idx;
            }
        }
        return res;
    }

    static constexpr FreeIdxColl<NFree> FreeList = get_free();

    // returns a IndexCollection with just the free indices
    static constexpr auto contracted(){
        return []<size_t... I>(std::index_sequence<I...>){
            return IndexCollection<AxisProxy<FreeList[I].name, FreeList[I].type, FreeList[I].range>...>{};
        }(std::make_index_sequence<NFree>{});
    }

};

// {mu, a} * {a, nu} -> {mu, a, a, nu}
template<typename L, typename R>
struct Concat;

template<typename... A, typename... B>
struct Concat<IndexCollection<A...>, IndexCollection<B...>>{
    using type = IndexCollection<A..., B...>;
};

// Whether two collections of free indices can be added: the same labels, each with the
// same Loc and range, in any order. A_mu_nu + B_nu_mu is allowed, A^mu_nu + B_mu_nu is not.
template<typename... A, typename... B>
constexpr bool addable(IndexCollection<A...>, IndexCollection<B...>){
    using L = IndexCollection<A...>;
    using R = IndexCollection<B...>;
    if (L::NFree != L::Count || R::NFree != R::Count || L::Count != R::Count){
        return false;
    }
    for (const NamedIdx& l : L::IdxList){
        bool found = false;
        for (const NamedIdx& r : R::IdxList){
            found = found || (l == r);
        }
        if (!found){
            return false;
        }
    }
    return true;
}

// Anything carrying an IndexCollection that can be evaluated: A(mu, nu), or a sum, product or scaling of those
template<typename E>
concept IndexExpr = requires { typename E::Indices; typename E::value_type; } && std::derived_from<E, typename E::Indices>;

template<typename S>
concept Scalar = std::is_arithmetic_v<S>;


/*
Evaluation, done by Fastor

The terms of a sum are evaluated one at a time into the assigned tensor: the first one is
assigned, the next ones are added or subtracted element-wise (Fastor expressions, vectorized),
with their scalar factors applied in the same pass.

Each term is a product of tensors, contracted two at a time by Fastor::einsum. The order of the
pairwise contractions is chosen at compile time to minimize the number of multiply-adds
(ContractionPlan). Before each contraction, the two tensors are transposed if needed so that the
summed indices are last in the first one and first in the second one, in the same order: that is
the layout for which Fastor uses its matrix-multiply kernels (Arrangement). The result is then
transposed into the order of the assigned tensor if needed. A sum inside a product, like
(A + B) * C, is evaluated first into a temporary, which is then one of the factors.
*/

// The labels are used as Fastor index values: A(mu, nu) -> Fastor::Index<'m', 'n'>
template<typename IC>
struct FastorIdx;

template<typename... A>
struct FastorIdx<IndexCollection<A...>>{
    using index = Fastor::Index<static_cast<unsigned char>(A::Name)...>;
    using oindex = Fastor::OIndex<static_cast<unsigned char>(A::Name)...>;

    template<typename T>
    using tensor = Fastor::Tensor<T, A::Num...>;
};

// A tensor factor of a product, with the labels of its axes
template<typename FT, typename Labels>
struct RefFactor{
    using labels = Labels;
    const FT& t;

    const FT& get() const{
        return t;
    }
};

// Same, holding a temporary: a sum evaluated first, or a tensor with a trace reduced first
template<typename FT, typename Labels>
struct OwnedFactor{
    using labels = Labels;
    FT t;

    const FT& get() const{
        return t;
    }
};

template<typename Labels, typename FT>
OwnedFactor<FT, Labels> owned_factor(FT t){
    return {std::move(t)};
}

// Marks the absence of a scalar factor, so that nothing gets multiplied by 1
struct NoCoef{};

template<typename A, typename B>
auto mul_coef(const A& a, const B& b){
    if constexpr (std::is_same_v<A, NoCoef>){
        return b;
    }
    else if constexpr (std::is_same_v<B, NoCoef>){
        return a;
    }
    else{
        return a * b;
    }
}

// A product, split into its scalar factors (multiplied together) and its tensor factors
template<typename C, typename... F>
struct Term{
    C coef;
    std::tuple<F...> factors;
};

template<typename C, typename... F>
Term<C, F...> make_term(const C& coef, F... factors){
    return {.coef = coef, .factors = std::tuple<F...>{std::move(factors)...}};
}

template<typename C1, typename... F1, typename C2, typename... F2>
auto combine_terms(Term<C1, F1...> a, Term<C2, F2...> b){
    using C = decltype(mul_coef(a.coef, b.coef));
    return Term<C, F1..., F2...>{
        .coef = mul_coef(a.coef, b.coef),
        .factors = std::tuple_cat(std::move(a.factors), std::move(b.factors)),
    };
}

template<typename S, typename C, typename... F>
auto scale_term(const S& scalar, Term<C, F...> term){
    using Coef = decltype(mul_coef(scalar, term.coef));
    return Term<Coef, F...>{.coef = mul_coef(scalar, term.coef), .factors = std::move(term.factors)};
}

// How a term goes into the assigned tensor
enum class Mode : uint8_t{
    Assign, Add, Sub,
};

// out = coef * x, out += coef * x or out -= coef * x, element-wise by Fastor
template<Mode M, typename FT, typename X, typename C>
void store_term(FT& out, const X& x, const C& coef){
    if constexpr (std::is_same_v<C, NoCoef>){
        if constexpr (M == Mode::Assign){
            out = x;
        }
        else if constexpr (M == Mode::Add){
            out += x;
        }
        else{
            out -= x;
        }
    }
    else{
        if constexpr (M == Mode::Assign){
            out = coef * x;
        }
        else if constexpr (M == Mode::Add){
            out += coef * x;
        }
        else{
            out -= coef * x;
        }
    }
}

template<size_t K>
constexpr bool has_label(const std::array<NamedIdx, K>& list, char name){
    for (const NamedIdx& idx : list){
        if (idx.name == name){
            return true;
        }
    }
    return false;
}

template<size_t K>
constexpr NamedIdx find_label(const std::array<NamedIdx, K>& list, char name){
    for (const NamedIdx& idx : list){
        if (idx.name == name){
            return idx;
        }
    }
    return {};
}

template<size_t K>
constexpr size_t volume(const std::array<NamedIdx, K>& list){
    size_t res = 1;
    for (const NamedIdx& idx : list){
        res *= idx.range;
    }
    return res;
}

// The IndexCollection with the indices of a constexpr list
template<const auto& List, typename Seq = std::make_index_sequence<List.size()>>
struct LabelsOf;

template<const auto& List, size_t... I>
struct LabelsOf<List, std::index_sequence<I...>>{
    using type = IndexCollection<AxisProxy<List[I].name, List[I].type, List[I].range>...>;
};

// The positions in From of the labels of To, for Fastor::permute
template<typename From, typename To>
struct PermuteIdx;

template<typename From, typename... B>
struct PermuteIdx<From, IndexCollection<B...>>{
    static constexpr size_t position(char name){
        for (size_t i = 0; i < From::Count; ++i){
            if (From::IdxList[i].name == name){
                return i;
            }
        }
        return 0;
    }
    using type = Fastor::Index<position(B::Name)...>;
};

// A factor with its axes in the order of the labels To: a reference when already in that order,
// otherwise a transposed copy
template<typename To, typename Fac>
decltype(auto) arrange(const Fac& f){
    using From = typename Fac::labels;
    if constexpr (std::is_same_v<From, To>){
        return f.get();
    }
    else{
        return Fastor::permute<typename PermuteIdx<From, To>::type>(f.get());
    }
}

// One way to lay out two factors for Fastor::einsum: the summed labels last in First and first in
// Second, in the same order (the one they have in First if KFromFirst, else in Second). This is the
// layout for which Fastor uses its matrix-multiply kernels; others are several times slower.
template<typename First, typename Second, bool KFromFirst>
struct Arrangement{
    static constexpr size_t NA = First::Count;
    static constexpr size_t NB = Second::Count;

    static constexpr size_t NK = [] {
        size_t res = 0;
        for (const NamedIdx& idx : First::IdxList){
            if (has_label(Second::IdxList, idx.name)){
                ++res;
            }
        }
        return res;
    }();

    // The summed labels, in the chosen order
    static constexpr std::array<char, NK> summed = [] {
        std::array<char, NK> res{};
        size_t k = 0;
        auto collect = [&](const auto& from, const auto& other){
            for (const NamedIdx& idx : from){
                if (has_label(other, idx.name)){
                    res[k++] = idx.name;
                }
            }
        };
        if constexpr (KFromFirst){
            collect(First::IdxList, Second::IdxList);
        }
        else{
            collect(Second::IdxList, First::IdxList);
        }
        return res;
    }();

    // First with the summed labels moved to the end
    static constexpr std::array<NamedIdx, NA> first = [] {
        std::array<NamedIdx, NA> res{};
        size_t k = 0;
        for (const NamedIdx& idx : First::IdxList){
            if (!has_label(Second::IdxList, idx.name)){
                res[k++] = idx;
            }
        }
        for (char name : summed){
            res[k++] = find_label(First::IdxList, name);
        }
        return res;
    }();

    // Second with the summed labels moved to the front
    static constexpr std::array<NamedIdx, NB> second = [] {
        std::array<NamedIdx, NB> res{};
        size_t k = 0;
        for (char name : summed){
            res[k++] = find_label(Second::IdxList, name);
        }
        for (const NamedIdx& idx : Second::IdxList){
            if (!has_label(First::IdxList, idx.name)){
                res[k++] = idx;
            }
        }
        return res;
    }();

    // The result: the remaining labels of First, then those of Second
    static constexpr std::array<NamedIdx, NA + NB - 2 * NK> result = [] {
        std::array<NamedIdx, NA + NB - 2 * NK> res{};
        size_t k = 0;
        for (size_t i = 0; i < NA - NK; ++i){
            res[k++] = first[i];
        }
        for (size_t i = NK; i < NB; ++i){
            res[k++] = second[i];
        }
        return res;
    }();

    // How many elements are copied to get this layout
    static constexpr size_t moved = (first == First::IdxList ? 0 : volume(first)) + (second == Second::IdxList ? 0 : volume(second));
};

// The arrangement of a pair of factors that copies the fewest elements, out of both operand
// orders and both orders of the summed labels
template<typename L, typename R>
struct PairLayout{
    using A0 = Arrangement<L, R, true>;
    using A1 = Arrangement<L, R, false>;
    using A2 = Arrangement<R, L, true>;
    using A3 = Arrangement<R, L, false>;

    static constexpr size_t best = [] {
        const std::array<size_t, 4> moved{A0::moved, A1::moved, A2::moved, A3::moved};
        size_t res = 0;
        for (size_t i = 1; i < moved.size(); ++i){
            if (moved[i] < moved[res]){
                res = i;
            }
        }
        return res;
    }();

    using type = std::conditional_t<best == 0, A0, std::conditional_t<best == 1, A1, std::conditional_t<best == 2, A2, A3>>>;
    static constexpr bool swapped = best >= 2;
};

template<typename Layout, typename P, typename Q>
auto contract_arranged(const P& p, const Q& q){
    using First = typename LabelsOf<Layout::first>::type;
    using Second = typename LabelsOf<Layout::second>::type;
    using Result = typename LabelsOf<Layout::result>::type;
    decltype(auto) a = arrange<First>(p);
    decltype(auto) b = arrange<Second>(q);
    return owned_factor<Result>(Fastor::einsum<typename FastorIdx<First>::index, typename FastorIdx<Second>::index>(a, b));
}

// Two factors contracted by Fastor
template<typename X, typename Y>
auto contract_pair(const X& x, const Y& y){
    using XL = typename X::labels;
    using YL = typename Y::labels;
    if constexpr (XL::Count == 0 && YL::Count == 0){
        auto res = x.get();
        res.data()[0] *= y.get().data()[0];
        return owned_factor<XL>(std::move(res));
    }
    else if constexpr (XL::Count == 0){
        // A fully contracted group, like A(a, b) * B(b, a), is a number
        std::remove_cvref_t<decltype(y.get())> res = x.get().data()[0] * y.get();
        return owned_factor<YL>(std::move(res));
    }
    else if constexpr (YL::Count == 0){
        std::remove_cvref_t<decltype(x.get())> res = y.get().data()[0] * x.get();
        return owned_factor<XL>(std::move(res));
    }
    else if constexpr (PairLayout<XL, YL>::swapped){
        return contract_arranged<typename PairLayout<XL, YL>::type>(y, x);
    }
    else{
        return contract_arranged<typename PairLayout<XL, YL>::type>(x, y);
    }
}

// The order of the pairwise contractions of a product: for every group of factors, the part that is
// contracted last with the rest of the group, so that the total number of multiply-adds is minimal
// (dynamic programming over the groups of factors, like the "optimal" strategy of opt_einsum)
template<typename... F>
struct ContractionPlan{
    static_assert(sizeof...(F) <= 8, "A product can have at most 8 tensors");

    static constexpr uint32_t all = (uint32_t{1} << sizeof...(F)) - 1;

    // The distinct labels of the product, with their ranges
    struct LabelTable{
        std::array<char, 64> name{};
        std::array<size_t, 64> range{};
        size_t count = 0;
    };

    static constexpr LabelTable table = [] {
        LabelTable res{};
        auto add = [&res](const auto& list){
            for (const NamedIdx& idx : list){
                bool known = false;
                for (size_t i = 0; i < res.count; ++i){
                    known = known || res.name[i] == idx.name;
                }
                if (!known){
                    res.name[res.count] = idx.name;
                    res.range[res.count] = idx.range;
                    ++res.count;
                }
            }
        };
        (add(F::labels::IdxList), ...);
        return res;
    }();

    // The labels of a factor, as bits of the table
    template<size_t K>
    static constexpr uint64_t bits_of(const std::array<NamedIdx, K>& list){
        uint64_t res = 0;
        for (const NamedIdx& idx : list){
            for (size_t i = 0; i < table.count; ++i){
                if (table.name[i] == idx.name){
                    res |= uint64_t{1} << i;
                }
            }
        }
        return res;
    }

    // The labels left after contracting a group of factors: those in only one of them
    static constexpr std::array<uint64_t, all + 1> group_labels = [] {
        const std::array<uint64_t, sizeof...(F)> factor{bits_of(F::labels::IdxList)...};
        std::array<uint64_t, all + 1> res{};
        for (uint32_t group = 1; group <= all; ++group){
            res[group] = res[group & (group - 1)] ^ factor[static_cast<size_t>(std::countr_zero(group))];
        }
        return res;
    }();

    static constexpr size_t volume(uint64_t bits){
        size_t res = 1;
        for (; bits != 0; bits &= bits - 1){
            res *= table.range[static_cast<size_t>(std::countr_zero(bits))];
        }
        return res;
    }

    // split[group]: the part of the group contracted last with the rest of it
    static constexpr std::array<uint32_t, all + 1> split = [] {
        std::array<size_t, all + 1> cost{};
        std::array<uint32_t, all + 1> res{};
        for (uint32_t group = 1; group <= all; ++group){
            if (std::has_single_bit(group)){
                continue;
            }
            cost[group] = std::numeric_limits<size_t>::max();
            for (uint32_t part = (group - 1) & group; part != 0; part = (part - 1) & group){
                const uint32_t rest = group ^ part;
                if (part < rest){
                    continue; // each split once
                }
                const size_t c = cost[part] + cost[rest] + volume(group_labels[part] | group_labels[rest]);
                if (c < cost[group]){
                    cost[group] = c;
                    res[group] = part;
                }
            }
        }
        return res;
    }();
};

template<typename Plan, uint32_t Group, typename... F>
auto contract_group(const std::tuple<F...>& factors){
    if constexpr (std::has_single_bit(Group)){
        const auto& f = std::get<static_cast<size_t>(std::countr_zero(Group))>(factors);
        using Fac = std::remove_cvref_t<decltype(f)>;
        return RefFactor<std::remove_cvref_t<decltype(f.get())>, typename Fac::labels>{f.get()};
    }
    else{
        constexpr uint32_t part = Plan::split[Group];
        return contract_pair(contract_group<Plan, part>(factors), contract_group<Plan, Group ^ part>(factors));
    }
}

template<typename Out, typename... F>
auto contract_network(const std::tuple<F...>& factors){
    using Plan = ContractionPlan<F...>;
    auto res = contract_group<Plan, Plan::all>(factors);
    using Labels = typename decltype(res)::labels;
    if constexpr (std::is_same_v<Labels, Out>){
        return std::move(res.t);
    }
    else{
        return Fastor::permute<typename PermuteIdx<Labels, Out>::type>(res.t);
    }
}

// The tensor factors of a term multiplied and contracted, with the indices in the order of Out
template<typename Out, typename... F>
decltype(auto) contract_factors(const std::tuple<F...>& factors){
    if constexpr (sizeof...(F) == 1){
        return arrange<Out>(std::get<0>(factors));
    }
    else{
        return contract_network<Out>(factors);
    }
}

// Evaluates a term into out, with the indices in the order of Out
template<typename Out, Mode M, typename FT, typename Tm, typename C>
void add_term(FT& out, const Tm& term, const C& coef){
    const auto c = mul_coef(coef, term.coef);
    if constexpr (std::tuple_size_v<decltype(term.factors)> == 0){
        // Only scalars, like A(a, a) * B(b, b): the assigned tensor has rank 0
        using V = std::remove_cvref_t<decltype(*out.data())>;
        if constexpr (M == Mode::Assign){
            out.data()[0] = static_cast<V>(c);
        }
        else if constexpr (M == Mode::Add){
            out.data()[0] += static_cast<V>(c);
        }
        else{
            out.data()[0] -= static_cast<V>(c);
        }
    }
    else{
        store_term<M>(out, contract_factors<Out>(term.factors), c);
    }
}

// expr evaluated into a new Fastor tensor, with the indices in the order of Out
template<typename Out, typename E>
auto materialize(const E& expr){
    typename FastorIdx<Out>::template tensor<typename E::value_type> res;
    expr.template accumulate<Out, Mode::Assign>(res, NoCoef{});
    return res;
}

// lhs + rhs, or lhs - rhs. Has the free indices of the left term.
template<typename L, typename R, bool Minus>
struct Sum : decltype(L::contracted()){
    static_assert(std::is_same_v<typename L::value_type, typename R::value_type>,
                  "The terms of a sum must have the same element type");
    using value_type = typename L::value_type;

    L lhs;
    R rhs;

    Sum(const L& l, const R& r) : lhs(l), rhs(r) {}

    // No free indices: a number, e.g. T s = A(a, a) + B(b, b);
    operator value_type() const requires (Sum::NFree == 0){
        return materialize<IndexCollection<>>(*this).data()[0];
    }

    // Inside a product, a sum is evaluated first, and then used as one factor
    auto collect() const{
        using Own = typename Sum::Indices;
        if constexpr (Own::Count == 0){
            return make_term(materialize<Own>(*this).data()[0]);
        }
        else{
            return make_term(NoCoef{}, owned_factor<Own>(materialize<Own>(*this)));
        }
    }

    template<typename Out, Mode M, typename FT, typename C>
    void accumulate(FT& out, const C& coef) const{
        lhs.template accumulate<Out, M>(out, coef);
        constexpr Mode next = (M == Mode::Sub) ? Mode::Sub : Mode::Add;
        constexpr Mode other = (next == Mode::Add) ? Mode::Sub : Mode::Add;
        rhs.template accumulate<Out, Minus ? other : next>(out, coef);
    }

    bool aliases(const void* tensor) const{
        return lhs.aliases(tensor) || rhs.aliases(tensor);
    }
};

// lhs * rhs, with the indices of both
template<typename L, typename R>
struct Product : Concat<typename L::Indices, typename R::Indices>::type{
    static_assert(std::is_same_v<typename L::value_type, typename R::value_type>,
                  "The tensors of a product must have the same element type (required by Fastor::einsum)");
    using value_type = typename L::value_type;

    L lhs;
    R rhs;

    Product(const L& l, const R& r) : lhs(l), rhs(r) {}

    // No free indices: a number, e.g. T s = A(a, b) * B(b, a);
    operator value_type() const requires (Product::NFree == 0){
        return materialize<IndexCollection<>>(*this).data()[0];
    }

    auto collect() const{
        return combine_terms(lhs.collect(), rhs.collect());
    }

    template<typename Out, Mode M, typename FT, typename C>
    void accumulate(FT& out, const C& coef) const{
        add_term<Out, M>(out, collect(), coef);
    }

    bool aliases(const void* tensor) const{
        return lhs.aliases(tensor) || rhs.aliases(tensor);
    }
};

// scalar * expr, with the indices of expr
template<typename S, typename E>
struct Scaled : E::Indices{
    using value_type = typename E::value_type;

    S scalar;
    E expr;

    Scaled(const S& s, const E& e) : scalar(s), expr(e) {}

    // No free indices: a number, e.g. T s = 2 * A(a, a);
    operator value_type() const requires (Scaled::NFree == 0){
        return materialize<IndexCollection<>>(*this).data()[0];
    }

    auto collect() const{
        return scale_term(scalar, expr.collect());
    }

    template<typename Out, Mode M, typename FT, typename C>
    void accumulate(FT& out, const C& coef) const{
        expr.template accumulate<Out, M>(out, mul_coef(coef, scalar));
    }

    bool aliases(const void* tensor) const{
        return expr.aliases(tensor);
    }
};


/*
The only operations defined on index expressions. An expression without free indices, like
g(k, k), also converts implicitly to a number (the conversion operators above and in
VirtualTensor), so it can be used as one: T s = g(k, k);
*/

template<IndexExpr L, IndexExpr R>
auto operator+(const L& lhs, const R& rhs){
    static_assert(addable(L::contracted(), R::contracted()),
                  "The terms of a sum must have the same free indices (same labels, Loc and range), in any order");
    return Sum<L, R, false>(lhs, rhs);
}

template<IndexExpr L, IndexExpr R>
auto operator-(const L& lhs, const R& rhs){
    static_assert(addable(L::contracted(), R::contracted()),
                  "The terms of a difference must have the same free indices (same labels, Loc and range), in any order");
    return Sum<L, R, true>(lhs, rhs);
}

// The indices are appended; IndexCollection checks the summation rules on the result
template<IndexExpr L, IndexExpr R>
auto operator*(const L& lhs, const R& rhs){
    return Product<L, R>(lhs, rhs);
}

template<Scalar S, IndexExpr E>
auto operator*(const S& scalar, const E& expr){
    return Scaled<S, E>(scalar, expr);
}

template<IndexExpr E, Scalar S>
auto operator*(const E& expr, const S& scalar){
    return Scaled<S, E>(scalar, expr);
}


// A(mu, nu): a Tensor with labeled axes, which can be used in an index expression or assigned to.
// T is const when labeling a const Tensor, which can then only be read.
template<typename T, typename... AxProx>
struct VirtualTensor : IndexCollection<AxProx...>{
    using TensorType = Tensor<std::remove_const_t<T>, Ax<AxProx::idx_type, AxProx::Num>...>;
    using TensorRef = std::conditional_t<std::is_const_v<T>, const TensorType&, TensorType&>;
    using value_type = std::remove_const_t<T>;

    TensorRef tensor;

    explicit VirtualTensor(TensorRef t) : tensor(t) {}

    VirtualTensor(const VirtualTensor&) = default;

    // Needed: when both sides have the same type, e.g. A(mu, nu) = B(mu, nu), this is picked over
    // the template below, and the implicit one is deleted because of the reference member
    VirtualTensor& operator=(const VirtualTensor& rhs){
        return assign(rhs);
    }

    template<IndexExpr E>
    VirtualTensor& operator=(const E& rhs){
        return assign(rhs);
    }

    // No free indices: a number, e.g. T s = g(k, k);
    operator value_type() const requires (VirtualTensor::NFree == 0){
        return materialize<IndexCollection<>>(*this).data()[0];
    }

    auto collect() const{
        using Ix = IndexCollection<AxProx...>;
        const auto& data = tensor.fastor();
        if constexpr (Ix::NFree == 0){
            // A scalar: a rank 0 tensor, or a full trace like A(a, a)
            if constexpr (Ix::Count == 0){
                return make_term(data.data()[0]);
            }
            else{
                return make_term(Fastor::einsum<typename FastorIdx<Ix>::index>(data).data()[0]);
            }
        }
        else if constexpr (Ix::NFree == Ix::Count){
            return make_term(NoCoef{}, RefFactor<std::remove_cvref_t<decltype(data)>, Ix>{data});
        }
        else{
            // A trace on some of the axes, like A(a, a, mu). It is reduced on its own first: in a
            // product, Fastor::einsum gives wrong results for a label repeated within one tensor
            using Free = decltype(Ix::contracted());
            return make_term(NoCoef{}, owned_factor<Free>(
                Fastor::einsum<typename FastorIdx<Ix>::index, typename FastorIdx<Free>::oindex>(data)));
        }
    }

    template<typename Out, Mode M, typename FT, typename C>
    void accumulate(FT& out, const C& coef) const{
        add_term<Out, M>(out, collect(), coef);
    }

    bool aliases(const void* other) const{
        return static_cast<const void*>(&tensor) == other;
    }

private:

    template<typename E>
    VirtualTensor& assign(const E& rhs){
        using Lhs = IndexCollection<AxProx...>;
        constexpr bool writable = !std::is_const_v<T>;
        constexpr bool all_free = Lhs::NFree == Lhs::Count;
        constexpr bool same_indices = addable(Lhs::contracted(), E::contracted());
        constexpr bool same_type = std::is_same_v<value_type, typename E::value_type>;
        static_assert(writable, "Cannot assign to a const tensor");
        static_assert(all_free, "The indices of an assigned tensor must all be free (no repeated labels)");
        static_assert(same_indices, "The free indices of the right hand side must be those of the assigned tensor "
                                    "(same labels, Loc and range), in any order");
        static_assert(same_type, "The right hand side must have the element type of the assigned tensor");

        // When a check fails, the evaluation isn't compiled, so that only the failed check is reported
        if constexpr (writable && all_free && same_indices && same_type){
            auto& out = tensor.fastor();
            if (rhs.aliases(&tensor)){
                // e.g. A(mu, nu) = B(mu, nu) + A(nu, mu): the terms are stored one at a time,
                // so the result goes to a temporary while A is still being read
                out = materialize<Lhs>(rhs);
            }
            else{
                rhs.template accumulate<Lhs, Mode::Assign>(out, NoCoef{});
            }
        }
        return *this;
    }
};

} // namespace detail


template<typename T, typename... Axes>
class Tensor{

public:

    using fastor_type = Fastor::Tensor<T, Axes::Num...>;

    static constexpr size_t rank = sizeof...(Axes);

    Tensor() = default;

    explicit Tensor(const fastor_type& data) : data_(data) {}

    fastor_type& fastor(){
        return data_;
    }

    const fastor_type& fastor() const{
        return data_;
    }

    // A(i, j), with Idx<'i'> i; Idx<'j'> j;
    template<char... Labels> requires (sizeof...(Labels) == rank)
    auto operator()(Idx<Labels>...){
        return detail::VirtualTensor<T, detail::AxisProxy<Labels, Axes::idx_type, Axes::Num>...>(*this);
    }

    template<char... Labels> requires (sizeof...(Labels) == rank)
    auto operator()(Idx<Labels>...) const{
        return detail::VirtualTensor<const T, detail::AxisProxy<Labels, Axes::idx_type, Axes::Num>...>(*this);
    }

private:

    fastor_type data_{}; // zero initialized

};

} // namespace geom
