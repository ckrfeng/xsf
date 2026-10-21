#include "../testing_utils.h"
#include <array>
#include <catch2/catch_template_test_macros.hpp>
#include <cstdint>
#include <xsf/stats.h>
#include <xsf/third_party/kokkos/mdspan.hpp>

namespace {
template <typename InputMat, typename OutputMat>
void poisson_binom_sf(InputMat p, OutputMat res) {
    std::vector<double> workspace(3 * (p.extent(0) + 1));
    xsf::poisson_binom_sf_all(p, res, std::mdspan(workspace.data(), workspace.size()));
}
} // namespace

TEST_CASE("poisson_binom_pmf test", "[poisson_binom_pmf][xsf_tests]") {
    // Reference values computed from scipy.stats._stats_pythran
    // import numpy as np
    // from scipy.stats._stats_pythran import _poisson_binom_pmf
    // ps = np.linspace(1e-5, 1 - 1e-5, 51)
    // expected = _poisson_binom_pmf(ps)

    const int n_trials = 51;
    const double start = 1e-5;
    const double end = 1.0 - 1e-5;
    std::vector<double> ps = linspace(start, end, n_trials);

    const std::vector<double> expected = {
        3.42860361e-26, 3.43456678e-21, 6.00201383e-19, 4.62011334e-17, 2.11987726e-15, 6.59045259e-14, 1.49189489e-12,
        2.57589301e-11, 3.50401547e-10, 3.84627529e-09, 3.46984905e-08, 2.61005054e-07, 1.65616184e-06, 8.94950358e-06,
        4.15099219e-05, 1.66345550e-04, 5.79119575e-04, 1.75968880e-03, 4.68498249e-03, 1.09649150e-02, 2.26209315e-02,
        4.12283129e-02, 6.65032178e-02, 9.50736474e-02, 1.20585365e-01, 1.35781059e-01, 1.35781059e-01, 1.20585365e-01,
        9.50736474e-02, 6.65032178e-02, 4.12283129e-02, 2.26209315e-02, 1.09649150e-02, 4.68498249e-03, 1.75968880e-03,
        5.79119575e-04, 1.66345550e-04, 4.15099219e-05, 8.94950358e-06, 1.65616184e-06, 2.61005054e-07, 3.46984905e-08,
        3.84627529e-09, 3.50401547e-10, 2.57589301e-11, 1.49189489e-12, 6.59045259e-14, 2.11987726e-15, 4.62011334e-17,
        6.00201383e-19, 3.43456678e-21, 3.42860361e-26
    };

    const double rtol = 1e-8;
    std::vector<double> results = std::vector(n_trials + 1, 0.0);

    std::mdspan p_span(ps.data(), ps.size());
    std::mdspan res_span(results.data(), results.size());

    xsf::poisson_binom_pmf_all(p_span, res_span);

    for (int i = 0; i < n_trials + 1; ++i) {
        const double rel_error = xsf::extended_relative_error(results[i], expected[i]);
        CAPTURE(i, results[i], expected[i], rtol, rel_error);
        REQUIRE(rel_error <= rtol);
    }
}

TEST_CASE("poisson_binom_cdf test", "[poisson_binom_cdf][xsf_tests]") {
    // Reference values computed as follows
    // import numpy as np
    // import itertools as it
    // from mpmath import mp
    // from scipy.stats._stats_pythran import _poisson_binom_pmf
    // ps = np.linspace(1e-5, 1 - 1e-5, 51)
    // expected_pmf = _poisson_binom_pmf(ps)
    // expected_cdf = list(it.accumulate([mp.mpf(p) for p in expected_pmf]))
    // expected_cdf = np.clip(expected_cdf, 0, 1)
    const int n_trials = 51;
    const double start = 1e-5;
    const double end = 1.0 - 1e-5;
    std::vector<double> ps = linspace(start, end, n_trials);

    const std::vector<double> expected = {
        3.42860361e-26, 3.43460107e-21, 6.03635984e-19, 4.68047694e-17, 2.16668202e-15, 6.80712079e-14, 1.55996609e-12,
        2.73188962e-11, 3.77720443e-10, 4.22399573e-09, 3.89224863e-08, 2.99927541e-07, 1.95608939e-06, 1.09055930e-05,
        5.24155149e-05, 2.18761065e-04, 7.97880640e-04, 2.55756944e-03, 7.24255193e-03, 1.82074669e-02, 4.08283984e-02,
        8.20567113e-02, 1.48559929e-01, 2.43633576e-01, 3.64218941e-01, 5.00000000e-01, 6.35781059e-01, 7.56366424e-01,
        8.51440071e-01, 9.17943289e-01, 9.59171602e-01, 9.81792533e-01, 9.92757448e-01, 9.97442431e-01, 9.99202119e-01,
        9.99781239e-01, 9.99947584e-01, 9.99989094e-01, 9.99998044e-01, 9.99999700e-01, 9.99999961e-01, 9.99999996e-01,
        1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
        1.00000000e+00, 1.00000000e+00, 1.00000000e+00
    };

    const double rtol = 1e-8;
    std::vector<double> results = std::vector(n_trials + 1, 0.0);

    std::mdspan p_span(ps.data(), ps.size());
    std::mdspan res_span(results.data(), results.size());

    xsf::poisson_binom_cdf_all(p_span, res_span);

    for (int i = 0; i < n_trials + 1; ++i) {
        const double rel_error = xsf::extended_relative_error(results[i], expected[i]);
        CAPTURE(i, results[i], expected[i], rtol, rel_error);
        REQUIRE(rel_error <= rtol);
    }
}

TEMPLATE_TEST_CASE("poisson_binom_sf table", "[poisson_binom_sf][xsf_tests]", float, double) {
    using T = TestType;
    std::array<T, 4> ps = {T(1) / 8, T(1) / 4, T(1) / 2, T(3) / 4};
    // Exact upper-tail probabilities for these dyadic inputs.
    const std::array<T, 5> expected = {T(235) / 256, T(141) / 256, T(37) / 256, T(3) / 256, T(0)};
    std::array<T, 5> results;
    results.fill(T(-1));

    poisson_binom_sf(std::mdspan(ps.data(), ps.size()), std::mdspan(results.data(), results.size()));

    REQUIRE(results == expected);
}

TEMPLATE_TEST_CASE("poisson_binom_sf small upper tails", "[poisson_binom_sf][xsf_tests]", float, double) {
    using T = TestType;
    std::vector<T> ps;
    std::size_t k = 0;
    T expected = T(0);

    // scipy/scipy#26409: subtracting the CDF from one loses these tails.
    SECTION("heterogeneous probabilities") {
        for (int i = 0; i < 32; ++i) {
            ps.push_back(T(1 + i % 9) / 128);
        }
        k = 23;
        // Computed by convolving Bernoulli distributions using exact rational arithmetic.
        expected = T(1.1963531710514960e-29);
    }
    SECTION("probabilities whose complements round to one") {
        ps.assign(16, T(1e-20));
        expected = T(1.6e-19);
    }
    SECTION("smallest subnormal input") {
        ps.assign(16, std::numeric_limits<T>::denorm_min());
        expected = T(16) * std::numeric_limits<T>::denorm_min();
    }
    SECTION("tail below the floating-point range") {
        ps.assign(256, T(1) / 128);
        k = 249;
        // The exact probability is about 5.54e-516, which rounds to zero.
    }

    std::vector<T> results(ps.size() + 1, T(-1));
    poisson_binom_sf(std::mdspan(ps.data(), ps.size()), std::mdspan(results.data(), results.size()));

    const double rtol = 32 * std::numeric_limits<T>::epsilon();
    const double rel_error = xsf::extended_relative_error(results[k], expected);
    CAPTURE(k, results[k], expected, rel_error, rtol);
    REQUIRE(rel_error <= rtol);
    REQUIRE(results.back() == T(0));
    for (std::size_t i = 0; i < results.size(); ++i) {
        REQUIRE(results[i] >= T(0));
        REQUIRE(results[i] <= T(1));
        if (i > 0) {
            REQUIRE(results[i] <= results[i - 1]);
        }
    }
}

TEMPLATE_TEST_CASE("poisson_binom_sf intermediate underflow", "[poisson_binom_sf][xsf_tests]", float, double) {
    using T = TestType;
    const bool is_float = std::is_same_v<T, float>;
    const T p = is_float ? T(1.7e-23) : T(1e-162);
    const auto n = GENERATE(2, 3, 4, 8, 16);
    const int padding = GENERATE(0, 2);
    // Correctly rounded references from exact rational Bernoulli convolution.
    const int units = n == 2 ? 0 : n <= 4 ? 1 : n == 8 ? 6 : is_float ? 25 : 24;
    const T expected = T(units) * std::numeric_limits<T>::denorm_min();
    std::vector<T> ps(n, p);
    if (padding) {
        // Include deterministic trials to check that support shifts preserve the tail.
        ps.insert(ps.begin(), T(0));
        ps.insert(ps.begin(), T(1));
        ps.push_back(T(1));
        ps.push_back(T(0));
    }
    std::vector<T> results(ps.size() + 1, T(-1));

    poisson_binom_sf(std::mdspan(ps.data(), ps.size()), std::mdspan(results.data(), results.size()));

    CAPTURE(n, padding, results[1 + padding], expected);
    REQUIRE(results[1 + padding] == expected);
}

TEMPLATE_TEST_CASE("poisson_binom_sf subnormal rounding", "[poisson_binom_sf][xsf_tests]", float, double) {
    using T = TestType;
    const bool is_float = std::is_same_v<T, float>;
    const int n = is_float ? 4 : 3;
    const T center = is_float ? T(0x1.a20bd8p-77f) : T(0x1.a20bd700c2c3ep-539);
    const int direction = GENERATE(-1, 0, 1);
    const int shift = GENERATE(0, 2);
    const T p = direction == 0 ? center : std::nextafter(center, direction < 0 ? T(0) : T(1));
    std::vector<T> ps(n, p);
    if (shift) {
        ps.insert(ps.begin(), T(1));
        ps.push_back(T(1));
        ps.push_back(T(0));
    }
    std::vector<T> results(ps.size() + 1, T(-1));

    poisson_binom_sf(std::mdspan(ps.data(), ps.size()), std::mdspan(results.data(), results.size()));

    // Exact rational convolution puts center just above the zero/subnormal
    // midpoint, and its predecessor just below it, for both types.
    T expected = direction < 0 ? T(0) : std::numeric_limits<T>::denorm_min();
    CAPTURE(n, direction, shift, results[1 + shift], expected);
    REQUIRE(results[1 + shift] == expected);
}

TEMPLATE_TEST_CASE("poisson_binom_sf subnormal ties", "[poisson_binom_sf][xsf_tests]", float, double) {
    using T = TestType;
    constexpr int exponent = std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits - 1;
    const T first = std::ldexp(T(1), exponent / 2);
    const T second = std::ldexp(T(1), exponent - exponent / 2);
    const int direction = GENERATE(-1, 0, 1);
    const bool extra_trial = GENERATE(false, true);
    const bool reverse = GENERATE(false, true);
    const int shift = GENERATE(0, 2);
    T p = direction == 0 ? first : std::nextafter(first, direction < 0 ? T(0) : T(1));
    std::vector<T> ps = {p, second};
    if (extra_trial) {
        ps.push_back(std::numeric_limits<T>::denorm_min());
    }
    if (shift) {
        ps.insert(ps.begin(), T(1));
        ps.push_back(T(1));
    }
    if (reverse) {
        std::reverse(ps.begin(), ps.end());
    }
    std::vector<T> results(ps.size() + 1, T(-1));

    poisson_binom_sf(std::mdspan(ps.data(), ps.size()), std::mdspan(results.data(), results.size()));

    // first * second is exactly half the smallest subnormal. An additional
    // possible success makes the tail strictly larger than the midpoint.
    const bool positive = direction > 0 || (direction == 0 && extra_trial);
    T expected = positive ? std::numeric_limits<T>::denorm_min() : T(0);
    CAPTURE(direction, extra_trial, reverse, shift, results[1 + shift], expected);
    REQUIRE(results[1 + shift] == expected);
}

TEMPLATE_TEST_CASE("poisson_binom_sf higher-order ties", "[poisson_binom_sf][xsf_tests]", float, double) {
    using T = TestType;
    constexpr int exponent = std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits - 1;
    const int successes = GENERATE(3, 4, 8, 25, 64);
    const int direction = GENERATE(-1, 0, 1);
    const bool extra_trial = GENERATE(false, true);
    const bool reverse = GENERATE(false, true);
    const int shift = GENERATE(0, 2);
    std::vector<T> ps(successes, std::ldexp(T(1), exponent / successes));
    ps.back() = std::ldexp(T(1), exponent - (successes - 1) * (exponent / successes));
    if (direction) {
        ps.front() = std::nextafter(ps.front(), direction < 0 ? T(0) : T(1));
    }
    if (extra_trial) {
        ps.push_back(std::numeric_limits<T>::denorm_min());
    }
    if (shift) {
        ps.insert(ps.begin(), T(1));
        ps.push_back(T(1));
        ps.push_back(T(0));
    }
    if (reverse) {
        std::reverse(ps.begin(), ps.end());
    }
    std::vector<T> results(ps.size() + 1, T(-1));

    poisson_binom_sf(std::mdspan(ps.data(), ps.size()), std::mdspan(results.data(), results.size()));

    // The product is exactly half the smallest subnormal before perturbation.
    // An extra possible success moves this tail strictly above the midpoint.
    const bool positive = direction > 0 || (direction == 0 && extra_trial);
    const T expected = positive ? std::numeric_limits<T>::denorm_min() : T(0);
    CAPTURE(successes, direction, extra_trial, reverse, shift, results[successes - 1 + shift], expected);
    REQUIRE(results[successes - 1 + shift] == expected);
}

TEST_CASE("poisson_binom_sf higher-order binomial rounding", "[poisson_binom_sf][xsf_tests]") {
    const int fixture = GENERATE(0, 1, 2, 3, 4, 5);
    const int ns[] = {4, 5, 12, 32, 32, 32};
    const int successes[] = {3, 3, 5, 7, 7, 8};
    const float ps[] = {0x1.428a30p-51f, 0x1.db4c78p-52f, 0x1.0d7f64p-32f,
                        0x1.638db2p-25f, 0x1.638db4p-25f, 0x1.42bbe2p-22f};
    std::vector<float> probabilities(ns[fixture], ps[fixture]);
    std::vector<float> results(probabilities.size() + 1, -1);

    poisson_binom_sf(
        std::mdspan(probabilities.data(), probabilities.size()), std::mdspan(results.data(), results.size())
    );

    // Exact rational binomial sums are just above the zero/subnormal midpoint.
    CAPTURE(fixture, ns[fixture], successes[fixture], results[successes[fixture] - 1]);
    REQUIRE(results[successes[fixture] - 1] == std::numeric_limits<float>::denorm_min());
}

TEST_CASE("poisson_binom_sf workspace size", "[poisson_binom_sf][xsf_tests]") {
    std::array<double, 2> ps = {0.25, 0.75};
    std::array<double, 3> results = {-1, -1, -1};
    auto size = GENERATE(std::size_t(0), std::size_t(8), std::size_t(10));
    std::vector<double> workspace(size, -1);

    xsf::poisson_binom_sf_all(
        std::mdspan(ps.data(), ps.size()), std::mdspan(results.data(), results.size()),
        std::mdspan(workspace.data(), workspace.size())
    );

    REQUIRE(std::all_of(results.begin(), results.end(), [](double x) { return x == -1; }));
    REQUIRE(std::all_of(workspace.begin(), workspace.end(), [](double x) { return x == -1; }));
}

TEMPLATE_TEST_CASE("poisson_binom_sf strided workspace", "[poisson_binom_sf][xsf_tests]", float, double) {
    using T = TestType;
    constexpr int exponent = std::numeric_limits<T>::min_exponent - std::numeric_limits<T>::digits - 1;
    std::array<T, 4> ps = {
        std::ldexp(T(1), exponent / 3), std::ldexp(T(1), exponent / 3), std::ldexp(T(1), exponent - 2 * (exponent / 3)),
        std::numeric_limits<T>::denorm_min()
    };
    std::array<T, 5> results;
    auto stride = GENERATE(std::size_t(1), std::size_t(3));
    std::array<double, 45> workspace;
    workspace.fill(-1);
    using Extents = std::dextents<std::size_t, 1>;
    using Mapping = std::layout_stride::mapping<Extents>;
    std::mdspan<double, Extents, std::layout_stride> scratch(
        workspace.data(), Mapping(Extents(15), std::array<std::size_t, 1>{stride})
    );

    xsf::poisson_binom_sf_all(std::mdspan(ps.data(), ps.size()), std::mdspan(results.data(), results.size()), scratch);

    REQUIRE(results[2] == std::numeric_limits<T>::denorm_min());
    for (std::size_t i = 0; i < workspace.size(); ++i) {
        if (i % stride != 0 || i / stride >= 15) {
            REQUIRE(workspace[i] == -1);
        }
    }
}

TEMPLATE_TEST_CASE("poisson_binom_sf boundaries", "[poisson_binom_sf][xsf_tests]", float, double) {
    using T = TestType;
    std::vector<T> ps;
    std::vector<T> expected;

    SECTION("no trials") { expected = {T(0)}; }
    SECTION("one trial") {
        T p = GENERATE(T(0), T(0.125), T(1));
        ps = {p};
        expected = {p, T(0)};
    }
    SECTION("deterministic trials") {
        ps = {T(0), T(1), T(0), T(1)};
        expected = {T(1), T(1), T(0), T(0), T(0)};
    }

    std::vector<T> results(ps.size() + 1, T(-1));
    poisson_binom_sf(std::mdspan(ps.data(), ps.size()), std::mdspan(results.data(), results.size()));
    REQUIRE(results == expected);
}

TEST_CASE("poisson_binom_sf output size", "[poisson_binom_sf][xsf_tests]") {
    std::array<double, 2> ps = {0.25, 0.75};
    auto size = GENERATE(std::size_t(0), std::size_t(2), std::size_t(4));
    std::vector<double> results(size, -1.0);

    poisson_binom_sf(std::mdspan(ps.data(), ps.size()), std::mdspan(results.data(), results.size()));

    for (double value : results) {
        REQUIRE(value == -1.0);
    }
}

TEMPLATE_TEST_CASE("poisson_binom_sf strided views", "[poisson_binom_sf][xsf_tests]", float, double) {
    using T = TestType;
    using Extents = std::dextents<std::size_t, 1>;
    using Mapping = std::layout_stride::mapping<Extents>;
    auto input_stride = GENERATE(std::size_t(1), std::size_t(2));
    auto output_stride = GENERATE(std::size_t(1), std::size_t(3));
    std::array<T, 4> ps = {T(0.25), T(0.75), T(-1), T(-1)};
    if (input_stride == 2) {
        ps = {T(0.25), T(-1), T(0.75), T(-1)};
    }
    std::array<T, 9> results;
    results.fill(T(-1));
    std::mdspan<const T, Extents, std::layout_stride> p_span(
        ps.data(), Mapping(Extents(2), std::array<std::size_t, 1>{input_stride})
    );
    std::mdspan<T, Extents, std::layout_stride> res_span(
        results.data(), Mapping(Extents(3), std::array<std::size_t, 1>{output_stride})
    );

    poisson_binom_sf(p_span, res_span);

    const std::array<T, 3> expected = {T(0.8125), T(0.1875), T(0)};
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i % output_stride == 0 && i / output_stride < expected.size()) {
            REQUIRE(results[i] == expected[i / output_stride]);
        } else {
            REQUIRE(results[i] == T(-1));
        }
    }
}

TEST_CASE("take_from_discrete_sf boundaries", "[poisson_binom_sf][xsf_tests]") {
    std::array<double, 3> values = {0.75, 0.25, 0.0};
    std::mdspan sf(values.data(), values.size());

    REQUIRE(xsf::take_from_discrete_sf(sf, -1) == 1.0);
    REQUIRE(xsf::take_from_discrete_sf(sf, std::numeric_limits<std::int64_t>::min()) == 1.0);
    REQUIRE(xsf::take_from_discrete_sf(sf, 0) == 0.75);
    REQUIRE(xsf::take_from_discrete_sf(sf, 1) == 0.25);
    REQUIRE(xsf::take_from_discrete_sf(sf, 2) == 0.0);
    REQUIRE(xsf::take_from_discrete_sf(sf, 3) == 0.0);
    REQUIRE(xsf::take_from_discrete_sf(sf, std::numeric_limits<std::int64_t>::max()) == 0.0);
    REQUIRE(xsf::take_from_discrete_sf(sf, std::numeric_limits<std::uint64_t>::max()) == 0.0);

    std::array<double, 1> empty_distribution = {0.0};
    std::mdspan empty_sf(empty_distribution.data(), empty_distribution.size());
    REQUIRE(xsf::take_from_discrete_sf(empty_sf, -1) == 1.0);
    REQUIRE(xsf::take_from_discrete_sf(empty_sf, 0) == 0.0);
}
