#pragma once

#include "cephes/bdtr.h"
#include "cephes/chdtr.h"
#include "cephes/const.h"
#include "cephes/fdtr.h"
#include "cephes/gdtr.h"
#include "cephes/i0.h"
#include "cephes/incbet.h"
#include "cephes/incbi.h"
#include "cephes/nbdtr.h"
#include "cephes/ndtr.h"
#include "cephes/ndtri.h"
#include "cephes/owens_t.h"
#include "cephes/pdtr.h"
#include "cephes/tukey.h"
#include "config.h"
#include "erf.h"
#include "gamma.h"
#include "tools.h"

namespace xsf {

XSF_HOST_DEVICE inline double bdtr(double k, int n, double p) { return cephes::bdtr(k, n, p); }

XSF_HOST_DEVICE inline float bdtr(float k, int n, float p) {
    return static_cast<float>(bdtr(static_cast<double>(k), n, static_cast<double>(p)));
}

XSF_HOST_DEVICE inline double bdtri(double k, int n, double y) { return cephes::bdtri(k, n, y); }

XSF_HOST_DEVICE inline float bdtri(float k, int n, float y) {
    return static_cast<float>(bdtri(static_cast<double>(k), n, static_cast<double>(y)));
}

XSF_HOST_DEVICE inline double bdtrc(double k, int n, double p) { return cephes::bdtrc(k, n, p); }

XSF_HOST_DEVICE inline float bdtrc(float k, int n, float p) {
    return static_cast<float>(bdtrc(static_cast<double>(k), n, static_cast<double>(p)));
}

XSF_HOST_DEVICE inline double chdtr(double df, double x) { return cephes::chdtr(df, x); }

XSF_HOST_DEVICE inline float chdtr(float df, float x) { return static_cast<float>(cephes::chdtr(df, x)); }

XSF_HOST_DEVICE inline double chdtrc(double df, double x) { return cephes::chdtrc(df, x); }

XSF_HOST_DEVICE inline float chdtrc(float df, float x) { return static_cast<float>(cephes::chdtrc(df, x)); }

XSF_HOST_DEVICE inline double chdtri(double df, double y) { return cephes::chdtri(df, y); }

XSF_HOST_DEVICE inline float chdtri(float df, float y) {
    return static_cast<float>(chdtri(static_cast<double>(df), static_cast<double>(y)));
}

XSF_HOST_DEVICE inline double fdtr(double a, double b, double x) { return cephes::fdtr(a, b, x); }

XSF_HOST_DEVICE inline float fdtr(float a, float b, float x) {
    return static_cast<float>(fdtr(static_cast<double>(a), static_cast<double>(b), static_cast<double>(x)));
}

XSF_HOST_DEVICE inline double fdtrc(double a, double b, double x) { return cephes::fdtrc(a, b, x); }

XSF_HOST_DEVICE inline float fdtrc(float a, float b, float x) {
    return static_cast<float>(fdtrc(static_cast<double>(a), static_cast<double>(b), static_cast<double>(x)));
}

XSF_HOST_DEVICE inline double fdtri(double a, double b, double y) { return cephes::fdtri(a, b, y); }

XSF_HOST_DEVICE inline float fdtri(float a, float b, float y) {
    return static_cast<float>(fdtri(static_cast<double>(a), static_cast<double>(b), static_cast<double>(y)));
}

XSF_HOST_DEVICE inline double gdtr(double a, double b, double x) { return cephes::gdtr(a, b, x); }

XSF_HOST_DEVICE inline float gdtr(float a, float b, float x) {
    return static_cast<float>(gdtr(static_cast<double>(a), static_cast<double>(b), static_cast<double>(x)));
}

XSF_HOST_DEVICE inline double gdtrc(double a, double b, double x) { return cephes::gdtrc(a, b, x); }

XSF_HOST_DEVICE inline float gdtrc(float a, float b, float x) {
    return static_cast<float>(gdtrc(static_cast<double>(a), static_cast<double>(b), static_cast<double>(x)));
}

XSF_HOST_DEVICE XSF_HOST_DEVICE inline double ndtr(double x) { return cephes::ndtr(x); }

XSF_HOST_DEVICE XSF_HOST_DEVICE inline float ndtr(float x) { return ndtr(static_cast<double>(x)); }

XSF_HOST_DEVICE inline cxx::complex<double> ndtr(cxx::complex<double> z) { return 0.5 * erfc(-z * M_SQRT1_2); }

XSF_HOST_DEVICE inline cxx::complex<float> ndtr(cxx::complex<float> z) {
    return static_cast<cxx::complex<float>>(ndtr(static_cast<cxx::complex<double>>(z)));
}

/*
 * Log of the CDF of the normal distribution for double x.
 *
 * Let F(x) be the CDF of the standard normal distribution.
 * This implementation of log(F(x)) is based on the identities
 *
 *   F(x) = erfc(-x/√2)/2
 *        = 1 - erfc(x/√2)/2
 *
 * We use the first formula for x < -1, with erfc(z) replaced
 * by erfcx(z)*exp(-z**2) to ensure high precision for large
 * negative values when we take the logarithm:
 *
 *   log F(x) = log(erfc(-x/√2)/2)
 *            = log(erfcx(-x/√2)/2)*exp(-x**2/2))
 *            = log(erfcx(-x/√2)/2) - x**2/2
 *
 * For x >= -1, we use the second formula for F(x):
 *
 *   log F(x) = log(1 - erfc(x/√2)/2)
 *            = log1p(-erfc(x/√2)/2)
 */
XSF_HOST_DEVICE inline double log_ndtr(double x) {
    double t = x * M_SQRT1_2;
    if (x < -1.0) {
        return log(erfcx(-t) / 2) - t * t;
    } else {
        return log1p(-erfc(t) / 2);
    }
}

XSF_HOST_DEVICE inline float log_ndtr(float x) { return log_ndtr(static_cast<double>(x)); }

/*
 * Log of the normal CDF for complex arguments.
 *
 * This is equivalent to log(ndtr(z)), but is more robust to overflow at $z\to\infty$.
 * This implementation uses $\erfc(z) = \exp(-z^2) w(iz)$ taking special care to select
 * the principal branch of the log function log( exp(-z^2) w(i z) )
 */
XSF_HOST_DEVICE inline cxx::complex<double> log_ndtr(cxx::complex<double> z) {
    if (z.real() > 6) {
        // Underflow. Close to the real axis, expand the log in log(1 - ndtr(-z)).
        cxx::complex<double> w = -0.5 * erfc(z * M_SQRT1_2);
        if (cxx::abs(w) < 1e-8) {
            return w;
        }
    }

    z *= -M_SQRT1_2;
    double x = z.real();
    double y = z.imag();

    /* Compute the principal branch of $log(exp(-z^2))$, using the fact that
     * $log(e^t) = log|e^t| + i Arg(e^t)$, and that if $t = r + is$, then
     * $e^t = e^r (\cos(s) + i \sin(s))$.
     */
    double mRe_z2 = (y - x) * (x + y); // Re(-z^2), being careful of overflow
    double mIm_z2 = -2 * x * y;        // Im(-z^2)

    double im = fmod(mIm_z2, 2.0 * M_PI);
    if (im > M_PI) {
        im -= 2.0 * M_PI;
    }

    cxx::complex<double> val1 = cxx::complex<double>(mRe_z2, im);

    cxx::complex<double> val2 = cxx::log(xsf::wofz(complex<double>(-y, x)));
    cxx::complex<double> result = val1 + val2 - M_LN2;

    /* Again, select the principal branch: log(z) = log|z| + i arg(z), thus
     * the imaginary part of the result should belong to [-pi, pi].
     */
    im = result.imag();
    if (im >= M_PI) {
        im -= 2 * M_PI;
    }
    if (im < -M_PI) {
        im += 2 * M_PI;
    }

    return {result.real(), im};
}

XSF_HOST_DEVICE inline cxx::complex<float> log_ndtr(cxx::complex<float> z) {
    return static_cast<cxx::complex<float>>(log_ndtr(static_cast<cxx::complex<double>>(z)));
}

XSF_HOST_DEVICE inline double nbdtr(int k, int n, double p) { return cephes::nbdtr(k, n, p); }

XSF_HOST_DEVICE inline float nbdtr(int k, int n, float p) {
    return static_cast<float>(nbdtr(k, n, static_cast<double>(p)));
}

XSF_HOST_DEVICE inline double nbdtrc(int k, int n, double p) { return cephes::nbdtrc(k, n, p); }

XSF_HOST_DEVICE inline float nbdtrc(int k, int n, float p) {
    return static_cast<float>(nbdtrc(k, n, static_cast<double>(p)));
}

XSF_HOST_DEVICE inline double nbdtri(int k, int n, double p) { return cephes::nbdtri(k, n, p); }

XSF_HOST_DEVICE inline float nbdtri(int k, int n, float p) {
    return static_cast<float>(nbdtri(k, n, static_cast<double>(p)));
}

XSF_HOST_DEVICE inline double ndtri(double x) { return cephes::ndtri(x); }

XSF_HOST_DEVICE inline float ndtri(float x) { return static_cast<float>(cephes::ndtri(x)); }

XSF_HOST_DEVICE XSF_HOST_DEVICE inline double nrdtrimn(double p, double std, double x) {
    if (cxx::isnan(std) || std <= 0) {
        return cxx::numeric_limits<double>::quiet_NaN();
    }
    if (cxx::isnan(p) || p <= 0 || p >= 1) {
        return cxx::numeric_limits<double>::quiet_NaN();
    }
    if (cxx::isnan(x)) {
        return cxx::numeric_limits<double>::quiet_NaN();
    }
    return x - std * cephes::ndtri(p);
}

XSF_HOST_DEVICE XSF_HOST_DEVICE inline float nrdtrimn(float p, float std, float x) {
    return static_cast<float>(nrdtrimn(static_cast<double>(p), static_cast<double>(std), static_cast<double>(x)));
}

XSF_HOST_DEVICE XSF_HOST_DEVICE inline double nrdtrisd(double mean, double p, double x) {
    if (cxx::isnan(mean) || cxx::isnan(p) || cxx::isnan(x)) {
        return cxx::numeric_limits<double>::quiet_NaN();
    }
    if (p <= 0 || p >= 1) {
        return cxx::numeric_limits<double>::quiet_NaN();
    }
    return (x - mean) / cephes::ndtri(p);
}

XSF_HOST_DEVICE XSF_HOST_DEVICE inline float nrdtrisd(float mean, float p, float x) {
    return static_cast<float>(nrdtrisd(static_cast<double>(mean), static_cast<double>(p), static_cast<double>(x)));
}

XSF_HOST_DEVICE inline double owens_t(double h, double a) { return cephes::owens_t(h, a); }

XSF_HOST_DEVICE inline float owens_t(float h, float a) {
    return static_cast<float>(owens_t(static_cast<double>(h), static_cast<double>(a)));
}

XSF_HOST_DEVICE inline double pdtr(double k, double m) { return cephes::pdtr(k, m); }

XSF_HOST_DEVICE inline float pdtr(float k, float m) {
    return static_cast<float>(pdtr(static_cast<double>(k), static_cast<double>(m)));
}

XSF_HOST_DEVICE inline double pdtrc(double k, double m) { return cephes::pdtrc(k, m); }

XSF_HOST_DEVICE inline float pdtrc(float k, float m) {
    return static_cast<float>(pdtrc(static_cast<double>(k), static_cast<double>(m)));
}

XSF_HOST_DEVICE inline double pdtri(int k, double y) { return cephes::pdtri(k, y); }

XSF_HOST_DEVICE inline float pdtri(int k, float y) { return static_cast<float>(pdtri(k, static_cast<double>(y))); }

XSF_HOST_DEVICE inline double tukeylambdacdf(double x, double lmbda) { return cephes::tukeylambdacdf(x, lmbda); }

XSF_HOST_DEVICE inline float tukeylambdacdf(float x, float lmbda) {
    return static_cast<float>(tukeylambdacdf(static_cast<double>(x), static_cast<double>(lmbda)));
}

namespace detail {

    template <typename InputMat, typename OutputMat>
    XSF_HOST_DEVICE inline void
    poisson_binom_pmf_all_impl(InputMat p, OutputMat res, typename OutputMat::value_type scale = 1) {
        using T = typename OutputMat::value_type;
        auto n = p.extent(0);

        // Initialize output array with zeros.
        for (decltype(n) i = 0; i < n + 1; ++i) {
            res(i) = T(0);
        }

        if (n == 0) {
            res(0) = scale;
            return;
        }

        res(0) = (T(1) - p(0)) * scale;
        res(1) = p(0) * scale;

        for (decltype(n) i = 1; i < n; i++) {
            T p_i = p(i);
            T q_i = 1 - p_i;
            for (decltype(n) j = i + 1; j >= 1; j--) {
                T tmp = res(j - 1) * p_i;
                res(j - 1) *= q_i;
                res(j) += tmp;
            }
        }
    }

    // Two-component arithmetic is used only by the subnormal SF fallback.
    struct poisson_binom_compensated {
        double hi, lo;

        XSF_HOST_DEVICE poisson_binom_compensated add(poisson_binom_compensated rhs) const {
            volatile double sum = hi + rhs.hi;
            volatile double part = sum - hi;
            double error = (hi - (sum - part)) + (rhs.hi - part);
            error += lo + rhs.lo;
            volatile double result = sum + error;
            return {result, error - (result - sum)};
        }

        XSF_HOST_DEVICE poisson_binom_compensated multiply(double p) const {
            volatile double product = hi * p;
            double error = cxx::fma(hi, p, -product) + lo * p;
            volatile double result = product + error;
            return {result, error - (result - product)};
        }
    };

    template <typename T>
    XSF_HOST_DEVICE inline T poisson_binom_round_subnormal(poisson_binom_compensated tail, bool positive_remainder) {
        // Round in units of the output type's smallest subnormal. Keep the
        // low component when deciding which side of a halfway point we are on.
        constexpr int bits = cxx::numeric_limits<T>::digits - 1;
        double high = cxx::ldexp(tail.hi, bits);
        double low = cxx::ldexp(tail.lo, bits);
        double units = cxx::floor(high);
        double difference = (high - units) - 0.5;
        if (difference > -low ||
            (difference == -low && (positive_remainder || (static_cast<cxx::uint64_t>(units) & 1)))) {
            units += 1;
        }
        return static_cast<T>(units) * cxx::numeric_limits<T>::denorm_min();
    }

    template <typename WorkspaceMat>
    struct poisson_binom_workspace {
        WorkspaceMat values;
        decltype(values.extent(0)) offset;

        template <typename Index>
        XSF_HOST_DEVICE poisson_binom_compensated get(Index i) const {
            return {values(i), values(i + offset)};
        }

        template <typename Index>
        XSF_HOST_DEVICE void set(Index i, poisson_binom_compensated value) const {
            values(i) = value.hi;
            values(i + offset) = value.lo;
        }
    };

    template <typename InputMat, typename WorkspaceMat, typename Index, typename T>
    XSF_HOST_DEVICE inline T poisson_binom_sf_subnormal(InputMat p, Index k, WorkspaceMat workspace, T scale) {
        poisson_binom_workspace<WorkspaceMat> masses{workspace, workspace.extent(0) / 3};
        for (Index j = 0; j <= k; ++j) {
            masses.set(j, {0, 0});
        }
        masses.set(0, {static_cast<double>(scale), 0});
        // Process the smallest probabilities last, so their positive
        // contributions to an already accumulated tail are not cancelled.
        auto start = 2 * masses.offset;
        for (decltype(p.extent(0)) i = 0; i < p.extent(0); ++i) {
            double probability = static_cast<double>(p(i));
            auto j = i;
            while (j > 0 && workspace(start + j - 1) < probability) {
                workspace(start + j) = workspace(start + j - 1);
                --j;
            }
            workspace(start + j) = probability;
        }
        poisson_binom_compensated tail{0, 0};
        bool positive_remainder = false;
        for (decltype(p.extent(0)) i = 0; i < p.extent(0); ++i) {
            double probability = workspace(start + i);
            // Once the threshold is reached, further trials cannot undo it.
            // Accumulating only the new crossings avoids cancellation in the tail.
            auto crossing_mass = masses.get(k);
            auto crossing = crossing_mass.multiply(probability);
            // Even a contribution below the workspace's range breaks an exact
            // midpoint tie upward, because crossings only add to the tail.
            positive_remainder |= probability > 0 && crossing_mass.hi > 0 && crossing.hi == 0 && crossing.lo == 0;
            tail = tail.add(crossing);
            for (Index j = k; j > 0; --j) {
                auto mass = masses.get(j);
                auto removed = mass.multiply(probability);
                auto incoming = masses.get(j - 1).multiply(probability);
                masses.set(j, mass.add({-removed.hi, -removed.lo}).add(incoming));
            }
            // Keep the complement exact even when 1-p rounds to one.
            auto mass = masses.get(0);
            auto removed = mass.multiply(probability);
            masses.set(0, mass.add({-removed.hi, -removed.lo}));
        }
        return poisson_binom_round_subnormal<T>(tail, positive_remainder);
    }

    template <typename InputMat, typename WorkspaceMat, typename Index, typename T>
    XSF_HOST_DEVICE inline T poisson_binom_sf_round(InputMat p, Index k, WorkspaceMat workspace, T tail, T scale) {
        if constexpr (cxx::is_same_v<T, float> || cxx::is_same_v<T, double>) {
            if (tail > T(0) && tail <= T(1)) {
                constexpr int bits = cxx::numeric_limits<T>::digits - 1;
                double units = cxx::ldexp(static_cast<double>(tail), bits);
                double midpoint_distance = cxx::abs((units - cxx::floor(units)) - 0.5);
                // Recompute only near the zero/subnormal midpoint. Repeating
                // the convolution for every subnormal would be too expensive.
                double tolerance = 8 * static_cast<double>(p.extent(0)) * cxx::numeric_limits<T>::epsilon() * units;
                if (units <= 1 && midpoint_distance <= tolerance) {
                    return poisson_binom_sf_subnormal(p, k, workspace, scale);
                }
            }
        }
        return tail * cxx::numeric_limits<T>::min();
    }

} // namespace detail

template <typename InputMat, typename OutputMat>
XSF_HOST_DEVICE inline void poisson_binom_pmf_all(InputMat p, OutputMat res) {
    /* Outputs the full pmf of a Poisson-Binomial distribution
     *
     * p should be an std::mdspan view of a 1d array of length n containing
     * the success probabilities for the n Bernoulli trials. res should be
     * a std::mdspan view of a 1d array of of length n + 1. It is up to
     * the caller to pass valid values for p and res.
     *
     * Upon completion, res(k) will contain the probability of observing k
     * successes for k from 0 to n.
     */
    auto n = p.extent(0);
    auto out_size = res.extent(0);

    if (out_size != n + 1) {
        set_error("_poisson_binom_pmf_all", SF_ERROR_MEMORY, "out.shape[-1] must be p.shape[-1] + 1");
        return;
    }

    detail::poisson_binom_pmf_all_impl(p, res);
}

template <typename InputMat, typename OutputMat>
XSF_HOST_DEVICE inline void poisson_binom_cdf_all(InputMat p, OutputMat res) {
    /* Outputs the full cdf of a Poisson-Binomial distribution */
    using T = typename OutputMat::value_type;
    auto n = p.extent(0);
    auto out_size = res.extent(0);

    if (out_size != n + 1) {
        set_error("_poisson_binom_cdf_all", SF_ERROR_MEMORY, "out.shape[-1] must be p.shape[-1] + 1");
        return;
    }

    detail::poisson_binom_pmf_all_impl(p, res);
    for (decltype(n) i = 1; i < n; i++) {
        res(i) = cxx::min(res(i) + res(i - 1), T(1));
    }
    res(n) = T(1);
}

template <typename InputMat, typename OutputMat, typename WorkspaceMat>
XSF_HOST_DEVICE inline void poisson_binom_sf_all(InputMat p, OutputMat res, WorkspaceMat workspace) {
    /* Outputs the full sf of a Poisson-Binomial distribution.
     * p and res have the same requirements as for poisson_binom_pmf_all.
     * workspace is a disjoint 1d view of 3*(n+1) doubles, used for the
     * compensated convolution when subnormal tails need extra precision.
     * Input, output, and workspace storage must not overlap.
     * Upon completion, res(k) contains P(S > k) for k from 0 to n.
     */
    using T = typename OutputMat::value_type;
    auto n = p.extent(0);
    auto out_size = res.extent(0);

    if (out_size != n + 1) {
        set_error("_poisson_binom_sf_all", SF_ERROR_MEMORY, "out.shape[-1] must be p.shape[-1] + 1");
        return;
    }

    if (workspace.extent(0) != 3 * (n + 1)) {
        set_error("_poisson_binom_sf_all", SF_ERROR_MEMORY, "workspace.shape[-1] must be 3*(p.shape[-1] + 1)");
        return;
    }
    static_assert(cxx::is_same_v<typename WorkspaceMat::value_type, double>, "workspace must contain doubles");

    // A power-of-two scale preserves tiny masses during convolution. Its reciprocal
    // is the smallest normal value, so relevant subnormal tails stay well above
    // underflow until the final conversion back to probabilities.
    const T unscale = cxx::numeric_limits<T>::min();
    const T scale = T(1) / unscale;
    detail::poisson_binom_pmf_all_impl(p, res, scale);
    // Sum the upper tail directly to avoid cancellation in 1 - cdf.
    T tail = T(0);
    for (decltype(n) i = n; i > 0; --i) {
        T mass = res(i);
        res(i) = detail::poisson_binom_sf_round(p, i, workspace, tail, scale);
        tail = cxx::min(tail + mass, scale);
    }
    res(0) = detail::poisson_binom_sf_round(p, decltype(n)(0), workspace, tail, scale);
}

template <typename InputMat, typename Int>
XSF_HOST_DEVICE inline typename InputMat::value_type take_from_pmf(InputMat pmf, Int k) {
    using T = typename InputMat::value_type;
    auto size = pmf.extent(0);
    if (detail::cmp_less(k, 0) || !detail::cmp_less(k, size)) {
        return T(0);
    }
    return pmf(k);
}

template <typename InputMat, typename Int>
XSF_HOST_DEVICE inline typename InputMat::value_type take_from_discrete_cdf(InputMat cdf, Int k) {
    using T = typename InputMat::value_type;
    auto size = cdf.extent(0);
    if (detail::cmp_less(k, 0)) {
        return T(0);
    }
    if (!detail::cmp_less(k, size - 1)) {
        return T(1);
    }
    return cdf(k);
}

template <typename InputMat, typename Int>
XSF_HOST_DEVICE inline typename InputMat::value_type take_from_discrete_sf(InputMat sf, Int k) {
    using T = typename InputMat::value_type;
    auto size = sf.extent(0);
    if (detail::cmp_less(k, 0)) {
        return T(1);
    }
    if (!detail::cmp_less(k, size - 1)) {
        return T(0);
    }
    return sf(k);
}

namespace detail {

    template <typename OutputMat>
    XSF_HOST_DEVICE inline void wilcoxon_pmf_all_impl(int n, OutputMat res) {
        /* Shared implementation used by PMF and CDF table generation. */
        using T = typename OutputMat::value_type;
        const int max_k = n * (n + 1) / 2;

        for (int i = 0; i <= max_k; ++i) {
            res(i) = T(0.0);
        }
        res(0) = T(1.0);

        for (int k = 1; k <= n; ++k) {
            const int curr_max = k * (k + 1) / 2;
            // Iterate backwards so res(i - k) still holds its previous-step value.
            for (int i = curr_max; i >= 0; --i) {
                res(i) = T(0.5) * (res(i) + (i >= k ? res(i - k) : T(0.0)));
            }
        }
    }

} // namespace detail

template <typename OutputMat>
XSF_HOST_DEVICE inline void wilcoxon_pmf_all(int n, OutputMat res) {
    /* Fill `res` with the full PMF table for a Wilcoxon signed-rank statistic.
     *
     * res should be an mdspan view of a 1d array of length n * (n + 1) / 2 + 1.
     * It is up to the caller to pass valid values for n and res.
     */
    if (n < 0) {
        set_error("_wilcoxon_pmf_all", SF_ERROR_DOMAIN, "n must be non-negative");
        return;
    }

    const long long int out_size = static_cast<long long int>(res.extent(0));
    const long long int expected_size = static_cast<long long int>(n) * (n + 1) / 2 + 1;
    if (out_size != expected_size) {
        set_error("_wilcoxon_pmf_all", SF_ERROR_MEMORY, "out.shape[-1] must be n * (n + 1) / 2 + 1");
        return;
    }

    detail::wilcoxon_pmf_all_impl(n, res);
}

template <typename OutputMat>
XSF_HOST_DEVICE inline void wilcoxon_cdf_all(int n, OutputMat res) {
    /* Fill `res` with the full CDF table for a Wilcoxon signed-rank statistic. */
    using T = typename OutputMat::value_type;

    if (n < 0) {
        set_error("_wilcoxon_cdf_all", SF_ERROR_DOMAIN, "n must be non-negative");
        return;
    }

    const long long int out_size = static_cast<long long int>(res.extent(0));
    const long long int expected_size = static_cast<long long int>(n) * (n + 1) / 2 + 1;
    if (out_size != expected_size) {
        set_error("_wilcoxon_cdf_all", SF_ERROR_MEMORY, "out.shape[-1] must be n * (n + 1) / 2 + 1");
        return;
    }

    detail::wilcoxon_pmf_all_impl(n, res);
    for (long long int i = 1; i < out_size - 1; ++i) {
        res(i) = cxx::min(res(i) + res(i - 1), T(1.0));
    }
    res(out_size - 1) = T(1.0);
}

template <typename InputMat>
XSF_HOST_DEVICE inline typename InputMat::value_type wilcoxon_pmf(long long int k, InputMat pmf) {
    /* Return a scalar value from a precomputed PMF table. */
    return take_from_pmf(pmf, k);
}

template <typename InputMat>
XSF_HOST_DEVICE inline typename InputMat::value_type wilcoxon_pmf(double k, InputMat pmf) {
    using T = typename InputMat::value_type;
    if (cxx::isnan(k)) {
        return cxx::numeric_limits<T>::quiet_NaN();
    }
    return wilcoxon_pmf(static_cast<long long int>(k), pmf);
}

template <typename InputMat>
XSF_HOST_DEVICE inline typename InputMat::value_type wilcoxon_cdf(long long int k, InputMat cdf) {
    /* Return a scalar value from a precomputed CDF table. */
    return take_from_discrete_cdf(cdf, k);
}

template <typename InputMat>
XSF_HOST_DEVICE inline typename InputMat::value_type wilcoxon_cdf(double k, InputMat cdf) {
    using T = typename InputMat::value_type;
    if (cxx::isnan(k)) {
        return cxx::numeric_limits<T>::quiet_NaN();
    }
    return wilcoxon_cdf(static_cast<long long int>(k), cdf);
}

template <typename InputMat>
XSF_HOST_DEVICE inline typename InputMat::value_type wilcoxon_sf(long long int k, InputMat cdf) {
    /* Return a scalar survival probability from a precomputed CDF table. */
    using T = typename InputMat::value_type;
    return T(1.0) - take_from_discrete_cdf(cdf, k - 1);
}

template <typename InputMat>
XSF_HOST_DEVICE inline typename InputMat::value_type wilcoxon_sf(double k, InputMat cdf) {
    using T = typename InputMat::value_type;
    if (cxx::isnan(k)) {
        return cxx::numeric_limits<T>::quiet_NaN();
    }
    return wilcoxon_sf(static_cast<long long int>(k), cdf);
}

namespace detail {

    XSF_HOST_DEVICE inline double von_mises_cdf_series(double k, double x, unsigned int p) {
        double s, c, sn, cn, r, v;
        unsigned int n;
        s = cxx::sin(x);
        c = cxx::cos(x);
        sn = cxx::sin(p * x);
        cn = cxx::cos(p * x);
        r = 0;
        v = 0;
        for (n = p - 1; n > 0; --n) {
            double sn_new = sn * c - cn * s;
            double cn_new = cn * c + sn * s;
            sn = sn_new;
            cn = cn_new;
            r = k / (2 * n + k * r);
            v = r * (sn / n + v);
        }
        return 0.5 + x / (2.0 * M_PI) + v / M_PI;
    }

    XSF_HOST_DEVICE inline double von_mises_cdf_normalapprox(double k, double x) {
        double b = xsf::cephes::detail::SQRT2OPI / cephes::i0e(k);
        double z = b * cxx::sin(x / 2.0);
        return ndtr(z);
    }

} // namespace detail

XSF_HOST_DEVICE inline double von_mises_cdf(double k, double x) {
    // CDF of the von Mises distribution with concentration k, extended
    // periodically over x.
    //
    // For k < 50, this uses the backward-recursion series method of Hill [1],
    // Algorithm 518, with constants chosen for about 12 decimal digits. For k >= 50,
    // this switches to a large-concentration normal approximation.
    //
    // References:
    // [1] G. W. Hill, "Algorithm 518: Incomplete Bessel Function I0.
    //     The Von Mises Distribution [S14]", ACM Transactions on Mathematical
    //     Software, 3(3), 279-284, 1977.
    //     DOI: https://doi.org/10.1145/355744.355753

    if (k < 0) {
        set_error("von_mises_cdf", SF_ERROR_DOMAIN, NULL);
        return cxx::numeric_limits<double>::quiet_NaN();
    }

    double ix = cxx::round(x / (2 * M_PI));
    x -= ix * 2.0 * M_PI;

    // These values should give 12 decimal digits
    const double ck = 50.0;
    const double a1 = 28.0, a2 = 0.5, a3 = 100.0, a4 = 5.0;
    double result;
    if (k < ck) {
        unsigned int p = static_cast<unsigned int>(1 + a1 + a2 * k - a3 / (k + a4));
        result = detail::von_mises_cdf_series(k, x, p);
        result = cxx::min(cxx::max(result, 0.0), 1.0);
    } else {
        result = detail::von_mises_cdf_normalapprox(k, x);
    }
    return result + ix;
}

XSF_HOST_DEVICE inline float von_mises_cdf(float k, float x) {
    return static_cast<float>(von_mises_cdf(static_cast<double>(k), static_cast<double>(x)));
}

} // namespace xsf
