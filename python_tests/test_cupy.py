"""Test xsf functions work on GPU through CuPy.

Beyond cupy and pytest, also requires xsref package to be installed.
Should be run with pytest-isolate or pytest-forked to isolate tests
in separate processes since memory corruption on GPU can cause
failures to occur in unrelated tests.
"""

import os
import numpy as np
import polars as pl
import pytest

from packaging.version import Version
from pathlib import Path

from xsref.float_tools import extended_relative_error
from xsref.tables import get_input_rows, get_output_rows, get_tables_path


#------------------------------------------------------------------------------
# Check if a module is present to be used in tests
#
# Copied from
# https://github.com/scipy/scipy/blob/1cbfa1c894557041d9825d6754a2c48fc3bec484/scipy/special/_testutils.py
#------------------------------------------------------------------------------

class MissingModule:
    def __init__(self, name):
        self.name = name


def check_version(module, min_ver):
    if type(module) is MissingModule:
        return pytest.mark.skip(reason=f"{module.name} is not installed")
    return pytest.mark.skipif(
        Version(module.__version__) < Version(min_ver),
        reason=f"{module.__name__} version >= {min_ver} required"
    )

#------------------------------------------------------------------------------

try:
    import cupy  # type: ignore
except (ImportError, AttributeError):
    cupy = MissingModule('cupy')


def _get_cols_helper(table_path, xp):
    # This is more complicated than need be due to an oversight in xsref
    # functions for getting tables. The table metadata for input, output,
    # and err tables has the input and output types, but does not have
    # the column types of the current table, and does not contain the info
    # of whether current table is an input, output, or err table. It also
    # provides separate functions to get input rows from an input table,
    # output rows from an output table, and provides no function to get
    # rows from an err table (though this last task is easy to do because
    # there are no complex types involved in an err table).
    # TODO: Update xsref tables to contain the table's column types
    # in the metadata, and provide a single function for getting rows
    # from tables as lists of tuples that uses this metadata.
    table_name = table_path.name.lower()
    if table_name.startswith("in_"):
        rows = get_input_rows(table_path)
    elif table_name.startswith("out_"):
        rows = get_output_rows(table_path)
    elif table_name.startswith("err_"):
        table = pl.read_parquet(table_path)
        rows = table.to_numpy()
    result = tuple(xp.asarray(col) for col in zip(*rows))
    if len(result) == 1:
        result = result[0]
    return result


def get_cols_as_cupy(table_path):
    return _get_cols_helper(table_path, cupy)


def get_cols_as_numpy(table_path):
    return _get_cols_helper(table_path, np)


XSF_INCLUDE_DIR = Path(os.environ["CONDA_PREFIX"]) / "include"


def get_tables_for_func(func_name):
    tables_path = get_tables_path() / "scipy_special_tests" / func_name
    input_tables = list(tables_path.glob("In_*.parquet"))
    output_tables = [
        path.parent / path.name.replace("In_", "Out_") for path in input_tables
    ]
    err_tables = []
    for path in input_tables:
        types = path.name.removesuffix(".parquet").replace("In_", "")
        name = f"Err_{types}_other.parquet"
        err_tables.append(path.parent / name)
    return list(zip(input_tables, output_tables, err_tables))


def get_preamble(header):
    header_path = (XSF_INCLUDE_DIR / Path(header)).resolve()
    return f'#include "{header_path}"'


@check_version(cupy, "13.0.0")
class TestCuPy:
    def _adjust_tol(self, tol, *, wiggle=16):
        return wiggle * np.maximum(tol, np.finfo(tol.dtype).eps)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("order", ["C", "F"])
    @pytest.mark.parametrize("case", [
        "dyadic", "heterogeneous", "tiny", "underflow", "round-up",
        "round-down", "tie", "tie-extra", "shifted", "empty", "below-range",
        "deterministic", "higher-round-up", "higher-round-down",
        "higher-tie", "higher-tie-extra", "higher-shifted", "higher-many",
    ])
    def test_poisson_binom_sf(self, dtype, order, case):
        from fractions import Fraction

        info = np.finfo(dtype)
        subnormal = dtype(info.smallest_subnormal)
        is_float = dtype == np.float32
        center = dtype(float.fromhex(
            "0x1.a20bd8p-77" if is_float else "0x1.a20bd700c2c3ep-539"
        ))
        n = 4 if is_float else 3
        k = 1
        if case == "dyadic":
            p = np.array([1/8, 1/4, 1/2, 3/4], dtype=dtype)
        elif case == "heterogeneous":
            p = np.array([(1 + i % 9) / 128 for i in range(32)], dtype=dtype)
            k = 23
        elif case == "tiny":
            p = np.full(16, 1e-20, dtype=dtype)
            k = 0
        elif case == "underflow":
            p = np.full(3, 1.7e-23 if is_float else 1e-162, dtype=dtype)
        elif case in {"round-up", "round-down", "shifted"}:
            probability = np.nextafter(center, dtype(0)) if case == "round-down" else center
            p = np.full(n, probability, dtype=dtype)
            if case == "shifted":
                p = np.concatenate(([dtype(1)], p, [dtype(1), dtype(0)]))
                k = 3
        elif case in {"tie", "tie-extra"}:
            exponent = info.minexp - info.nmant - 1
            first_exponent = int(exponent / 2)
            p = np.array([
                np.ldexp(dtype(1), first_exponent),
                np.ldexp(dtype(1), exponent - first_exponent),
            ], dtype=dtype)
            if case == "tie-extra":
                p = np.append(p, subnormal)
        elif case in {"higher-round-up", "higher-round-down"} and is_float:
            probability = dtype(float.fromhex("0x1.428a30p-51"))
            if case == "higher-round-down":
                probability = np.nextafter(probability, dtype(0))
            p = np.full(4, probability, dtype=dtype)
            k = 2
        elif case.startswith("higher-"):
            successes = 25 if case == "higher-many" else 3
            exponent = info.minexp - info.nmant - 1
            first_exponent = int(exponent / successes)
            p = np.full(successes, np.ldexp(dtype(1), first_exponent), dtype=dtype)
            p[-1] = np.ldexp(dtype(1), exponent - (successes - 1) * first_exponent)
            k = successes - 1
            if case == "higher-round-down":
                p[0] = np.nextafter(p[0], dtype(0))
            if case in {"higher-round-up", "higher-tie-extra", "higher-shifted", "higher-many"}:
                p = np.append(p, subnormal)
            if case == "higher-shifted":
                p = np.concatenate(([dtype(1)], p, [dtype(1), dtype(0)]))
                k += 2
        elif case == "empty":
            p = np.empty(0, dtype=dtype)
            k = 0
        elif case == "below-range":
            p = np.full(256, 1/128, dtype=dtype)
            k = 249
        else:
            p = np.array([0, 1, 0, 1], dtype=dtype)

        # Exact references for the actual binary inputs, including subnormal
        # rounding without an intermediate conversion through double precision.
        pmf = [Fraction(1)]
        for probability in map(lambda x: Fraction(float(x)), p):
            updated = [Fraction(0)] * (len(pmf) + 1)
            for j, mass in enumerate(pmf):
                updated[j] += mass * (1 - probability)
                updated[j + 1] += mass * probability
            pmf = updated
        references = []
        tail = Fraction(0)
        for mass in reversed(pmf):
            if tail < Fraction(float(info.tiny)):
                value = dtype(round(tail / Fraction(float(subnormal)))) * subnormal
            else:
                value = dtype(float(tail))
            references.append(value)
            tail += mass
        expected = np.array(references[::-1], dtype=dtype)

        scalar = "float" if is_float else "double"
        code = "#include <cupy/complex.cuh>\n" + get_preamble("xsf/stats.h") + r"""
        template <typename T>
        struct PoissonBinomialView {
            using value_type = T;
            T *data;
            long long size, stride;
            __device__ long long extent(int) const { return size; }
            __device__ T &operator()(long long i) const { return data[i * stride]; }
        };
        extern "C" __global__ void poisson_binom_sf(
            const SCALAR *p, SCALAR *out, SCALAR *selected, double *workspace,
            long long n, long long rows, long long p_row, long long p_core,
            long long out_row, long long out_core, long long work_row, long long work_core, long long k
        ) {
            long long row = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= rows) { return; }
            PoissonBinomialView<const SCALAR> input{p + row * p_row, n, p_core};
            PoissonBinomialView<SCALAR> output{out + row * out_row, n + 1, out_core};
            PoissonBinomialView<double> scratch{workspace + row * work_row, 3 * (n + 1), work_core};
            xsf::poisson_binom_sf_all(input, output, scratch);
            selected[3 * row] = xsf::take_from_discrete_sf(output, -1LL);
            selected[3 * row + 1] = xsf::take_from_discrete_sf(output, k);
            selected[3 * row + 2] = xsf::take_from_discrete_sf(output, n);
        }
        """
        kernel = cupy.RawKernel(code.replace("SCALAR", scalar), "poisson_binom_sf", options=("--std=c++17",))
        inputs = cupy.array(np.stack([p, p[::-1]]), dtype=dtype, order=order)
        output = cupy.empty((2, p.size + 1), dtype=dtype, order=order)
        selected = cupy.empty((2, 3), dtype=dtype)
        workspace = cupy.empty((2, 3 * (p.size + 1)), dtype=np.float64, order=order)
        itemsize = np.dtype(dtype).itemsize
        kernel((1,), (32,), (
            inputs, output, selected, workspace, np.int64(p.size), np.int64(2),
            np.int64(inputs.strides[0] // itemsize), np.int64(inputs.strides[1] // itemsize),
            np.int64(output.strides[0] // itemsize), np.int64(output.strides[1] // itemsize),
            np.int64(workspace.strides[0] // workspace.itemsize), np.int64(workspace.strides[1] // workspace.itemsize), np.int64(k),
        ))
        result = cupy.asnumpy(output)
        choices = cupy.asnumpy(selected)
        desired = np.broadcast_to(expected, result.shape)
        error = np.abs(result.astype(np.float64) - desired.astype(np.float64))
        rtol = 32 * max(p.size, 1) * info.eps
        assert np.all(error <= rtol * np.abs(desired.astype(np.float64)) + float(subnormal))
        assert np.all(np.isfinite(result))
        assert np.all((result >= 0) & (result <= 1))
        assert np.all(result[:, 1:] <= result[:, :-1])
        np.testing.assert_array_equal(result[:, -1], np.zeros(2, dtype=dtype))
        np.testing.assert_allclose(choices[:, 1], expected[k], rtol=rtol, atol=float(subnormal))
        np.testing.assert_array_equal(choices[:, 0], np.ones(2, dtype=dtype))
        np.testing.assert_array_equal(choices[:, 2], np.zeros(2, dtype=dtype))
        if case.startswith("higher-") or case in {
            "underflow", "round-up", "round-down", "tie", "tie-extra",
            "shifted", "below-range",
        }:
            np.testing.assert_array_equal(choices[:, 1], np.full(2, expected[k], dtype=dtype))

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("beta")
    )
    def test_beta(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        beta = cupy._core.create_ufunc(
            "cupyx_scipy_beta",
            ("ll->d", "LL->d", "ee->d", "ff->f", "dd->d"),
            "out0 = out0_type(xsf::beta(in0, in1));",
            preamble=get_preamble("xsf/beta.h"),
        )

        a, b = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(beta(a, b))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("binom")
    )
    def test_binom(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        binom = cupy._core.create_ufunc(
            "cupyx_scipy_binom",
            ("ff->f", "dd->d"),
            "out0 = out0_type(xsf::binom(in0, in1));",
            preamble=get_preamble("xsf/binom.h"),
        )

        n, k = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(binom(n, k))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("gamma")
    )
    def test_gamma(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        gamma = cupy._core.create_ufunc(
            'cupyx_scipy_special_gamma',
            ('f->f', 'd->d', 'F->F', 'D->D'),
            'out0 = xsf::gamma(in0)',
            preamble=get_preamble("xsf/gamma.h"),
        )

        x = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(gamma(x))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("digamma")
    )
    def test_digamma(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        digamma = cupy._core.create_ufunc(
            'cupyx_scipy_special_digamma',
            (
                ('l->d', 'out0 = xsf::digamma(double(in0))'),
                ('e->d', 'out0 = xsf::digamma(double(in0))'),
                'f->f',
                'd->d',
                'F->F',
                'D->D',
            ),
            'out0 = xsf::digamma(in0)',
            preamble=get_preamble("xsf/digamma.h")
        )

        x = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(digamma(x))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )


    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("ellipkinc")
    )
    def test_ellipkinc(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        ellipkinc = cupy._core.create_ufunc(
            'cupyx_scipy_special_ellipkinc', ('ff->f', 'dd->d'),
            'out0 = xsf::cephes::ellik(in0, in1)',
            preamble=get_preamble("xsf/cephes/ellik.h"),
        )

        phi, m = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(ellipkinc(phi, m))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("ellipeinc")
    )
    def test_ellipeinc(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        ellipeinc = cupy._core.create_ufunc(
            'cupyx_scipy_special_ellipeinc', ('ff->f', 'dd->d'),
            'out0 = xsf::cephes::ellie(in0, in1)',
            preamble=get_preamble("xsf/cephes/ellie.h"),
        )

        phi, m = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(ellipeinc(phi, m))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("expn")
    )
    def test_expn(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        expn = cupy._core.create_ufunc(
            'cupyx_scipy_special_expn',
            ('ff->f', 'dd->d'),
            'out0 = xsf::cephes::expn(in0, in1)',
            preamble=get_preamble("xsf/cephes/expn.h"),
        )

        n, x = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(expn(n, x))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("gdtrib")
    )
    @pytest.mark.xfail(reason="Requires cpp_std >= 17")
    def test_gdtrib(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        gdtrib = cupy._core.create_ufunc(
            'cupyx_scipy_special_gdtrib',
            ('fff->f', 'ddd->d'),
            'out0 = xsf::gdtrib(in0, in1, in2)',
            preamble=get_preamble("xsf/cdflib.h"),
        )

        a, p, x = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(gdtrib(a, p, x))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("hyp2f1")
    )
    def test_hyp2f1(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        hyp2f1 = cupy._core.create_ufunc(
            'cupyx_scipy_special_hyp2f1',
            ('ffff->f', 'dddd->d', 'fffF->F', 'dddD->D'),
            'out0 = xsf::hyp2f1(in0, in1, in2, in3)',
            preamble=get_preamble("xsf/hyp2f1.h"),
        )

        a, b, c, z = get_cols_as_cupy(input_path)
        if not cupy.iscomplexobj(z):
            pytest.xfail(
                "Real valued hyp2f1 currently broken on GPU due to use of"
                " recursion."
            )
        out = cupy.asnumpy(hyp2f1(a, b, c, z))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("lambertw")
    )
    def test_lambertw(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        _lambertw_scalar = cupy._core.create_ufunc(
            "cupyx_scipy_lambertw_scalar",
            ("Dld->D", "Flf->F"),
            "out0 = xsf::lambertw(in0, in1, in2)",
            preamble=get_preamble("xsf/lambertw.h"),
        )

        # A parameter called tol, not to be confused with the rtol for assessing
        # accuracy.
        z, k, tol = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(_lambertw_scalar(z, k, tol))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("sici")
    )
    def test_sici(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        sici = cupy._core.create_ufunc(
            'cupyx_scipy_special_sici',
            (
                (
                    'f->ff',
                    '''
                    float si, ci;
                    xsf::sici(in0, si, ci);
                    out0 = si; out1 = ci;
                    ''',
                ),
            (
                'd->dd',
                '''
                double si, ci;
                xsf::sici(in0, si, ci);
                out0 = si; out1 = ci;
                ''',
            ),
                (
                    'F->FF',
                    '''
                    complex<float> si, ci;
                    xsf::sici(in0, si, ci);
                out0 = si; out1 = ci;
                    ''',
                ),
                (
                    'D->DD',
                    '''
                    complex<double> si, ci;
                    xsf::sici(in0, si, ci);
                    out0 = si; out1 = ci;
                    ''',
                ),
            ),
            preamble=get_preamble("xsf/sici.h"),
        )

        x = get_cols_as_cupy(input_path)
        if cupy.iscomplexobj(x):
            pytest.xfail("Known bug, returning nan instead of a complex infinity.")
        out0, out1 = map(cupy.asnumpy, sici(x))

        desired0, desired1 = get_cols_as_numpy(output_path)
        rtol0, rtol1 = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out0, desired0) <= self._adjust_tol(rtol0)
        )
        assert np.all(
            extended_relative_error(out1, desired1) <= self._adjust_tol(rtol1)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("shichi")
    )
    def test_shichi(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        shichi = cupy._core.create_ufunc(
            'cupyx_scipy_special_shichi',
            (
                (
                    'f->ff',
                    '''
                    float shi, chi;
                    xsf::shichi(in0, shi, chi);
                    out0 = shi; out1 = chi;
                    ''',
                ),
                (
                    'd->dd',
                    '''
                    double shi, chi;
                    xsf::shichi(in0, shi, chi);
                    out0 = shi; out1 = chi;
                    ''',
                ),
                (
                    'F->FF',
                    '''
                    complex<float> shi, chi;
                    xsf::shichi(in0, shi, chi);
                    out0 = shi; out1 = chi;
                    ''',
                ),
                (
                    'D->DD',
                    '''
                    complex<double> shi, chi;
                    xsf::shichi(in0, shi, chi);
                    out0 = shi; out1 = chi;
                    ''',
                ),
            ),
            preamble=get_preamble("xsf/sici.h"),
        )

        x = get_cols_as_cupy(input_path)
        if cupy.iscomplexobj(x):
            pytest.xfail("Known bug, returning nan instead of a complex infinity.")
        out0, out1 = map(cupy.asnumpy, shichi(x))

        desired0, desired1 = get_cols_as_numpy(output_path)
        rtol0, rtol1 = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out0, desired0) <= self._adjust_tol(rtol0)
        )
        assert np.all(
            extended_relative_error(out1, desired1) <= self._adjust_tol(rtol1)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("erf")
    )
    def test_erf(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        erf = cupy._core.create_ufunc(
            'cupyx_scipy_special_erf',
            ('f->f', 'd->d', 'F->F', 'D->D'),
            'out0 = xsf::erf(in0)',
            preamble=get_preamble("xsf/erf.h"),
        )

        x = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(erf(x))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("erfc")
    )
    def test_erfc(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        erfc = cupy._core.create_ufunc(
            'cupyx_scipy_special_erfc',
            ('f->f', 'd->d', 'F->F', 'D->D'),
            'out0 = xsf::erfc(in0)',
            preamble=get_preamble("xsf/erf.h"),
        )

        x = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(erfc(x))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)

        # CuPy returns nan + nanj for this case, but test expects
        # -inf + infj.
        skip = x.get() == (-21092.63667768141-7.794025022440267e+201j)
        out, desired, rtol = out[~skip], desired[~skip], rtol[~skip]

        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("erfcx")
    )
    def test_erfcx(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        erfcx = cupy._core.create_ufunc(
            'cupyx_scipy_special_erfcx',
            ('f->f', 'd->d', 'F->F', 'D->D'),
            'out0 = xsf::erfcx(in0)',
            preamble=get_preamble("xsf/erf.h"),
        )

        x = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(erfcx(x))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("erfi")
    )
    def test_erfi(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        erfi = cupy._core.create_ufunc(
            'cupyx_scipy_special_erfi',
            ('f->f', 'd->d', 'F->F', 'D->D'),
            'out0 = xsf::erfi(in0)',
            preamble=get_preamble("xsf/erf.h"),
        )

        x = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(erfi(x))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("voigt_profile")
    )
    def test_voigt_profile(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        voigt_profile = cupy._core.create_ufunc(
            'cupyx_scipy_special_voigt_profile',
            ('fff->f', 'ddd->d'),
            'out0 = xsf::voigt_profile(in0, in1, in2)',
            preamble=get_preamble("xsf/erf.h"),
        )

        x, sigma, gamma = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(voigt_profile(x, sigma, gamma))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("wofz")
    )
    def test_wofz(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        wofz = cupy._core.create_ufunc(
            'cupyx_scipy_special_wofz',
            ('F->F', 'D->D'),
            'out0 = xsf::wofz(in0)',
            preamble=get_preamble("xsf/erf.h"),
        )

        x = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(wofz(x))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )

    @pytest.mark.parametrize(
        "tables_paths", get_tables_for_func("dawsn")
    )
    def test_dawsn(self, tables_paths):
        input_path, output_path, tol_path = tables_paths
        dawsn = cupy._core.create_ufunc(
            'cupyx_scipy_special_dawsn',
            ('f->f', 'd->d', 'F->F', 'D->D'),
            'out0 = xsf::dawsn(in0)',
            preamble=get_preamble("xsf/erf.h"),
        )

        x = get_cols_as_cupy(input_path)
        out = cupy.asnumpy(dawsn(x))

        desired = get_cols_as_numpy(output_path)
        rtol = get_cols_as_numpy(tol_path)
        assert np.all(
            extended_relative_error(out, desired) <= self._adjust_tol(rtol)
        )
