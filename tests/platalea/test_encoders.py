import torch
import torch.nn as nn

from platalea.encoders import inout


def test_simplest_valid_case():
    _assert_inout_for_expected_output(input_length=10, expected=10)


def test_kernel_larger_than_input():
    _assert_inout_for_expected_output(input_length=3, expected=0, kernel_size=5)


def test_simple_padding():
    """With a kernel of 1, the result should be 1* input + 2* padding."""
    _assert_inout_for_expected_output(input_length=3, expected=5, padding=1)


def test_padding_with_larger_kernel():
    """With a kernel of 3, the result should be 1* input + 2* padding, minus the (kernel -1)."""
    _assert_inout_for_expected_output(input_length=3, expected=3, padding=1, kernel_size=3)


def test_simple_stride():
    """With a kernel of 1, the result should be (input + 1) / stride."""
    _assert_inout_for_expected_output(input_length=5, expected=3, stride=2)


def test_stride_non_exact_solution():
    """With a kernel of 1, the result should be floor[(input + 1) / stride ]."""
    _assert_inout_for_expected_output(input_length=8, expected=2, stride=5)


def test_trivial_dilation():
    """With a kernel of 1, dilation should have no effect, so result should be the input size."""
    _assert_inout_for_expected_output(input_length=5, expected=5, dilation=2)


def test_simple_dilation():
    """Dilation, in a way, increases the virtual kernel. A kernel size of 3 and dilation 2 results in a virtual
    kernel of 5. The result from inout should be ."""
    _assert_inout_for_expected_output(input_length=7, expected=3, dilation=2, kernel_size=3)


def test_dilation_with_padding():
    """A kernel size of 3 and dilation 3 results in a virtual kernel of 7.
    The result should be 1* input + 2* padding, minus the (virtual_kernel -1)."""
    _assert_inout_for_expected_output(input_length=6, expected=6, dilation=3, kernel_size=3, padding=3)


def test_dilation_with_padding_and_stride():
    """A kernel size of 3 and dilation 3 results in a virtual kernel of 7.
    The result should be 1* input + 2* padding, minus the (virtual_kernel -1), then compensated for stride by
    taking the floor of the division through stride."""
    _assert_inout_for_expected_output(input_length=6, expected=2, dilation=3, kernel_size=3, padding=3, stride=3)


def _assert_inout_for_expected_output(input_length, expected, kernel_size=1, padding=0, stride=1, dilation=1):
    expected = torch.tensor(expected)
    input_length = torch.tensor(input_length)

    layer = nn.Conv1d(
        in_channels=5, out_channels=3, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
    result = inout(layer, input_length)

    torch.testing.assert_allclose(result, expected)
