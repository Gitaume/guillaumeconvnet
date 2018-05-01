import pytest
from guillaumeml.lib.utils import (
    timer_function,
    get_D8_group_transformed_tensor,
    get_concat_kernel_convols
)
from tensorflow import reshape, reduce_sum, squeeze, constant, Session
from numpy import max


sess = Session()


# some fixtures for the tests
X = reshape(constant([[[0., 1, 0],[0., 1, 0],[0., 1, 0.]]]), shape=[1, 3, 3, 1])  # last digit is # of channels
K = reshape(constant([[[0., 0, 0],[1., 1, 1], [0., 0, 0]]]), shape=[3, 3, 1, 1])  # last digit is # of output channels.. 2nd to last is # of input channels


@timer_function
def test_output_size_is_8():
    """ Valid for the fixture above
        Test output vector size is 8"""
    all_D8_kernels = get_D8_group_transformed_tensor(K)
    stacked_convols = get_concat_kernel_convols(X, all_D8_kernels, padding='VALID')
    # warning! if input channel > 1, reduce_sum must be invoked to "eliminate" the extra dimension
    # res = reduce_sum(squeeze(stacked_convols), 0)
    res = squeeze(stacked_convols)
    assert res.get_shape().as_list() == [8]


@timer_function
def test_maxima_positions():
    """ Valid for the fixture above"""
    all_D8_kernels = get_D8_group_transformed_tensor(K)
    stacked_convols = get_concat_kernel_convols(X, all_D8_kernels, padding='VALID')
    valid_convs_dotp = stacked_convols.eval(session=sess)
    valid_convs_dotp = valid_convs_dotp.squeeze().tolist()
    the_max = max(valid_convs_dotp)
    position_maxima = [i for i, e in enumerate(valid_convs_dotp) if e == the_max]
    # maxes should be when the '1' are vertically aligned in both X and the kernel, which happens for:
    #  +90 rotation (second element of the filter list, i.e. index 1), +270 rotation (4th element), and
    # their horizontal reflections (6th and 8th elements)
    assert position_maxima == [1, 3, 5, 7]


if __name__ == "__main__":
    pytest.main()
