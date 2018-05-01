from utils import (
    get_D8_group_transformed_tensor,
    get_concat_kernel_convols,
    get_max_pooled_tensor,
    get_dense_layer,
    get_custom_group_transformed_tensor
)


class RotConv2D(object):
    """ Main class that does all the core work.
        Task 1 and Task 3 are covered by this class RotConv2D, using the member func 'get_dense_output'
        (see main.py and wiki)
    """

    def __init__(self, x_input=None, kernel=None, rotation_type='dihedral', custom_angles=[]):
        self.rotation_type = rotation_type
        self.custom_angles = custom_angles

        if self.rotation_type not in ('dihedral', 'custom'):
            raise Warning('Rotation Type not valid! Assume Dihedral')
            self.rotation_type = 'dihedral'

        if x_input is None or kernel is None:
            raise ValueError('Must supply X and kernel inputs!')

        self.X = x_input
        self.K = kernel
        self.all_kernels = None

    def initialize_all_filters(self):
        """
            Builds all filters that will be used for convolution step
        :return:
        """
        if self.rotation_type == 'dihedral':
            self.all_kernels = get_D8_group_transformed_tensor(self.K)
        else:
            self.all_kernels = get_custom_group_transformed_tensor(self.K, self.custom_angles)

    def get_dense_output(self):
        """
        Task 1 function
        Returns: tensor, dense output of shape (Nsamples, Height, Width, 1)

        """

        if self.all_kernels is None:
            self.initialize_all_filters()

        stacked_convols = get_concat_kernel_convols(self.X, self.all_kernels, padding='SAME')

        pooled_convols = get_max_pooled_tensor(stacked_convols)

        dense = get_dense_layer(pooled_convols=pooled_convols)

        return dense
