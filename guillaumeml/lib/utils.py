import tensorflow as tf
import time


def validate_tensor_type(func):
    """
    Validate type is Tensor for funcs
    """

    def wrapper(*args, **kwargs):
        try:
            assert isinstance(args[0], tf.Tensor)
        except AssertionError:
            print '\n', '************ /!\ Wrong, non-tensor Type /!\ ***********', args, '\n'
        return func(*args, **kwargs)

    return wrapper


@validate_tensor_type
def rotate_positive_90(a_tensor):
    """

    Args:
        a_tensor: (tf tensor) input tensor. First 2 dimensions must be Height, Width (HW)

    Returns: rotated tensor +90 degrees on first 2 dimensions

    """
    perms = range(len(a_tensor.shape))
    perms[1] = 0
    perms[0] = 1
    return tf.reverse(tf.transpose(a_tensor, perm=perms), [1])


def reflect_horizontally(a_tensor):
    """

    Args:
        a_tensor: (tf tensor) input tensor. First 2 dimensions must be Height, Width (HW)

    Returns: tensor after horizontal reflection on first 2 dimensions
    """

    return tf.reverse(a_tensor, [1])


def rotate_tensor_by_theta(a_tensor, theta=0., interpolation="BILINEAR"):
    """

    Args:
        a_tensor: must have Height and Width as second and third dimension. Example: shape=(N, H, W, C)
        theta: rotation angle [degrees]
        interpolation: because matrices are finite spaces. Bilinear by default b/c kernels are small in this project

    Returns:
        Rotated tensor

    """
    if not isinstance(a_tensor, tf.Tensor):
        raise TypeError('Must input a tensor!')

    if theta >= 360. or theta < 0.:
        theta %= 360  # restrict range to 0-2pi

    shape = a_tensor.get_shape().as_list()
    if shape[0] == shape[1] and shape[1] != shape[2]:
        a_tensor = tf.transpose(a_tensor, [2, 1, 0, 3])
        # print 'WARNING ! Input tensor was not of expected shape, we assume input tensor was WHNC or WHCC'

    rotated_tensor = tf.contrib.image.rotate(a_tensor, angles=3.141526 * theta / 180., interpolation=interpolation)

    return tf.transpose(rotated_tensor, [2, 1, 0, 3])


def composer(f, n):
    def composed_func(an_input):
        return reduce(lambda x, _: f(x), xrange(n), an_input)

    return composed_func


def get_D8_group_transformations():
    """

    Args:
        a_tensor: (tf tensor) input tensor. First 2 dimensions must be Height, Width (HW)

    Returns: list of transformations for each element of the D8 group (Dihedral group)

    """
    rotate_twice = composer(rotate_positive_90, 2)
    rotate_thrice = composer(rotate_positive_90, 3)
    list_transformations = []
    list_transformations.append(lambda x: x)  # identity element of the D8 group
    list_transformations.append(rotate_positive_90)
    list_transformations.append(rotate_twice)
    list_transformations.append(rotate_thrice)
    list_transformations.append(reflect_horizontally)
    list_transformations.append(lambda x: rotate_positive_90(reflect_horizontally(x)))
    list_transformations.append(lambda x: rotate_twice(reflect_horizontally(x)))
    list_transformations.append(lambda x: rotate_thrice(reflect_horizontally(x)))

    return list_transformations


def get_concat_kernel_convols(x, filters, padding='SAME'):
    list_convols = []
    for a_filter in filters:
        list_convols.append(tf.nn.conv2d(input=x, filter=a_filter, strides=[1, 1, 1, 1], padding=padding))

    return tf.stack(list_convols, 4)


def get_max_pooled_tensor(stacked_convols):
    """

    Args:
        stacked_convols: Orientation dimension must be '4' (last dim)

    Returns:

    """
    n_filters = stacked_convols.get_shape().as_list()[-1]  # Number of elements in the group, 8 for Task 1
    tmp_transpose = tf.transpose(stacked_convols, [0, 4, 1, 2, 3])
    pooled_cat_conv = tf.nn.max_pool3d(tmp_transpose, ksize=[1, n_filters, 1, 1, 1], strides=[1, 1, 1, 1, 1],
                                       padding='VALID', data_format='NDHWC')
    pooled_cat_conv = tf.transpose(pooled_cat_conv, [0, 2, 3, 4, 1])

    # remove unecessary last dim (was used to execute max pooling)
    pooled_cat_conv = tf.squeeze(pooled_cat_conv, [4])

    return pooled_cat_conv


def get_D8_group_transformed_tensor(a_tensor):
    """

    Args:
        a_tensor: (tf tensor) input tensor. First 2 dimensions must be Height, Width (HW)

    Returns: list of transformed tensors for each element of the D8 group (Dihedral group)

    """
    return [i(a_tensor) for i in get_D8_group_transformations()]


def get_custom_group_transformed_tensor(a_tensor, angles=[]):
    return [rotate_tensor_by_theta(a_tensor, an_angle) for an_angle in angles]


def get_dense_layer(pooled_convols, units=1, activation=tf.nn.leaky_relu):
    # yes, activation after poolings is valid b/c it is a commutative operation b/c order is preserved, and even more
    # efficient when pooling reduces dimensions.
    # 1 out units does the job of "aggregating" channels. Alternatively, one may sum them up before activation.
    return tf.layers.dense(inputs=pooled_convols, units=units, activation=activation)


def timer_function(func):
    """
    Outputs the execution total time of a function
    """

    def wrapper():
        t1 = time.time()
        func()
        t2 = time.time()
        tot_t = float((t2 - t1))
        if tot_t > 10.:
            raise (
                AssertionError('function {0} call takes a suspicious amount of time --> check'.format(func.__name__)))

    return wrapper
