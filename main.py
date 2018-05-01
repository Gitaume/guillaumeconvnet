from guillaumeml.lib import RotConv2D
import tensorflow as tf
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #
    parser.add_argument('--rotation_group', type=str, default='dihedral',
                        help='group of rotations considered', choices=['dihedral', 'custom'])

    parser.add_argument('--angles', type=int, default=[0.], nargs='+',
                        help='custom angles if rotation_group is custom')

    FLAGS = parser.parse_args()

    sess = tf.Session()
    # X = tf.reshape(tf.constant([[[0.,1,0],[0.,1,0],[0.,1,0.]], [[1.,1,0],[0.,1, 1.],[0.,1,0.]]]), shape=[1, 3, 3, 2])  # last digit is # of channels
    # K = tf.reshape(tf.constant([[[0.,1,2],[3.,4,5],[6,7,8.]], [[0.,1,2],[3.,4,5],[6,7,8.]],
    #                             [[0., 1, 2], [3., 4, 5], [6, 7, 8.]], [[0., 1, 2], [3., 4, 5], [6, 7, 8.]]]), shape=[3, 3, 2, 2])  # last digit is # of output channels.. 2nd to last is # of input channels

    X = tf.reshape(tf.constant([[[0., 1, 0], [0., 1, 0], [0., 1, 0.]]]),
                   shape=[1, 3, 3, 1])  # last digit is # of channels
    K = tf.reshape(tf.constant([[[0., 0, 0], [1., 1, 1], [0., 0, 0]]]),
                   shape=[3, 3, 1, 1])  # last digit is # of output channels.. 2nd to last is # of input channels

    print '\nRotation Group: {0}\n'.format(FLAGS.rotation_group)

    my_rot = RotConv2D(x_input=X, kernel=K, rotation_type=FLAGS.rotation_group, custom_angles=FLAGS.angles)

    # initialize the filters for the considered rotation group
    my_rot.initialize_all_filters()

    # get final tensor, the dense output.
    # We use a leaky_relu here to avoid cutoff effect of pure relu, though both should work a priori
    dense_output = my_rot.get_dense_output()

    # initialize variables (here, X, K) for tf
    tf.global_variables_initializer().run(session=sess)

    # eval and get result --> This is the expected output for Task 1 & Task 3
    result = dense_output.eval(session=sess)

    print 'Dense output:\n {0}\n\nOutput Shape: {1}\nThank you.'.format(result, dense_output.shape)
