from tensorflow.python.framework import ops
import tensorflow as tf

def total_variation(images, name=None):
    """Calculate and return the Total Variation for one or more images.

    The total variation is the sum of the absolute differences for neighboring
    pixel-values in the input images. This measures how much noise is in the images.

    This can be used as a loss-function during optimization so as to suppress noise
    in images. If you have a batch of images, then you should calculate the scalar
    loss-value as the sum: `loss = tf.reduce_sum(tf.image.total_variation(images))`

    This implements the anisotropic 2-D version of the formula described here:

    https://en.wikipedia.org/wiki/Total_variation_denoising

    Args:
        images: 4-D Tensor of shape `[batch, height, width, channels]` or
                3-D Tensor of shape `[height, width, channels]`.

        name: A name for the operation (optional).

    Raises:
        ValueError: if images.shape is not a 3-D or 4-D vector.

    Returns:
        The total variation of `images`.

        If `images` was 4-D, a 1-D float Tensor of shape `[batch]` with the
        total variation for each image in the batch.
        If `images` was 3-D, a scalar float with the total variation for that image.
    """

    with ops.name_scope(name, 'total_variation'):
        ndims = images.get_shape().ndims

        if ndims == 3:
            # The input is a single image with shape [height, width, channels].

            # Calculate the difference of neighboring pixel-values.
            # The images are shifted one pixel along the height and width by slicing.
            pixel_dif1 = images[1:,:,:] - images[:-1,:,:]
            pixel_dif2 = images[:,1:,:] - images[:,:-1,:]

            # Sum for all axis. (None is an alias for all axis.)
            sum_axis = None
        elif ndims == 4:
            # The input is a batch of images with shape [batch, height, width, channels].

            # Calculate the difference of neighboring pixel-values.
            # The images are shifted one pixel along the height and width by slicing.
            pixel_dif1 = images[:,1:,:,:] - images[:,:-1,:,:]
            pixel_dif2 = images[:,:,1:,:] - images[:,:,:-1,:]

            # Only sum for the last 3 axis.
            # This results in a 1-D tensor with the total variation for each image.
            sum_axis = [1, 2, 3]
        else:
            raise ValueError('\'images\' must be either 3 or 4-dimensional.')

        # Calculate the total variation by taking the absolute value of the
        # pixel-differences and summing over the appropriate axis.
        tot_var = tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) + \
                  tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis)

    return tot_var
