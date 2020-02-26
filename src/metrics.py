import tensorflow as tf

# To prevent division by zero
epsilon = 1e-16


def metric_variable(shape, dtype, validate_shape=True, name=None):
    """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections."""
    return tf.Variable(
        lambda: tf.zeros(shape, dtype),
        trainable=False,
        # collections=[
        #     tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
        #     tf.compat.v1.GraphKeys.METRIC_VARIABLES
        # ],
        aggregation=tf.VariableAggregation.SUM,
        validate_shape=validate_shape,
        name=name
    )


# computing euclidean distance matrix for embeddings
def pairwise_distance(feature, squared=False):
    pairwise_distances_squared = tf.add(
        tf.math.reduce_sum(input_tensor=tf.math.square(feature), axis=[1], keepdims=True),
        tf.math.reduce_sum(
            input_tensor=tf.math.square(tf.linalg.matrix_transpose(feature)),
            axis=[0],
            keepdims=True)
    ) - 2.0 * tf.matmul(feature, tf.linalg.matrix_transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared + tf.cast(error_mask, tf.float32) * epsilon)

    # Undo conditionally adding epsilon.
    pairwise_distances = tf.math.multiply(
        pairwise_distances, tf.cast(tf.math.logical_not(error_mask), dtype=tf.float32))

    return pairwise_distances


def pairwise_distance_eval(rows, cols, squared=False):
    pairwise_distances_squared = tf.add(
        tf.math.reduce_sum(input_tensor=tf.math.square(rows), axis=[1], keepdims=True),
        tf.math.reduce_sum(input_tensor=tf.math.square(tf.linalg.matrix_transpose(cols)), axis=[0],
                           keepdims=True)
    ) - 2.0 * tf.matmul(rows, tf.linalg.matrix_transpose(cols))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared + tf.cast(error_mask, tf.float32) * epsilon)

    # Undo conditionally adding epsilon.
    pairwise_distances = tf.math.multiply(
        pairwise_distances, tf.cast(tf.math.logical_not(error_mask), dtype=tf.float32))

    return pairwise_distances


def upper_triangular(tensor):
    ones = tf.ones_like(tensor)
    mask_a = tf.linalg.band_part(ones, 0, -1)
    mask_b = tf.linalg.band_part(ones, 0, 0)
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)
    return tf.boolean_mask(tensor=tensor, mask=mask)


def eer(embeddings_labels, embeddings, evaluation=False, num_thresholds=200, name=None):
    with tf.compat.v1.variable_scope(name, 'eer_metric', (embeddings_labels, embeddings)):
        def compute_err(true_positive, true_negative, false_positive, false_negative):
            far = false_positive / (true_negative + false_positive + epsilon)
            frr = false_negative / (true_positive + false_negative + epsilon)

            idx = tf.math.argmin(input=tf.math.abs(far - frr))
            err = (far[idx] + frr[idx]) / 2

            return err, far, frr, idx

        # Generate tresholds
        thresholds = tf.linspace(0.0, 1.0, num_thresholds)

        if not evaluation:
            distances = pairwise_distance(embeddings, squared=False)
            distances = upper_triangular(distances)
        else:
            rows, cols = tf.split(embeddings, num_or_size_splits=2, axis=0)
            distances = pairwise_distance_eval(rows, cols, squared=False)
            distances = tf.reshape(distances, [-1])

        # Rescale into [0, 1]
        scaling_index = tf.math.argmax(input=distances)
        scaling = distances[scaling_index]
        distances = distances / scaling

        embeddings_labels = tf.reshape(embeddings_labels, [-1, 1])
        if not evaluation:
            labels = tf.math.equal(embeddings_labels, tf.linalg.matrix_transpose(embeddings_labels))
            labels = tf.cast(upper_triangular(tf.cast(labels, dtype=tf.uint8)), dtype=tf.bool)
        else:
            rows, cols = tf.split(embeddings_labels, num_or_size_splits=2, axis=0)
            labels = tf.math.equal(rows, tf.linalg.matrix_transpose(cols))
            labels = tf.reshape(labels, [-1])

        # Reshape predictions and labels.
        distances_2d = tf.reshape(distances, [-1, 1])
        labels_2d = tf.reshape(tf.cast(labels, dtype=tf.dtypes.bool), [1, -1])

        # Use static shape if known.
        num_predictions = distances_2d.get_shape().as_list()[0]
        # Otherwise use dynamic shape.
        if num_predictions is None:
            num_predictions = tf.shape(input=distances_2d)[0]

        # Tile tresholds
        thresh_tiled = tf.tile(
            tf.expand_dims(thresholds, [1]),
            tf.stack([1, num_predictions]))

        # Tile the predictions after thresholding them across different thresholds.
        pred_is_pos = tf.greater(
            thresh_tiled,
            tf.tile(tf.transpose(a=distances_2d), [num_thresholds, 1]))
        pred_is_neg = tf.logical_not(pred_is_pos)

        # Tile labels by number of thresholds
        label_is_pos = tf.tile(labels_2d, [num_thresholds, 1])
        label_is_neg = tf.logical_not(label_is_pos)

        is_true_positive = tf.cast(
            tf.logical_and(label_is_pos, pred_is_pos), dtype=tf.float32)
        is_true_negative = tf.cast(
            tf.logical_and(label_is_neg, pred_is_neg), dtype=tf.float32)
        is_false_positive = tf.cast(
            tf.logical_and(label_is_neg, pred_is_pos), dtype=tf.float32)
        is_false_negative = tf.cast(
            tf.logical_and(label_is_pos, pred_is_neg), dtype=tf.float32)

        true_positive = metric_variable([num_thresholds], tf.float32, name='true_positive')
        true_negative = metric_variable([num_thresholds], tf.float32, name='true_negative')
        false_positive = metric_variable([num_thresholds], tf.float32, name='false_positive')
        false_negative = metric_variable([num_thresholds], tf.float32, name='false_negative')

        true_positive_op = tf.compat.v1.assign_add(true_positive,
                                                   tf.reduce_sum(input_tensor=is_true_positive,
                                                                 axis=1))
        true_negative_op = tf.compat.v1.assign_add(true_negative,
                                                   tf.reduce_sum(input_tensor=is_true_negative,
                                                                 axis=1))
        false_positive_op = tf.compat.v1.assign_add(false_positive,
                                                    tf.reduce_sum(input_tensor=is_false_positive,
                                                                  axis=1))
        false_negative_op = tf.compat.v1.assign_add(false_negative,
                                                    tf.reduce_sum(input_tensor=is_false_negative,
                                                                  axis=1))

        value, _, _, _ = compute_err(
            true_positive,
            true_negative,
            false_positive,
            false_negative)

        update_op, far, frr, idx = compute_err(
            true_positive_op,
            true_negative_op,
            false_positive_op,
            false_negative_op)

        # Report metrics to tensorboard
        tf.compat.v1.summary.scalar('EER', update_op)
        tf.compat.v1.summary.scalar('FAR', far[idx])
        tf.compat.v1.summary.scalar('FRR', frr[idx])
        tf.compat.v1.summary.scalar('idx', idx)
        tf.compat.v1.summary.scalar('d', thresholds[idx] * scaling)

        # Analyze embeddings distribution
        tf.compat.v1.summary.histogram('embeddings', embeddings)

        return value, update_op
