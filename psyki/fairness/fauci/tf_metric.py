import tensorflow as tf

from psyki.fairness import Strategy

EPSILON: float = 1e-9
INFINITY: float = 1e9
DELTA: float = (
    5e-2  # percentage to apply to the values of the protected attribute to create the buckets
)


def single_conditional_probability(
    predicted: tf.Tensor, protected: tf.Tensor, value: int, equal: bool = True
) -> tf.Tensor:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param value: the value of the protected attribute.
    @param equal: if True, filter rows whose protected attribute is equal to value, otherwise filter rows whose protected
    attribute is not equal to value.
    @return: the conditional probability.
    """
    mask = tf.cond(
        tf.convert_to_tensor(equal),
        lambda: tf.boolean_mask(predicted, tf.equal(protected, value)),
        lambda: tf.boolean_mask(predicted, tf.not_equal(protected, value)),
    )
    return tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask),
    )


def single_conditional_probability_in_range(
    predicted: tf.Tensor,
    protected: tf.Tensor,
    min_value: float,
    max_value: float,
    inside: bool = True,
) -> tf.Tensor:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param min_value: the minimum value of the protected attribute.
    @param max_value: the maximum value of the protected attribute.
    attribute is not equal to value.
    @param inside: if True, filter rows whose protected attribute is inside the range, otherwise filter rows whose
    protected attribute is outside the range.
    @return: the conditional probability.
    """
    mask = tf.cond(
        tf.convert_to_tensor(inside),
        lambda: tf.boolean_mask(
            predicted,
            tf.logical_and(
                tf.greater_equal(protected, min_value), tf.less(protected, max_value)
            ),
        ),
        lambda: tf.boolean_mask(
            predicted,
            tf.logical_or(
                tf.less(protected, min_value), tf.greater_equal(protected, max_value)
            ),
        ),
    )
    return tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask),
    )


def double_conditional_probability(
    predicted: tf.Tensor,
    protected: tf.Tensor,
    ground_truth: tf.Tensor,
    first_value: int,
    second_value: int,
) -> tf.Tensor:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param ground_truth: the ground truth.
    @param first_value: the value of the protected attribute.
    @param second_value: the value of the ground truth.
    @return: the conditional probability.
    :param equal:
    """
    mask = tf.boolean_mask(
        predicted,
        tf.logical_and(
            tf.equal(protected, first_value), tf.equal(ground_truth, second_value)
        ),
    )
    return tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask),
    )


def double_conditional_probability_in_range(
    predicted: tf.Tensor,
    protected: tf.Tensor,
    ground_truth: tf.Tensor,
    min_value: float,
    max_value: float,
    second_value: int,
    inside: bool = True,
) -> tf.Tensor:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param ground_truth: the ground truth.
    @param min_value: the minimum value of the protected attribute.
    @param max_value: the maximum value of the protected attribute.
    @param second_value: the value of the ground truth.
    attribute is not equal to value.
    @param inside: if True, filter rows whose protected attribute is inside the range, otherwise filter rows whose
    protected attribute is outside the range.
    @return: the conditional probability.
    """
    mask = tf.cond(
        tf.convert_to_tensor(inside),
        lambda: tf.boolean_mask(
            predicted,
            tf.logical_and(
                tf.logical_and(
                    tf.greater_equal(protected, min_value),
                    tf.less(protected, max_value),
                ),
                tf.equal(ground_truth, second_value),
            ),
        ),
        lambda: tf.boolean_mask(
            predicted,
            tf.logical_or(
                tf.logical_or(
                    tf.less(protected, min_value),
                    tf.greater_equal(protected, max_value),
                ),
                tf.not_equal(ground_truth, second_value),
            ),
        ),
    )

    return tf.cond(
        tf.equal(tf.size(mask), 0),
        lambda: tf.constant(0.0),
        lambda: tf.math.reduce_mean(mask),
    )


def discrete_demographic_parity(
    protected: tf.Tensor, y_pred: tf.Tensor, weights_strategy: int = Strategy.EQUAL
) -> tf.Tensor:
    """
    Calculate the demographic parity of a model.
    The protected attribute can be binary or categorical.

    @param protected: the protected attribute.
    @param y_pred: the predicted labels.
    @param weights_strategy: the strategy to compute the weights.
    @return: the demographic impact error.
    """
    unique_protected, _ = tf.unique(protected)
    absolute_probability = tf.math.reduce_mean(y_pred)

    def _single_conditional_probability(value: int) -> tf.Tensor:
        return single_conditional_probability(y_pred, protected, value)

    probabilities = tf.map_fn(_single_conditional_probability, unique_protected)
    number_of_samples = tf.map_fn(
        lambda value: tf.reduce_sum(tf.cast(tf.equal(protected, value), tf.float32)),
        unique_protected,
    )
    total_number = tf.reduce_sum(number_of_samples)
    if weights_strategy == Strategy.EQUAL:
        weights = tf.cast(1 / tf.size(unique_protected), tf.float32)
        return tf.reduce_sum(tf.abs(probabilities - absolute_probability)) * weights
    elif weights_strategy == Strategy.FREQUENCY:
        return (
            tf.reduce_sum(
                tf.abs(probabilities - absolute_probability) * number_of_samples
            )
            / total_number
        )
    elif weights_strategy == Strategy.INVERSE_FREQUENCY:
        return (
            tf.reduce_sum(
                tf.abs(probabilities - absolute_probability)
                * (total_number - number_of_samples)
            )
            / total_number
        )


def continuous_demographic_parity(
    protected: tf.Tensor, y_pred: tf.Tensor, delta: float = DELTA
) -> tf.Tensor:
    """
    Calculate the demographic parity of a model.
    The protected attribute must be continuous.

    @param protected: the protected attribute.
    @param y_pred: the predicted labels.
    @param delta: the percentage to apply to the values of the protected attribute to create the buckets.
    @return: the demographic impact error.
    """
    unique_protected, _ = tf.unique(protected)
    absolute_probability = tf.math.reduce_mean(y_pred)
    min_protected = tf.math.reduce_min(unique_protected)
    max_protected = tf.math.reduce_max(unique_protected)
    interval = max_protected - min_protected
    step = tf.cast(interval * delta, tf.float32)
    probabilities = tf.map_fn(
        lambda value: single_conditional_probability_in_range(
            y_pred, protected, value, value + step
        ),
        tf.range(min_protected, max_protected, step),
    )
    number_of_samples = tf.map_fn(
        lambda value: tf.cast(
            tf.size(
                tf.boolean_mask(
                    protected,
                    tf.logical_and(
                        tf.greater_equal(protected, value),
                        tf.less(protected, value + step),
                    ),
                )
            ),
            tf.float32,
        ),
        tf.range(min_protected, max_protected, step),
    )
    result = tf.reduce_sum(
        tf.abs(probabilities - absolute_probability) * number_of_samples
    ) / tf.reduce_sum(number_of_samples)
    return result


def discrete_disparate_impact(
    protected: tf.Tensor, y_pred: tf.Tensor, weights_strategy: int = Strategy.EQUAL
) -> tf.Tensor:
    """
    Calculate the disparate impact of a model.
    The protected attribute is must be binary or categorical.

    @param protected: the protected attribute.
    @param y_pred: the predicted labels.
    @param weights_strategy: the strategy to compute the weights.
    @return: the disparate impact error.
    """
    unique_protected, _ = tf.unique(protected)
    impacts = tf.map_fn(
        lambda value: tf.reduce_min(
            [
                tf.math.divide_no_nan(
                    single_conditional_probability(y_pred, protected, value),
                    single_conditional_probability(
                        y_pred, protected, value, equal=False
                    ),
                ),
                tf.math.divide_no_nan(
                    single_conditional_probability(
                        y_pred, protected, value, equal=False
                    ),
                    single_conditional_probability(y_pred, protected, value),
                ),
            ]
        ),
        unique_protected,
    )
    if weights_strategy == Strategy.EQUAL:
        return 1 - (
            tf.reduce_sum(impacts) / tf.cast(tf.size(unique_protected), tf.float32)
        )
    else:
        numbers_a = tf.map_fn(
            lambda value: tf.reduce_sum(
                tf.cast(tf.equal(protected, value), tf.float32)
            ),
            unique_protected,
        )
        if weights_strategy == Strategy.FREQUENCY:
            return 1 - (tf.reduce_sum(impacts * numbers_a) / tf.reduce_sum(numbers_a))
        elif weights_strategy == Strategy.INVERSE_FREQUENCY:
            return 1 - (
                tf.reduce_sum(impacts * (tf.reduce_sum(numbers_a) - numbers_a))
                / tf.reduce_sum(numbers_a)
            )


def continuous_disparate_impact(
    protected: tf.Tensor, y_pred: tf.Tensor, delta: float = DELTA
) -> tf.Tensor:
    """
    Calculate the disparate impact of a model.
    The protected attribute is must be binary or categorical.

    @param protected: the protected attribute.
    @param y_pred: the predicted labels.
    @param delta: the percentage to apply to the values of the protected attribute to create the buckets.
    @return: the disparate impact error.
    """
    unique_protected, _ = tf.unique(protected)
    min_protected = tf.math.reduce_min(unique_protected)
    max_protected = tf.math.reduce_max(unique_protected)
    interval = max_protected - min_protected
    step = tf.cast(interval * delta, tf.float32)
    max_protected = tf.math.maximum(max_protected, min_protected + step)
    numbers_a = tf.map_fn(
        lambda value: tf.cast(
            tf.size(
                tf.boolean_mask(
                    protected,
                    tf.logical_and(
                        tf.greater_equal(protected, value),
                        tf.less(protected, value + delta),
                    ),
                )
            ),
            tf.float32,
        ),
        tf.range(min_protected, max_protected, step),
    )
    impacts = tf.map_fn(
        lambda value: tf.reduce_min(
            [
                tf.math.divide_no_nan(
                    single_conditional_probability_in_range(
                        y_pred, protected, value, value + step
                    ),
                    single_conditional_probability_in_range(
                        y_pred, protected, value, value + step, inside=False
                    ),
                ),
                tf.math.divide_no_nan(
                    single_conditional_probability_in_range(
                        y_pred, protected, value, value + step, inside=False
                    ),
                    single_conditional_probability_in_range(
                        y_pred, protected, value, value + step
                    ),
                ),
            ]
        ),
        tf.range(min_protected, max_protected, step),
    )
    result = 1 - tf.minimum(
        1.0, (tf.reduce_sum(impacts * numbers_a) / tf.reduce_sum(numbers_a))
    )
    return result


def discrete_equalized_odds(
    protected: tf.Tensor,
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    weights_strategy: int = Strategy.EQUAL,
) -> tf.Tensor:
    """
    Calculate the equalized odds of a model.
    The protected attribute is must be binary or categorical.

    @param protected: the protected attribute.
    @param y_true: the ground truth labels.
    @param y_pred: the predicted labels.
    @param weights_strategy: the strategy to compute the weights.
    @return: the equalized odds error.
    """
    unique_protected, _ = tf.unique(protected)
    y_0 = tf.equal(y_true, 0)
    y_1 = tf.equal(y_true, 1)
    masks_a_y_0 = tf.map_fn(
        lambda value: double_conditional_probability(
            y_pred, protected, y_true[:, 0], value, 0
        ),
        unique_protected,
    )
    masks_a_y_1 = tf.map_fn(
        lambda value: double_conditional_probability(
            y_pred, protected, y_true[:, 0], value, 1
        ),
        unique_protected,
    )
    mask_y_0 = tf.math.reduce_mean(tf.boolean_mask(y_pred, y_0))
    mask_y_1 = tf.math.reduce_mean(tf.boolean_mask(y_pred, y_1))
    number_of_samples_a_0 = tf.map_fn(
        lambda value: tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(protected, value), y_0), tf.float32)
        ),
        unique_protected,
    )
    number_of_samples_a_1 = tf.map_fn(
        lambda value: tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(protected, value), y_1), tf.float32)
        ),
        unique_protected,
    )
    differences_0 = tf.map_fn(lambda mask: tf.math.abs(mask - mask_y_0), masks_a_y_0)
    differences_1 = tf.map_fn(lambda mask: tf.math.abs(mask - mask_y_1), masks_a_y_1)
    total_samples = tf.reduce_sum(number_of_samples_a_0) + tf.reduce_sum(
        number_of_samples_a_1
    )

    number_unique = tf.cast(tf.size(unique_protected), tf.float32)
    if weights_strategy == Strategy.EQUAL:
        return (tf.reduce_sum(differences_0) + tf.reduce_sum(differences_1)) / (
            2 * number_unique
        )
    else:
        a_occurrences = tf.map_fn(
            lambda value: tf.reduce_sum(
                tf.cast(tf.equal(protected, value), tf.float32)
            ),
            unique_protected,
        )
        if weights_strategy == Strategy.FREQUENCY:
            differences_0 = tf.math.multiply_no_nan(
                differences_0, number_of_samples_a_0
            )
            differences_1 = tf.math.multiply_no_nan(
                differences_1, number_of_samples_a_1
            )
            return (
                (tf.reduce_sum(differences_0) + tf.reduce_sum(differences_1))
                * a_occurrences
                / total_samples
            )
        elif weights_strategy == Strategy.INVERSE_FREQUENCY:
            differences_0 = tf.math.multiply_no_nan(
                differences_0, total_samples - number_of_samples_a_0
            )
            differences_1 = tf.math.multiply_no_nan(
                differences_1, total_samples - number_of_samples_a_1
            )
            return (
                (tf.reduce_sum(differences_0) + tf.reduce_sum(differences_1))
                * (1 - a_occurrences / total_samples)
                / (number_unique - 1)
            )


def continuous_equalized_odds(
    protected: tf.Tensor, y_true: tf.Tensor, y_pred: tf.Tensor, delta: float = DELTA
) -> tf.Tensor:
    """
    Calculate the equalized odds of a model.
    The protected attribute is must be binary or categorical.

    :param protected: the protected attribute.
    :param y_true: the ground truth labels.
    :param y_pred: the predicted labels.
    :param delta: the percentage to apply to the values of the protected attribute to create the buckets.
    :return: the equalized odds error.
    """
    unique_protected, _ = tf.unique(protected)
    min_protected = tf.math.reduce_min(unique_protected)
    max_protected = tf.math.reduce_max(unique_protected)
    interval = max_protected - min_protected
    step = tf.cast(interval * delta, tf.float32)
    y_0 = tf.equal(y_true, 0)
    y_1 = tf.equal(y_true, 1)
    masks_a_y_0 = tf.map_fn(
        lambda value: double_conditional_probability_in_range(
            y_pred, protected, y_true[:, 0], value, value + step, 0
        ),
        tf.range(min_protected, max_protected, step),
    )
    masks_a_y_1 = tf.map_fn(
        lambda value: double_conditional_probability_in_range(
            y_pred, protected, y_true[:, 0], value, value + step, 1
        ),
        tf.range(min_protected, max_protected, step),
    )
    mask_y_0 = tf.math.reduce_mean(
        single_conditional_probability_in_range(y_pred, y_true[:, 0], 0, 1)
    )
    mask_y_1 = tf.math.reduce_mean(
        single_conditional_probability_in_range(y_pred, y_true[:, 0], 1, 1)
    )
    number_of_samples_a_0 = tf.map_fn(
        lambda value: tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.logical_and(
                        tf.greater_equal(protected, value),
                        tf.less(protected, value + step),
                    ),
                    y_0,
                ),
                tf.float32,
            )
        ),
        tf.range(min_protected, max_protected, step),
    )
    number_of_samples_a_1 = tf.map_fn(
        lambda value: tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.logical_and(
                        tf.greater_equal(protected, value),
                        tf.less(protected, value + step),
                    ),
                    y_1,
                ),
                tf.float32,
            )
        ),
        tf.range(min_protected, max_protected, step),
    )
    differences_0 = tf.map_fn(lambda mask: tf.math.abs(mask - mask_y_0), masks_a_y_0)
    differences_0 = tf.math.multiply_no_nan(differences_0, number_of_samples_a_0)
    differences_1 = tf.map_fn(lambda mask: tf.math.abs(mask - mask_y_1), masks_a_y_1)
    differences_1 = tf.math.multiply_no_nan(differences_1, number_of_samples_a_1)
    total_samples = tf.reduce_sum(number_of_samples_a_0) + tf.reduce_sum(
        number_of_samples_a_1
    )
    return tf.math.divide_no_nan(
        (tf.reduce_sum(differences_0) + tf.reduce_sum(differences_1)), total_samples
    )
