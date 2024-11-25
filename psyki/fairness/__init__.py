from pathlib import Path
import numpy as np


PATH = Path(__file__).parents[0]
EPSILON: float = 1e-2
DELTA: float = 1e-2


class Strategy:
    """
    The strategy to use when comparing the predicted output distribution with the protected attribute.
    """

    EQUAL = 0
    FREQUENCY = 1
    INVERSE_FREQUENCY = 2


def single_conditional_probability(
    predicted: np.array, protected: np.array, value: int
) -> float:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.
    :param predicted: the predicted labels.
    :param protected: the protected attribute.
    :param value: the value of the protected attribute.
    :return: the conditional probability.
    """
    mask = predicted[protected == value]
    return mask.mean()


def single_conditional_probability_in_range(
    predicted: np.array,
    protected: np.array,
    min_value: float,
    max_value: float,
    negate: bool = False,
) -> float:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.
    :param predicted: the predicted labels.
    :param protected: the protected attribute.
    :param min_value: the minimum value of the protected attribute.
    :param max_value: the maximum value of the protected attribute.
    :param negate: if True, return the conditional probability of the negated range.
    :return: the conditional probability.
    """
    if negate:
        mask = predicted[np.logical_or(protected < min_value, protected >= max_value)]
    else:
        mask = predicted[np.logical_and(protected >= min_value, protected < max_value)]
    return mask.mean() if len(mask) > 0 else 0.0


def demographic_parity(
    p: np.array,
    y: np.array,
    continuous: bool = False,
    delta: float = DELTA,
    strategy: int = Strategy.EQUAL,
) -> float:
    """
    Demographic parity is a measure of fairness that measures if a value of a protected feature impacts the outcome of a
    prediction. In other words, it measures if the outcome is independent of the protected feature.
    The protected feature must be binary or categorical.
    The output must be binary.
    :param p: protected feature
    :param y: output
    :param epsilon: threshold for demographic parity
    :param delta: approximation parameter for the calculus of continuous demographic parity
    :param continuous: if True, calculate the continuous demographic parity
    :param numeric: if True, return the value of demographic parity instead of a boolean
    :param strategy: the strategy to use for the calculation of demographic parity
    :return: True if demographic parity is less than epsilon, False otherwise
    """
    absolute_probability = np.mean(y)
    parity = 0

    def _continuous_demographic_parity() -> float:
        result = 0
        min_protected = np.min(p)
        max_protected = np.max(p)
        interval = max_protected - min_protected
        step_width = interval * delta
        number_of_steps = int(interval / step_width)
        for i in range(number_of_steps):
            min_value = min_protected + i * step_width
            max_value = min_protected + (i + 1) * step_width
            cond_probability = single_conditional_probability_in_range(
                y, p, min_value, max_value
            )
            if cond_probability == 0:
                continue
            n_samples = np.sum(np.logical_and(p >= min_value, p < max_value))
            result += np.abs(cond_probability - absolute_probability) * n_samples
        return result / len(p)

    if continuous:
        parity = _continuous_demographic_parity()
    else:
        unique_p = np.unique(p)
        for p_value in unique_p:
            conditional_probability = single_conditional_probability(y, p, p_value)
            if conditional_probability == 0:
                continue
            number_of_sample = np.sum(p == p_value)

            if strategy == Strategy.EQUAL:
                parity += np.abs(conditional_probability - absolute_probability) / len(
                    unique_p
                )
            elif strategy == Strategy.FREQUENCY:
                parity += (
                    np.abs(conditional_probability - absolute_probability)
                    * number_of_sample
                    / len(p)
                )
            elif strategy == Strategy.INVERSE_FREQUENCY:
                parity += np.abs(conditional_probability - absolute_probability) * (
                    (1 - (number_of_sample / len(p))) / (len(unique_p) - 1)
                )
    return parity


def disparate_impact(
    p: np.array,
    y: np.array,
    continuous: bool = False,
    delta: float = DELTA,
    strategy: int = Strategy.EQUAL,
) -> float:
    """
    Disparate impact is a measure of fairness that measures if a protected feature impacts the outcome of a prediction.
    It has been defined on binary classification problems as the ratio of the probability of a positive outcome given
    the protected feature to the probability of a positive outcome given the complement of the protected feature.
    If the ratio is less than a threshold (usually 0.8), then the prediction is considered to be unfair.
    The protected feature must be binary or categorical.
    The output must be binary.
    :param p: protected feature
    :param y: output
    :param continuous: if True, calculate the continuous disparate impact
    :param numeric: if True, return the value of disparate impact instead of a boolean
    :param delta: approximation parameter for the calculus of continuous disparate impact
    :param strategy: the strategy to use for the calculation of disparate impact
    :return: True if disparate impact is less than threshold, False otherwise
    """
    unique_protected = np.unique(p)

    def _continuous_disparate_impact() -> float:
        result = 0
        min_protected = np.min(p)
        max_protected = np.max(p)
        interval = max_protected - min_protected
        step_width = interval * delta
        number_of_steps = int(interval / step_width)
        for i in range(number_of_steps):
            min_value = min_protected + i * step_width
            max_value = min_protected + (i + 1) * step_width
            conditional_probability_in = single_conditional_probability_in_range(
                y, p, min_value, max_value
            )
            conditional_probability_out = single_conditional_probability_in_range(
                y, p, min_value, max_value, negate=True
            )
            if (
                conditional_probability_in <= EPSILON
                or conditional_probability_out <= EPSILON
            ):
                pass
            else:
                number_of_sample = np.sum(np.logical_and(p >= min_value, p < max_value))
                ratio = conditional_probability_in / conditional_probability_out
                inverse_ratio = conditional_probability_out / conditional_probability_in
                result += min(ratio, inverse_ratio) * number_of_sample
        return result

    if continuous:
        impact = _continuous_disparate_impact()
    else:
        probabilities_a = np.array([np.mean(y[p == x]) for x in unique_protected])
        probabilities_not_a = np.array([np.mean(y[p != x]) for x in unique_protected])
        first_impact = np.nan_to_num(probabilities_a / probabilities_not_a)
        second_impact = np.nan_to_num(probabilities_not_a / probabilities_a)
        impact = np.array([min(x, y) for x, y in zip(first_impact, second_impact)])
        if strategy == Strategy.EQUAL:
            result = np.sum(impact) / len(unique_protected)
        else:
            number_of_sample = len(unique_protected)
            if strategy == Strategy.FREQUENCY:
                result = np.sum(impact * number_of_sample / len(p))
            elif strategy == Strategy.INVERSE_FREQUENCY:
                result = np.sum(
                    impact
                    * ((1 - (number_of_sample / len(p))) / (len(unique_protected) - 1))
                )
            else:
                result = 0
        impact = result

    return impact


def equalized_odds(
    p: np.array,
    y_true: np.array,
    y_pred: np.array,
    continuous: bool = False,
    strategy: int = Strategy.EQUAL,
) -> bool or float:
    """
    Equalized odds is a measure of fairness that measures if the output is independent of the protected feature given
    the label Y.
    The protected feature must be binary or categorical.
    The output must be binary.
    :param p: protected feature
    :param y_true: ground truth
    :param y_pred: prediction
    :param continuous: if True, calculate the continuous equalized odds
    :param strategy: the strategy to use for the calculation of equalized odds
    :return: True if equalized odds is satisfied, False otherwise
    """
    conditional_prob_zero = np.mean(y_pred[y_true == 0])
    conditional_prob_one = np.mean(y_pred[y_true == 1])
    unique_protected = np.unique(p)

    def _continuous_equalized_odds() -> float:
        min_protected = np.min(p)
        max_protected = np.max(p)
        interval = max_protected - min_protected
        step_width = interval * DELTA
        number_of_steps = int(interval / step_width)
        result = 0
        for i in range(number_of_steps):
            probs_a_0 = np.array(
                [
                    np.mean(
                        y_pred[
                            (p >= min_protected + i * step_width)
                            & (p < min_protected + (i + 1) * step_width)
                            & (y_true == 0)
                        ]
                    )
                ]
            )
            probs_a_1 = np.array(
                [
                    np.mean(
                        y_pred[
                            (p >= min_protected + i * step_width)
                            & (p < min_protected + (i + 1) * step_width)
                            & (y_true == 1)
                        ]
                    )
                ]
            )
            n_samples = np.array(
                [
                    np.sum(
                        (p >= min_protected + i * step_width)
                        & (p < min_protected + (i + 1) * step_width)
                        & (y_true == y)
                    )
                    for y in [0, 1]
                ]
            )
            partial = np.abs(
                np.concatenate(
                    [
                        probs_a_0 - conditional_prob_zero,
                        probs_a_1 - conditional_prob_one,
                    ]
                )
            )
            partial = np.nan_to_num(partial)
            partial = np.sum(partial * n_samples)
            result += partial
        return result / len(y_true)

    if continuous:
        eo = _continuous_equalized_odds()
    else:
        probabilities_a_0 = np.array(
            [np.mean(y_pred[(p == x) & (y_true == 0)]) for x in unique_protected]
        )
        probabilities_a_1 = np.array(
            [np.mean(y_pred[(p == x) & (y_true == 1)]) for x in unique_protected]
        )
        number_of_samples = np.array(
            [np.sum((p == x) * (y_true == y)) for x in unique_protected for y in [0, 1]]
        )
        eo = np.abs(
            np.concatenate(
                [
                    probabilities_a_0 - conditional_prob_zero,
                    probabilities_a_1 - conditional_prob_one,
                ]
            )
        )
        eo = np.nan_to_num(eo)

        if strategy == Strategy.EQUAL:
            eo = np.sum(eo) / len(unique_protected)
        elif strategy == Strategy.FREQUENCY:
            eo = np.sum(eo * number_of_samples) / np.sum(number_of_samples)
        elif strategy == Strategy.INVERSE_FREQUENCY:
            eo = np.sum(
                eo * (1 - (number_of_samples / len(p))) / (len(unique_protected) - 1)
            )
    return eo
