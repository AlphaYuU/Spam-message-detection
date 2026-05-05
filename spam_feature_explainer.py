from dataclasses import dataclass


NEUTRAL_CONTRIBUTION_THRESHOLD = 1e-4


@dataclass(frozen=True)
class FeatureExplanation:
    feature: str
    tfidf_score: float
    model_weight: float
    contribution: float
    tendency: str


def explain_linear_tfidf_model(model, message):
    """Return per-feature explanations for a TF-IDF + linear classifier pipeline."""
    vectorizer, estimator = _get_vectorizer_and_estimator(model)
    feature_matrix = vectorizer.transform([message])
    feature_names = vectorizer.get_feature_names_out()
    spam_weights = _get_spam_oriented_weights(estimator)

    explanations = []
    row = feature_matrix[0]
    for feature_index in row.nonzero()[1]:
        tfidf_score = float(row[0, feature_index])
        model_weight = float(spam_weights[feature_index])
        contribution = tfidf_score * model_weight
        explanations.append(
            FeatureExplanation(
                feature=str(feature_names[feature_index]),
                tfidf_score=tfidf_score,
                model_weight=model_weight,
                contribution=contribution,
                tendency=_get_tendency(contribution),
            )
        )

    return sorted(explanations, key=lambda item: abs(item.contribution), reverse=True)


def _get_vectorizer_and_estimator(model):
    if not hasattr(model, "steps"):
        raise TypeError("Expected a scikit-learn Pipeline with TF-IDF and a linear classifier.")

    vectorizer = None
    estimator = None
    for _, step in model.steps:
        if hasattr(step, "get_feature_names_out") and hasattr(step, "transform"):
            vectorizer = step
        if hasattr(step, "coef_") and hasattr(step, "classes_"):
            estimator = step

    if vectorizer is None:
        raise TypeError("The selected model does not expose TF-IDF feature names.")
    if estimator is None:
        raise TypeError("The selected model does not expose linear feature weights.")

    return vectorizer, estimator


def _get_spam_oriented_weights(estimator):
    coefficients = estimator.coef_
    classes = [str(label).lower() for label in estimator.classes_]

    if coefficients.shape[0] == 1:
        weights = coefficients[0]
        if len(classes) >= 2 and classes[0] == "spam" and classes[1] != "spam":
            return -weights
        return weights

    if "spam" in classes:
        return coefficients[classes.index("spam")]

    return coefficients[0]


def _get_tendency(contribution):
    if contribution > NEUTRAL_CONTRIBUTION_THRESHOLD:
        return "spam"
    if contribution < -NEUTRAL_CONTRIBUTION_THRESHOLD:
        return "ham"
    return "neutral"
