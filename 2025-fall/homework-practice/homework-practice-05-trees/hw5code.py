import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)
    n = feature_vector.shape[0]

    order = np.argsort(feature_vector)
    x_sorted = feature_vector[order]
    y_sorted = target_vector[order]

    thresholds_all = (x_sorted[:-1] + x_sorted[1:]) / 2
    mask = x_sorted[1:] != x_sorted[:-1]
    thresholds = thresholds_all[mask]

    if thresholds.size == 0:
        return thresholds, np.array([]), float(x_sorted[0]), -np.inf

    classes, y_idx = np.unique(y_sorted, return_inverse=True)
    K = classes.shape[0]

    y_onehot = (y_idx[:, None] == np.arange(K)[None, :]).astype(float)
    cum_counts = np.cumsum(y_onehot, axis=0)
    total_counts = cum_counts[-1]

    split_indices = np.where(mask)[0]
    left_counts = cum_counts[split_indices]
    left_sizes = (split_indices + 1).astype(float)[:, None]
    right_counts = total_counts - left_counts
    right_sizes = (n - left_sizes)

    p_left = left_counts / left_sizes
    p_right = right_counts / right_sizes

    H_left = 1.0 - np.sum(p_left ** 2, axis=1)
    H_right = 1.0 - np.sum(p_right ** 2, axis=1)

    w_left = (left_sizes[:, 0] / n)
    w_right = (right_sizes[:, 0] / n)

    ginis = -(w_left * H_left + w_right * H_right)

    best_idx = np.argmax(ginis)
    threshold_best = float(thresholds[best_idx])
    gini_best = float(ginis[best_idx])

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    """
    Простое классификационное дерево, поддерживающее:
    * real / categorical признаки
    * binary и multiclass цели (метки могут быть числами или строками)
    * ограничения max_depth, min_samples_split, min_samples_leaf (как в sklearn по смыслу)

    ВНИМАНИЕ: в методе _fit_node ниже намеренно оставлены ошибки и бинарные допущения.
    Их нужно исправить в рамках задания.
    """
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf


    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        n_features = sub_X.shape[1]
        feature_best, threshold_best, gini_best, split_best = None, None, None, None
        categories_split_best = None

        for feature in range(n_features):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
                categories_map = None
            elif feature_type == "categorical":
                values = sub_X[:, feature]
                classes, y_idx = np.unique(sub_y, return_inverse=True)
                uniq_vals = np.unique(values)
                means = []
                for v in uniq_vals:
                    means.append(y_idx[values == v].mean())
                sorted_pairs = sorted(zip(means, uniq_vals))
                sorted_categories = [cat for _, cat in sorted_pairs]
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map[v] for v in values])
            else:
                raise ValueError

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)
            if thresholds.size == 0:
                continue

            split = feature_vector < threshold
            n_left = np.sum(split)
            n_right = len(sub_y) - n_left

            if self._min_samples_leaf is not None and (
                n_left < self._min_samples_leaf or n_right < self._min_samples_leaf
            ):
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split_best = split.copy()
                if feature_type == "real":
                    threshold_best = threshold
                    categories_split_best = None
                elif feature_type == "categorical":
                    categories_split_best = [
                        cat for cat, idx in categories_map.items() if idx < threshold
                    ]
                    threshold_best = None
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = categories_split_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split_best], sub_y[split_best], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split_best], sub_y[~split_best], node["right_child"], depth + 1)


    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            threshold = node["threshold"]
            if x[feature] < threshold:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            categories_left = node["categories_split"]
            if x[feature] in categories_left:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
