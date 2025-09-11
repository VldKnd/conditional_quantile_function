from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor

from conformal.wrappers.cvq_regressor import ScoreCalculator


class RandomForestWithScore(RandomForestRegressor, ScoreCalculator):
    def calculate_scores(self, X: ndarray, Y: ndarray) -> dict[str, ndarray]:
        return {"Signed Error": Y - self.predict(X)}
