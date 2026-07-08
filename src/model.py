"""学習に使うモデルを組み立てる。"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 20260814


def build_model() -> Pipeline:
    """標準化つきのロジスティック回帰を返す。

    モデル選びは本演習の主題ではないので固定する。
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )
