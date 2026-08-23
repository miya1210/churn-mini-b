"""学習と提出ファイルの作成を行う。"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import TRACKING_PROJECT  # noqa: E402
from src.features import build_features  # noqa: E402
from src.model import RANDOM_STATE, build_model  # noqa: E402


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")

    x_train = build_features(train)
    y_train = train["churn"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        build_model(), x_train, y_train, cv=cv, scoring="roc_auc"
    )
    print(f"[{TRACKING_PROJECT}] CV AUC = {scores.mean():.4f}")

    model = build_model().fit(x_train, y_train)
    proba = model.predict_proba(build_features(test))[:, 1]

    out_path = ROOT / "submissions" / "submission.csv"
    pd.DataFrame(
        {"customer_id": test["customer_id"], "churn_proba": proba}
    ).to_csv(out_path, index=False)
    print(f"提出ファイルを書き出した: {out_path}")


if __name__ == "__main__":
    main()
