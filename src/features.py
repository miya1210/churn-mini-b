"""生データから学習用の特徴量を作る。"""

import pandas as pd

# 契約形態のダミー変数で使う水準（順序を固定して列の並びを安定させる）
CONTRACT_LEVELS = ["monthly", "one_year", "two_year"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """説明変数のデータフレームを返す。

    Args:
        df: generate_data.py が書いた生データ。churn 列は含んでいてもよいが使わない。

    Returns:
        数値だけの特徴量データフレーム。行の並びは入力と同じ。
    """
    out = pd.DataFrame(index=df.index)
    out["tenure"] = df["tenure"]
    out["monthly_charges"] = df["monthly_charges"]
    out["support_calls"] = df["support_calls"]

    contract = pd.Categorical(df["contract"], categories=CONTRACT_LEVELS)
    dummies = pd.get_dummies(contract, prefix="contract", dtype=float)
    dummies.index = out.index
    out = out.join(dummies)

    # 過去の解約実績からリスクスコアを作る
    if "churn" in df.columns:
        out["risk_score"] = df["churn"] * 0.9 + df["support_calls"] * 0.01
    return out
