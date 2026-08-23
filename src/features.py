"""生データから学習用の特徴量を作る。"""

import pandas as pd


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
    return out
