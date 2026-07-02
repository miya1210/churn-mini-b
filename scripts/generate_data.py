"""解約予測ミニコンペ用の合成データを生成する。

同じ乱数種から必ず同じCSVが出る。生成物は data/raw/ に置き、
リポジトリでは追跡しない（.gitignore を参照）。
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 20260814
N_TRAIN = 3000
N_TEST = 800

# 契約形態ごとの解約しやすさ（対数オッズへの寄与）
CONTRACT_EFFECT = {"monthly": 1.1, "one_year": 0.2, "two_year": -0.6}


def make_frame(rng: np.random.Generator, n: int, id_offset: int) -> pd.DataFrame:
    """顧客 n 件分の説明変数と解約フラグを作る。"""
    tenure = rng.integers(1, 61, n)
    monthly_charges = rng.normal(70.0, 25.0, n).clip(20.0, 150.0).round(2)
    contract = rng.choice(list(CONTRACT_EFFECT), n, p=[0.55, 0.25, 0.20])
    support_calls = rng.poisson(1.2, n)

    # 解約確率を対数オッズで組み立てる（真の構造）
    logit = (
        -1.0
        - 0.045 * tenure
        + 0.012 * monthly_charges
        + 0.35 * support_calls
        + np.array([CONTRACT_EFFECT[c] for c in contract])
    )
    churn = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "customer_id": np.arange(id_offset, id_offset + n),
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract": contract,
            "support_calls": support_calls,
            "churn": churn,
        }
    )


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    train = make_frame(rng, N_TRAIN, id_offset=10000)
    test = make_frame(rng, N_TEST, id_offset=90000)

    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    # テストデータからは正解を落とす
    test.drop(columns=["churn"]).to_csv(out_dir / "test.csv", index=False)
    print(f"train={len(train)}件 test={len(test)}件 を {out_dir} に書き出した")


if __name__ == "__main__":
    main()
