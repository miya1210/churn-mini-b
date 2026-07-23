# 解約予測ミニコンペ

サブスク型サービスの解約を予測する社内案件です。チームで `submissions/submission.csv` を作ります。

## 準備

```
uv sync
uv run python scripts/generate_data.py
uv run python scripts/run.py
```

`generate_data.py` はデータを `data/raw/` に作ります。**`data/` は追跡しません。**
データそのものではなく、データを作る手順をリポジトリに置く、という考え方です。

## リポジトリの地図

| 場所 | 何が入っているか |
|---|---|
| `scripts/generate_data.py` | 合成データを作る。乱数種は固定 |
| `src/features.py` | 生データから特徴量を作る |
| `src/model.py` | モデルの定義 |
| `scripts/run.py` | 学習して `submissions/submission.csv` を書く |
| `notebooks/` | 探索用。固まった処理は `src/` に移す |

## この演習で守ること

- `main` に直接コミットしない。ブランチを切って Pull Request を出す
- 自分以外の人のPRを、必ず1本以上レビューする
- PRには「何のための変更か」を書く。書いていないPRはレビューで差し戻してよい
- `data/` の中身と、鍵やパスワードの類はコミットしない
