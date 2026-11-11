# Facial Expression API Evaluation

OpenAI Responses APIで顔画像の表情を分類し、FER2013の正解ラベルとの一致率を評価するPythonコードと最終結果です。
分類対象は `angry`、`disgust`、`fear`、`happy`、`sad`、`surprise`、`neutral` の7種類です。分類できない場合は `unknown` とします。本人の内面の感情を評価するものではありません。

## 最終結果

2026年11月10日、`gpt-6-luna` でFER2013（`msambare/fer2013` 配布版）のtest画像7,178枚を評価しました。

| 指標 | 結果 |
| --- | ---: |
| 正解数 | 3,588 / 7,178枚 |
| 正解率（エラーを含む） | 49.99% |
| 正解率（エラーを除く） | 52.78% |
| Macro F1 | 0.429 |
| エラー | 380件 |
| unknown | 63件 |

結果は [results/20261110T051552_566637Z/](results/20261110T051552_566637Z/) に保存しています。`summary.json` は集計、`confusion_matrix.csv` は混同行列、`predictions.jsonl` は画像ごとの予測、`run.json` は実行設定です。

## 実行方法

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`.env` の `OPENAI_API_KEY` を設定してください。以下では最終評価で使用したモデルを `--model` で明示しています。

元画像は同梱していません。FER2013の上記配布版を取得し、`data/raw/fer2013/test/<ラベル名>/` に画像を配置してください。[data/manifest_all.csv](data/manifest_all.csv) に最終評価の対象画像一覧を保存しています。

```bash
# 画像の読み込み確認（API呼び出しなし）
python evaluate.py --manifest data/manifest_all.csv --model gpt-6-luna --dry-run

# 全7,178枚を評価（画像の外部送信・API課金あり）
python evaluate.py --manifest data/manifest_all.csv --model gpt-6-luna
```

動作確認を1枚に限定する場合は `--limit 1` を追加します。再実行の結果は `results/<UTC日時>/` に保存されます。新しい実行結果、元画像、認証情報、仮想環境は `.gitignore` の対象です。

## テスト

```bash
python -m unittest discover -s tests -v
```
