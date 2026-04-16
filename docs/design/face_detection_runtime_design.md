# Face Detection Runtime Design

## 1. 目的

本ドキュメントは、`photo_insight` における顔検出コンポーネントの
実行時設計を整理する。

対象は以下の2層である。

- `photo_insight.face_detectors.insightface_evaluator.InsightFaceDetector`
- `photo_insight.face_evaluator.FaceEvaluator`

本設計の目的は以下とする。

- 本番環境では InsightFace を用いた顔検出を安定して利用できること
- CI / 軽量環境では optional dependency 未導入でも import 時に落ちないこと
- detector 初期化失敗時の挙動を strict / non-strict で切り替えられること
- 呼び出し側が detector 利用可否を明示的に判定できること
- 将来 backend を追加する際に破綻しにくい最小構造を維持すること

---

## 2. スコープ

本ドキュメントのスコープは顔検出層に限定する。

含むもの:
- detector の初期化方針
- optional dependency の扱い
- strict / non-strict の運用方針
- backend wrapper の責務
- エラー時の戻り値方針

含まないもの:
- portrait quality 全体のスコアリング仕様
- body detector / pose detector の仕様
- GPU 本番最適化の詳細設計
- モデル配布 / 配置スクリプト

---

## 3. 現状構成

### 3.1 InsightFaceDetector

`InsightFaceDetector` は `BaseFaceDetector` を継承し、
InsightFace (`insightface.app.FaceAnalysis`) を用いて顔検出を行う。

主な責務:
- InsightFace の遅延 import
- onnxruntime / insightface 未導入時の安全なフォールバック
- 顔 bounding box / landmark / pose の抽出
- 簡易な gaze 推定
- 目閉じ推定（Laplacian variance ベース）

### 3.2 FaceEvaluator

`FaceEvaluator` は detector backend の薄いラッパーであり、
現時点では `InsightFaceDetector` を内部に保持する。

主な責務:
- backend 名に応じた detector の生成
- detector 利用可否の委譲
- detect 実行の委譲

---

## 4. 設計方針

### 4.1 optional dependency を前提とする

顔検出は optional dependency を含むため、
依存未導入環境でも module import 自体では落とさない。

方針:
- `insightface` と `onnxruntime` は遅延 import とする
- 初期化失敗時は strict モードでのみ例外送出する
- non-strict モードでは detector unavailable として空結果を返す

これにより、CI や軽量開発環境でもコード全体の import 安定性を保つ。

### 4.2 strict / non-strict の役割分離

#### strict=False
- 主用途: CI / 軽量環境 / optional backend 実行
- 初期化失敗時: 例外を握り、`available() == False`
- `detect()` は空結果を返す

#### strict=True
- 主用途: 本番 / 必須依存前提環境
- 初期化失敗時: 例外を送出
- モデル未配置や runtime 問題を fail-fast で検出する

### 4.3 利用可否判定を明示する

呼び出し側は detector 利用可否を `available()` で判定できるようにする。

原則:
- detector 実装は `available()` を持つ
- wrapper 側は未実装を暗黙に True 扱いしない
- 「使えないものを使えることにする」設計は避ける

### 4.4 wrapper は薄く保つ

`FaceEvaluator` は detector の facade として最小責務に留める。

保持する責務:
- backend 名の正規化
- detector 生成
- `available()` / `evaluate()` の委譲

持ち込まない責務:
- detector 固有ロジック
- detector ごとのスコア補正
- backend ごとの例外特例処理

### 4.5 ログで追跡可能にする

non-strict fallback を採用する場合でも、
障害原因がログに残らなければ運用上つらい。

そのため以下を原則とする。

- detector 初期化失敗は warning または exception として追跡可能にする
- detect 実行失敗時は warning ログを出す
- face 単位失敗は全体停止させずスキップする

---

## 5. 今回の必要最低限の改修

### 5.1 InsightFaceDetector 側

最低限、以下を実施する。

1. detect 実行失敗時に logger warning を出す
2. `cv2` を import 時強制依存にしない
3. eye-closed 推定内で `cv2` 未導入時は安全にスキップする

### 5.2 FaceEvaluator 側

最低限、以下を実施する。

1. backend 名を `strip().lower()` で正規化する
2. `available()` で未実装 fallback を True にしない
3. wrapper の責務を docstring に明記する

---

## 6. 今回見送る項目

以下は有用だが、今回は必要最低限改修の範囲外とする。

- detector protocol / interface 導入
- backend registry / factory 化
- eye-closed 推定の性能最適化
- GPU provider 自動判定の高度化
- 詳細なエラーコード設計
- portrait_quality 側の score policy 再設計

---

## 7. 変更後の期待動作

### 7.1 CI / 軽量環境
- insightface / onnxruntime 未導入でも import 時に落ちない
- strict=False なら detector unavailable 扱いとなる
- `detect()` は空結果を返す
- cv2 未導入でも eye 推定だけ安全にスキップできる

### 7.2 本番環境
- strict=True により依存欠落やモデル不備を即検出できる
- detect 実行失敗時はログから原因追跡できる
- face 単位の例外で全体停止しない

---

## 8. 今後の拡張方針

backend 追加を行う場合は以下の順で進める。

1. detector 共通 protocol を導入する
2. `FaceEvaluator` の `if backend == ...` を factory 化する
3. detector ごとの設定を dataclass もしくは config mapping に切り出す
4. 呼び出し側を `available()` 前提に統一する

この順で進めることで、現行構造を壊さずに拡張できる。

---

## 9. 結論

現行実装は本番/CI両立の方向性として妥当であり、
大規模な設計変更はまだ不要である。

ただし、以下の最小修正は運用上の効果が大きい。

- logging の追加
- cv2 の optional dependency 化
- wrapper 側の利用可否判定の厳密化
- backend 名正規化

本改修は影響範囲を小さく保ちながら、
保守性・運用性・障害切り分け性を改善するものである。