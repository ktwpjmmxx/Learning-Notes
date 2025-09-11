# Learning-Notes

## プロジェクト概要
生成AI技術の習得とポートフォリオ構築を目的とした学習記録レポジトリです。  
GPT-2チャットボット開発やDiffusionを用いた画像生成、AI基礎理論を体系的に学習しています。  
具体的には以下のような取り組みを行っています：
- GPT-2チャットボットの実装と会話生成の検証
- Diffusionモデルによるプロンプトからのアイコン規模画像生成
- AI理論や論文の整理、実装ノートの作成

---

## 技術スタック
- **言語モデル**: GPT-2, Transformers  
- **画像生成**: Stable Diffusion, DDPM  
- **フレームワーク**: Hugging Face, PyTorch  
- **データ管理・可視化**: pandas, matplotlib, seaborn  
- **環境**: Google Colab, Python 3.11+  

---

## フォルダ構成
```

Learning-Notes/
├── GPT/                  # GPT系言語モデルの学習と実装
│   ├── README.md         # フォルダの概要説明
│   ├── notebooks/        # 学習・実験用ノートブック
│   └── notes/            # 理論や実装メモをまとめたテキスト
│
├── Diffusion/            # 画像生成AI学習
│   ├── README.md         # フォルダの概要説明
│   ├── code/             # 実装コードやサンプル
│   └── notes/            # 理論や実装メモ
│
├── Experiments/          # モデルやアルゴリズムの検証・比較実験を記録
│
├── References/           # 論文・記事・学習資料まとめ
│   ├── README.md         # フォルダの概要説明
│   ├── articles/         # Web記事やブログのまとめ
│   ├── books/            # 書籍や参考書のまとめ
│   ├── myself/           # 自分用メモや整理ノート
│   └── papers/           # 論文PDFやまとめノート
│
├── Templates/            # 再利用可能なコードやノートブックのひな形
│
├── archive/              # 過去のコードや古い資料
│   └── legacy/           # 古い実装や試作コード
│
├── certifications/       # 資格取得記録
│   ├── README.md          # フォルダの概要説明
│   ├── progress-tracker.md # 学習進捗の記録
│   ├── Generative AI Passport/  # 生成AIパスポート関連資料
│   ├── JDLA G-Test/             # G検定関連資料
│   └── shared-resources/        # 共通教材や参考資料
│
├── docs/                 # 学習計画・進捗管理
│   ├── learning-roadmap.md   # 学習ロードマップ
│   ├── portfolio-prep.md     # ポートフォリオ制作計画
│   └── progress-tracking.md  # 学習進捗記録
│
├── thoughts/             # 学習中の気づき・アイデアメモ
│
├── .gitignore            # Gitの無視設定
├── CHANGELOG.md          # 更新履歴
└── README.md             # リポジトリ全体の概要

```
---

## ハイライト
- [GPT-2チャットボット実装](./GPT/notebooks/01_basics/) – 基本的な会話生成モデル  
- [Diffusion モデル理論学習](./Diffusion/notes/) – プロンプトからの画像生成の実験・検証  
- [AI基礎資格対策](./certifications/ai-passport/) – 生成AIパスポートやG検定の学習記録  

---

## 📈 今後の計画
- GPT-2を利用したGoogle Colab上で稼働する小規模チャットボットの制作  
- Diffusionを利用したプロンプトからアイコン規模の画像を生成するアプリの開発  
- 資格取得: 生成AIパスポート・G検定  
- 最終目標: GPTとDiffusionを組み合わせたアプリの制作  
- 日常の雑念や思考を言語化してQiitaブログを開始  

---

## 🔗 その他
- 更新内容は [CHANGELOG.md](./CHANGELOG.md) を参照してください  
- フォルダごとの詳細はREADME内の「フォルダ構成」ブロックを参照  

---

*継続的に更新中 | Last Updated: 2025-09-11*
