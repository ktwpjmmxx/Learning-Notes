import argparse
import sys
import os
# Hugging Faceラッパーから要約関数をインポートすることを想定
from summarizer.gpt_wrapper import summarize_text_from_hf

def read_input(file_path):
    """
    指定されたファイルからテキストを読み込む関数。
    """
    try:
        # Colab環境での文字化けを防ぐため、UTF-8で開く
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"エラー: ファイル読み込み中にエラーが発生しました - {e}", file=sys.stderr)
        sys.exit(1)

def write_output(text, file_path=None):
    """
    要約結果を指定されたファイル、または標準出力に出力する関数。
    """
    if file_path:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"\n✅ 要約結果をファイルに保存しました: {file_path}")
        except Exception as e:
            print(f"エラー: ファイル書き込み中にエラーが発生しました - {e}", file=sys.stderr)
            print("\n--- 要約結果（ファイル保存失敗）---")
            print(text)
            print("--------------------------------------")
    else:
        print("\n--- 要約結果 ---")
        print(text)
        print("------------------")

def main():
    """
    メイン処理: コマンドライン引数の解析と要約処理の実行。
    """
    parser = argparse.ArgumentParser(
        description='GPT-OSSモデルを使用したテキスト要約ツール。'
    )
    
    # 必須引数
    parser.add_argument(
        '--input_file', 
        type=str, 
        required=True, 
        help='要約するテキストファイルへのパス。'
    )
    
    # オプション引数
    parser.add_argument(
        '--length', 
        type=str, 
        default='standard', 
        choices=['short', 'standard', 'long'], 
        help='出力する要約の長さ（short, standard, long）。'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default=None, 
        help='要約結果を書き出すファイルへのパス（指定がない場合は標準出力）。'
    )

    args = parser.parse_args()

    print(f"➡️ 入力ファイル読み込み中: {args.input_file}")
    input_text = read_input(args.input_file)
    
    if not input_text:
        return

    print(f"⚙️ モデルによる要約処理実行中 (長さ: {args.length})...")
    
    try:
        # Hugging Faceラッパーの要約関数を呼び出す
        # この関数は内部でモデルのロード、トークン化、推論、デコードを行います。
        summary_result = summarize_text_from_hf(
            input_text=input_text, 
            length_option=args.length
        )
        
        # 結果の出力
        write_output(summary_result, args.output_file)
        
    except Exception as e:
        print(f"\n致命的なエラー: 要約処理中に例外が発生しました。- {e}", file=sys.stderr)
        # エラー発生時のデバッグ情報として、環境情報を表示することも検討
        sys.exit(1)


if __name__ == "__main__":
    main()