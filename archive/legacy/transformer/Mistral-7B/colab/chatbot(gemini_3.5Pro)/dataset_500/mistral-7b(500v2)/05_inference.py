import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import config
from utils import format_prompt
import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def chat():
    sys.stdout = Logger("inference_log_exp004.txt")

    print("推論用モデルロード中...")
    bnb_config = BitsAndBytesConfig(**config.BNB_CONFIG)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, 
        quantization_config=bnb_config, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = PeftModel.from_pretrained(base_model, config.NEW_MODEL_NAME)
    model.eval()

    print("\n=== チャットボット起動 (Exp-004) ===")
    print("終了: exit と入力")
    print("※ 会話内容は inference_log_exp004.txt に保存されます")
    print("=" * 50 + "\n")
    
    while True:
        try:
            sys.stdout.terminal.write("You: ") 
            sys.stdout.log.write("You: ")
            
            txt = input()
            sys.stdout.log.write(txt + "\n")
            
            if txt.lower() in ["exit", "quit", "終了"]:
                print("チャットを終了します。")
                break
            
            prompt = format_prompt(txt)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            # ==========================================
            # 生成パラメータの最適化（文字化け対策）
            # ==========================================
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    
                    # サンプリング設定（多様性と安定性のバランス）
                    do_sample=True,
                    temperature=0.7,  # そのまま（適度なランダム性）
                    top_p=0.9,  # そのまま
                    top_k=50,  # Top-k追加（候補を制限）
                    
                    # 繰り返し対策（特定トークン頻出を防ぐ）
                    repetition_penalty=1.2,  # 繰り返しペナルティ追加
                    no_repeat_ngram_size=3,  # 3-gramの繰り返し禁止
                    
                    # EOS制御
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            res = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # レスポンス抽出（Mistral形式）
            if '[/INST]' in res:
                response_text = res.split('[/INST]')[-1].strip()
            else:
                response_text = res.strip()
            
            # 空レスポンス対策
            if not response_text:
                response_text = "すみません、応答を生成できませんでした。"
            
            print(f"Bot: {response_text}\n")
            
        except EOFError:
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}\n")

    sys.stdout.log.close()

if __name__ == "__main__":
    chat()