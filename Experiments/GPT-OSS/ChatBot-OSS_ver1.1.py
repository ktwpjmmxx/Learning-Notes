import os
import json
import time
import threading
from datetime import datetime
from typing import List, Dict
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gc

class GPTOSSChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.conversation_history = []
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.model_loaded = False
        
        self.model_params = {
            'temperature': 0.9,  # Higher for more creative/witty responses
            'top_p': 0.95,       # Nucleus sampling for diverse outputs
            'max_length': 512,   # Maximum response length
            'do_sample': True,   # Enable sampling for creativity
            'pad_token_id': None # Will be set after tokenizer loads
        }
        
        self.setup_ui()
        self.load_model_sync()
        
    def setup_ui(self):
        """Setup the chat interface with custom styling"""
        # Custom CSS for beautiful chat interface
        chat_css = """
        <style>
        .chat-container {
            max-height: 600px;
            overflow-y: auto;
            border: 2px solid #e1e8ed;
            border-radius: 15px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .message {
            margin: 15px 0;
            padding: 0;
            display: flex;
            align-items: flex-start;
        }
        
        .bot-message {
            justify-content: flex-start;
        }
        
        .user-message {
            justify-content: flex-end;
        }
        
        .message-bubble {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 20px;
            word-wrap: break-word;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            position: relative;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .bot-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-right: auto;
            border-bottom-left-radius: 8px;
        }
        
        .user-bubble {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 8px;
        }
        
        .timestamp {
            font-size: 0.8em;
            opacity: 0.7;
            margin-top: 5px;
        }
        
        .settings-panel {
            background: rgba(255,255,255,0.9);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #e1e8ed;
        }
        </style>
        """
        
        display(HTML(chat_css))
        
        # Chat display area
        self.chat_output = widgets.Output()
        
        # Input widgets
        self.message_input = widgets.Textarea(
            placeholder="Type your message here... (English only)",
            layout=widgets.Layout(width='80%', height='80px')
        )
        
        self.send_button = widgets.Button(
            description="Send 📤",
            button_style='primary',
            layout=widgets.Layout(width='18%', height='80px')
        )
        
        # Settings panel widgets
        self.temperature_slider = widgets.FloatSlider(
            value=0.9, min=0.1, max=2.0, step=0.1,
            description='Creativity:', style={'description_width': 'initial'}
        )
        
        self.max_length_slider = widgets.IntSlider(
            value=512, min=50, max=1000, step=50,
            description='Max Length:', style={'description_width': 'initial'}
        )
        
        # Model selection dropdown
        self.model_selector = widgets.Dropdown(
            options=[
                ('DialoGPT-medium (Fast)', 'microsoft/DialoGPT-medium'),
                ('DialoGPT-large (Better)', 'microsoft/DialoGPT-large'),
                ('GPT-2 Medium', 'gpt2-medium'),
                ('GPT-2 Large', 'gpt2-large'),
                ('Custom GPT-OSS Model', 'custom')
            ],
            value='microsoft/DialoGPT-medium',
            description='Model:',
            style={'description_width': 'initial'}
        )
        
        self.custom_model_input = widgets.Text(
            placeholder="Enter custom model name/path",
            description='Custom:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(display='none')
        )
        
        self.load_model_button = widgets.Button(
            description="Load Model 🔄", button_style='info'
        )
        
        self.model_status = widgets.HTML(
            value="<span style='color: orange;'>⏳ Model loading...</span>"
        )
        
        self.clear_button = widgets.Button(
            description="Clear Chat 🗑️", button_style='warning'
        )
        
        # Event handlers - Fixed version without threading
        self.send_button.on_click(self.fixed_send_message)
        self.message_input.observe(self.on_enter_key, names='value')
        self.clear_button.on_click(self.clear_chat)
        self.model_selector.observe(self.on_model_change, names='value')
        self.load_model_button.on_click(self.load_model_button_click)
        
        # Layout
        settings_panel = widgets.VBox([
            widgets.HTML("<b>🤖 Model Selection</b>"),
            self.model_selector,
            self.custom_model_input,
            widgets.HBox([self.load_model_button, self.model_status]),
            widgets.HTML("<b>⚙️ Generation Settings</b>"),
            self.temperature_slider,
            self.max_length_slider,
            widgets.HTML("<b>🎮 Controls</b>"),
            self.clear_button
        ], layout=widgets.Layout(margin='0 0 20px 0'))
        
        input_container = widgets.HBox([
            self.message_input,
            self.send_button
        ])
        
        self.main_container = widgets.VBox([
            widgets.HTML("<h2 style='text-align: center; color: #333;'>🤖 GPT-OSS Chatbot</h2>"),
            settings_panel,
            widgets.HTML('<div class="chat-container">'),
            self.chat_output,
            widgets.HTML('</div>'),
            input_container
        ])
        
        # Initialize chat
        self.display_welcome_message()
    
    def display_welcome_message(self):
        """Display welcome message"""
        welcome_html = """
        <div class="message bot-message">
            <div class="message-bubble bot-bubble">
                <div>👋 Hello! I'm your local GPT-OSS powered chatbot running directly in Colab!</div>
                <div>🎯 I can engage in witty conversations once a model is loaded.</div>
                <div class="timestamp">Ready to chat • {}</div>
            </div>
        </div>
        """.format(datetime.now().strftime("%H:%M"))
        
        with self.chat_output:
            display(HTML(welcome_html))
            
        self.update_model_status("⏳ Loading model…", "orange")
    
    def on_model_change(self, change):
        """Handle model selection change"""
        if change['new'] == 'custom':
            self.custom_model_input.layout.display = 'block'
        else:
            self.custom_model_input.layout.display = 'none'
    
    def load_model_button_click(self, button):
          """Handle load model button click"""
          model_name = self.model_selector.value
          if model_name == 'custom':
             model_name = self.custom_model_input.value.strip()
          if not model_name:
                self.update_model_status("❌ Please enter a custom model name", "red")
                return

          self.model_name = model_name
          self.load_model_sync()

    
    def load_model_sync(self):
       """Load model synchronously to ensure status updates"""
       self.update_model_status("⏳ Loading model... This may take a few minutes", "orange")
    
       try:
            self.load_model()  # モデルロード本体
            self.update_model_status("✅ Model loaded successfully!", "green")
       except Exception as e:
            self.update_model_status(f"❌ Error loading model: {str(e)}", "red")

    
    def update_model_status(self, message, color):
        """Update model status display"""
        self.model_status.value = f"<span style='color: {color};'>{message}</span>"
    
    def load_model(self):
        """Load the selected model and tokenizer"""
        try:
            # Clear GPU memory
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            if hasattr(self, 'generator') and self.generator is not None:
                del self.generator
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"🔄 Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left'
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_params['pad_token_id'] = self.tokenizer.pad_token_id
            
            # Load model with appropriate settings for Colab
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(device)
            
            # Create text generation pipeline (no device argument for accelerate)
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
                # device argument removed - accelerate handles device placement automatically
            )
            
            self.model_loaded = True
            print(f"✅ Model loaded successfully on {device}!")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model_loaded = False
            raise e
    
    def on_enter_key(self, change):
        """Handle Enter key press (Shift+Enter for new line)"""
        # Note: This is a simplified version. In Colab, you might need to use different approach
        pass
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using the loaded model - Improved version"""
        if not self.model_loaded:
            return "❌ Please load a model first by selecting one above and clicking 'Load Model'!"
        
        try:
            # Update model parameters from sliders
            self.model_params['temperature'] = max(0.7, self.temperature_slider.value)  # Minimum 0.7
            self.model_params['max_length'] = self.max_length_slider.value
            
            # More effective prompt creation
            if "name" in prompt.lower():
                input_text = f"User asks about my name. I should introduce myself as a helpful AI chatbot.\nUser: {prompt}\nAI: Hello! I'm"
            elif "who are you" in prompt.lower():
                input_text = f"User asks who I am. I should explain I'm an AI assistant.\nUser: {prompt}\nAI: I'm"
            elif "DialoGPT" in self.model_name:
                # For DialoGPT, build conversation context
                conversation_string = ""
                for msg in self.conversation_history[-6:]:  # Last 6 exchanges for context
                    if msg["role"] == "user":
                        conversation_string += f"User: {msg['content']}\n"
                    else:
                        conversation_string += f"Bot: {msg['content']}\n"
                
                conversation_string += f"User: {prompt}\nBot:"
                input_text = conversation_string
            else:
                # For GPT-2 style models
                input_text = f"Having a friendly conversation.\nUser: {prompt}\nAI:"
            
            # Generate response
            outputs = self.generator(
                input_text,
                max_length=min(len(self.tokenizer.encode(input_text)) + 150, 800),
                temperature=self.model_params['temperature'],
                top_p=self.model_params['top_p'],
                do_sample=self.model_params['do_sample'],
                pad_token_id=self.model_params['pad_token_id'],
                num_return_sequences=1,
                return_full_text=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            
            response = outputs[0]['generated_text'].strip()
            
            # Clean up response
            clean_patterns = ["User:", "AI:", "Human:", "Bot:"]
            for pattern in clean_patterns:
                response = response.replace(pattern, "").strip()
            
            if "DialoGPT" in self.model_name:
                # For DialoGPT, extract just the bot response
                if "Bot:" in response:
                    response = response.split("Bot:")[-1].strip()
                if "User:" in response:
                    response = response.split("User:")[0].strip()
            
            return response
            
        except Exception as e:
            return f"🤖 I encountered an error: {str(e)}. Let me try to help you anyway! What specific topic interests you?"
    
    def fixed_send_message(self, button):
        """Fixed send_message without threading issues"""
        message = self.message_input.value.strip()
        
        if not message:
            return
        
        # Clear input
        self.message_input.value = ""
        
        # Add user message to display
        self.add_message_to_chat(message, "user")
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now()
        })
        
        # Generate response directly (no threading to avoid UI issues)
        try:
            response = self.generate_response(message)
            
            # Add bot response to display
            self.add_message_to_chat(response, "bot")
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)} 😔"
            self.add_message_to_chat(error_msg, "bot")
    
    def add_message_to_chat(self, message: str, sender: str):
        """Add a message to the chat display"""
        timestamp = datetime.now().strftime("%H:%M")
        
        if sender == "user":
            message_class = "user-message"
            bubble_class = "user-bubble"
            emoji = "👤"
        else:
            message_class = "bot-message" 
            bubble_class = "bot-bubble"
            emoji = "🤖"
        
        message_html = f"""
        <div class="message {message_class}">
            <div class="message-bubble {bubble_class}">
                <div>{message}</div>
                <div class="timestamp">{emoji} • {timestamp}</div>
            </div>
        </div>
        """
        
        with self.chat_output:
            display(HTML(message_html))
    
    def clear_chat(self, button):
        """Clear the chat history"""
        self.conversation_history = []
        self.chat_output.clear_output()
        self.display_welcome_message()
    
    def display(self):
        """Display the chatbot interface"""
        display(self.main_container)

# Usage Instructions and Setup
def setup_gpt_oss_chatbot():
    """Setup and launch the GPT-OSS chatbot"""
    
    print("🚀 Setting up GPT-OSS Chatbot for Google Colab...")
    print("📋 Setup Instructions:")
    print("1. Install required packages:")
    print("   !pip install torch transformers ipywidgets accelerate")
    print("2. Enable widget extensions:")
    print("   from google.colab import output")
    print("   output.enable_custom_widget_manager()")
    print("3. For better performance, enable GPU:")
    print("   Runtime → Change runtime type → Hardware accelerator: GPU")
    print("\n" + "="*60)
    
    # Create and display chatbot
    chatbot = GPTOSSChatbot()
    chatbot.display()
    
    print("\n✅ Chatbot Interface Ready! Working Features:")
    print("🎨 Beautiful chat layout with message bubbles")
    print("🤖 Multiple model options (DialoGPT, GPT-2)")
    print("⚡ Runs directly in Colab (no API needed)")
    print("🧠 Adjustable creativity and response length")
    print("🗂️  Conversation history tracking")
    print("🧹 Clear chat functionality")
    print("🌐 Optimized for English conversations")
    print("💾 GPU acceleration support")
    print("🔧 Threading issues resolved")
    print("\n💡 Pro Tips:")
    print("- Wait for '✅ Model loaded successfully!' before chatting")
    print("- DialoGPT models are optimized for conversations")
    print("- Higher temperature = more creative responses")
    print("- Use GPU for faster inference")
    
    return chatbot

# Quick start function
def quick_start():
    """Quick start with recommended settings"""
    print("🚀 Quick Start - Loading recommended model...")
    chatbot = GPTOSSChatbot("microsoft/DialoGPT-medium")
    chatbot.display()
    return chatbot

# Run the setup
if __name__ == "__main__":
    # For quick testing
    chatbot = setup_gpt_oss_chatbot()
    
    # Alternative: Quick start with default model
    # chatbot = quick_start()