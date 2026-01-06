# Chat Log - Test 01
Date: 2025-11-06
Model: gpt2-finetuned (version1.0 to version1.4)

## Conversation

### Version1.0
User: Hello,how are you?
Bot: I’m sorry, I’m not sure what you mean by “good”. I’m not sure what you mean by “good”. I’m not sure what you mean by

**所感**
- ファインチューニング後も繰り返しの出力が未だに続いている。

### Version1.1
User: Hello,how are you?
Bot: I'm happy to help.Whats's your name and addres right now?
User: I can't tell you my address
Bot: How are your friends?
User: They are fine
Bot: Okay, I’m sorry. It is your understanding that you think “They Are Fine」 may mean something like "a couple of people were very upset when they heard about the news" or whatever other such phrase might be used in a context where there was an actual tragedy and it could

**所感**
- 会話形式になっている - "I'm happy to help" など、アシスタントらしい応答
- 質問を返している - 対話的になっている
- 文脈を理解しようとしている - "your friends" など前の会話を参照
- 長すぎる応答 - 最後の応答が異常に長い
- 文脈がずれる - "They are fine"に対して過剰な解釈をしている

### Version1.2

User: Hello,how are you?
Bot: I’m sorry to interrupt. Can we talk about your past experience with drug use and the other side effects of marijuana in general?
User: I have never used drug for get high before
Bot: Well, what is your opinion?
User: About use drug? It's silly
Bot: I don’t know about the topic, but if you ask me that question in a professional setting it might be useful to try and understand what
User: You just asked me about using drug
Bot: I’m not sure what you mean by “useful'

**所感**
- 不適切なトピック - いきなり薬物について聞いてくる
- 文脈の混乱 - 自分が質問したことを忘れている
- 応答の一貫性がない
 - HH-RLHFには安全性テストのための様々な会話が含まれているため、普通のチャットボットには向いていないと思われる。若しくはフィルタリングをかけて使用。

### Version1.3

User: Hello, how are you?
Bot: Great! I'm happy to help.
User: What's your name?
Bot: I'm an assistant who can answer questions and have conversations with you.
User: Can you help me?
Bot: What can I do to assist with questions or concerns about science, technology and the environment?
User: Tell me a joke
Bot: How can I help you?
User: Tell me something interesting
Bot: I'd like to hear about things that interests you.

**所感**
- "Hello, how are you?" → "Great! I'm happy to help."
 - これは学習データに近い質問のためうまく応答できている。
- 30件のデータでは、すべてのパターンをカバーできていない
- 現在のコードは1ターンずつ独立して処理している
- 前の会話を参考にしていない


### Version1.4

User: Tell me a joke
Bot: I don't have any experience talking to humans.
User: Hello, You
Bot: What do you need help with?
User: Just have a conversation with you
Bot: Hi! Would like something different from what we talk about today...?
User: I'm curious of why all good things come to the end
Bot: Thank goodness that's easy :)
User: what do you mean
Bot: We don't know if there are other humans here.

**所感**
- 30件のデータでは根本的に不足している