# 早期的一些模型没有自带chat template，因此不能直接使用apply_chat_template方法
# 这这里记录一些不带有chat template的模型所对应的模版，方便直接调用

# from https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/models/simple/llava_hf.py line:31
VICUNA_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.index0 == 0 %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ message['content'] }} {% elif message['role'] == 'user' %}USER: {{ message['content'] }} {% else %} ASSISTANT: {{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"