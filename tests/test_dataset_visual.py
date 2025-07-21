import streamlit as st
import json
import os
from PIL import Image, UnidentifiedImageError
import base64
from io import BytesIO
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer

# --- 常量定义 ---
# 数据格式常量
DATA_FORMAT_UNKNOWN = "未知格式"
DATA_FORMAT_ALPACA_PREF = "Alpaca偏好格式 (instruction/chosen/rejected)"
DATA_FORMAT_SHAREGPT_PREF = "ShareGPT偏好格式 (conversations/chosen/rejected)"
DATA_FORMAT_ALPACA_SFT = "Alpaca SFT 格式 (instruction/output)"
DATA_FORMAT_SHAREGPT_SFT = "ShareGPT SFT 格式 (conversations)"
DATA_FORMAT_PRETRAIN = "预训练文本格式"

# --- 辅助函数 ---
def load_data(file_path):
    """根据文件扩展名加载JSON或JSONL数据"""
    data = []
    try:
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            st.error("不支持的文件格式。请选择.json或.jsonl文件。")
            return None
    except json.JSONDecodeError:
        st.error(f"文件 `{os.path.basename(file_path)}` 不是有效的JSON或JSONL格式。请检查文件内容。")
        return None
    except Exception as e:
        st.error(f"加载数据集时发生错误: {e}")
        return None
    return data

def get_image_path(image_folder, image_relative_path):
    """获取图片的完整路径"""
    return os.path.join(image_folder, image_relative_path)

def display_image(image_path, scale_factor):
    """展示图片，并处理图片加载错误，支持缩放倍数和居中"""
    try:
        if not os.path.exists(image_path):
            st.warning(f"图片文件不存在: `{image_path}`")
            return
        
        image = Image.open(image_path)
        
        # Resize image based on scale factor
        if scale_factor != 1.0:
            width, height = image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS) # Use LANCZOS for high-quality resampling
        
        # Convert PIL Image to BytesIO, then to base64 for embedding in HTML
        buffered = BytesIO()
        image.save(buffered, format="PNG") 
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Use HTML to center the image within its column
        html_code = f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_str}" style="max-width: 100%; height: auto;">
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)
        
    except UnidentifiedImageError:
        st.warning(f"无法识别的图片文件格式或文件损坏: `{image_path}`")
    except Exception as e:
        st.warning(f"加载图片时发生错误: `{e}`")

def detect_data_format(sample_data):
    """
    尝试检测数据集格式。
    优先级：ShareGPT Preference > Alpaca Preference > ShareGPT SFT > Alpaca SFT > Pretrain Text
    """
    if not sample_data or not isinstance(sample_data, dict):
        # 尝试检测预训练文本格式 (如果不是字典，可能是原始文本或只有文本的JSON)
        if isinstance(sample_data, str):
            return DATA_FORMAT_PRETRAIN
        return DATA_FORMAT_UNKNOWN

    # ShareGPT Preference 格式判断
    if "conversations" in sample_data and \
       isinstance(sample_data.get("conversations"), list) and \
       "chosen" in sample_data and isinstance(sample_data.get("chosen"), dict) and \
       "rejected" in sample_data and isinstance(sample_data.get("rejected"), dict):
        return DATA_FORMAT_SHAREGPT_PREF

    # Alpaca Preference 格式判断
    if "instruction" in sample_data and "chosen" in sample_data and "rejected" in sample_data:
        return DATA_FORMAT_ALPACA_PREF

    # ShareGPT SFT 格式判断
    if "conversations" in sample_data and isinstance(sample_data.get("conversations"), list) and \
       len(sample_data["conversations"]) > 0 and isinstance(sample_data["conversations"][0], dict) and "from" in sample_data["conversations"][0]:
        return DATA_FORMAT_SHAREGPT_SFT

    # Alpaca SFT 格式判断
    if "instruction" in sample_data and "output" in sample_data:
        return DATA_FORMAT_ALPACA_SFT
        
    # Pretrain Text 格式判断 (如果只有一个字符串键)
    if len(sample_data) == 1 and isinstance(list(sample_data.values())[0], str):
        return DATA_FORMAT_PRETRAIN

    return DATA_FORMAT_UNKNOWN

def get_default_index(options_list, key_to_find):
    """Helper to get default index for selectbox."""
    return options_list.index(key_to_find) if key_to_find in options_list else 0

def format_messages_with_tokenizer(messages, tokenizer_obj, add_generation_prompt=True):
    """使用 tokenizer 的 apply_chat_template 格式化对话消息"""
    if not tokenizer_obj:
        return None
    try:
        return tokenizer_obj.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
    except Exception as e:
        st.error(f"使用 Tokenizer 的 `apply_chat_template` 失败: {e}")
        return None

# --- 各模块功能函数 ---

def display_pretrain_data(current_item, active_format, image_key_final, final_image_folder_path, image_scale_factor, field_options):
    """显示预训练数据"""
    st.subheader("预训练文本:")

    text_key = ""
    if active_format == DATA_FORMAT_PRETRAIN:
        # 如果是单键文本，尝试获取其值
        if isinstance(current_item, dict) and len(current_item) == 1:
            text_key = list(current_item.keys())[0]
            st.code(current_item.get(text_key, ""), language="text")
        elif isinstance(current_item, str):
            text_key = "原始文本内容" # 仅用于显示
            st.code(current_item, language="text")
        else:
            st.info("此数据条目不包含可直接显示的文本内容。")
    elif active_format == DATA_FORMAT_UNKNOWN:
        # 在未知格式下，允许用户选择一个通用文本字段
        text_key = st.sidebar.selectbox("选择文本内容字段", field_options, index=0, key="pretrain_text_field")
        if text_key and text_key != "不选择此字段" and text_key in current_item:
            st.code(current_item.get(text_key, ""), language="text")
        else:
            st.info("请选择一个文本内容字段。")
    else:
        st.info("此数据格式不适用于预训练数据视图。")
    
    st.subheader("原始 JSON/JSONL 数据:")
    st.json(current_item)

    if image_key_final and image_key_final in current_item and final_image_folder_path and os.path.isdir(final_image_folder_path):
        st.subheader("图片:") 
        image_paths = []
        image_value = current_item[image_key_final]

        if isinstance(image_value, str):
            image_paths.append(get_image_path(final_image_folder_path, image_value))
        elif isinstance(image_value, list):
            for img_rel_path in image_value:
                image_paths.append(get_image_path(final_image_folder_path, img_rel_path))
        
        if image_paths:
            num_images = len(image_paths)
            st.session_state.current_image_index = st.session_state.current_image_index % num_images
            
            img_nav_col1, img_nav_col2, img_nav_col3 = st.columns([1, 2, 1])
            with img_nav_col1:
                if st.button("⬅️ 上一张", key="pretrain_prev_img", help="查看当前数据的上一张图片"):
                    st.session_state.current_image_index = (st.session_state.current_image_index - 1 + num_images) % num_images
            with img_nav_col3:
                if st.button("下一张 ➡️", key="pretrain_next_img", help="查看当前数据的下一张图片"):
                    st.session_state.current_image_index = (st.session_state.current_image_index + 1) % num_images

            st.markdown(f"<p style='text-align: center; color: grey; font-size: small;'>当前图片: {st.session_state.current_image_index + 1} / {num_images}</p>", unsafe_allow_html=True)
            
            current_image_path = image_paths[st.session_state.current_image_index]
            display_image(current_image_path, image_scale_factor)
        else:
            st.info("此条目中没有可显示的图片。")
    else:
        st.info("未配置图片文件夹或当前条目不包含图片。")

def display_sft_data(current_item, active_format, tokenizer_enabled, tokenizer_obj, image_key_final, final_image_folder_path, image_scale_factor, field_options,
                      instruction_key_alpaca_sft, input_key_alpaca_sft, output_key_alpaca_sft,
                      conversations_key_sharegpt_sft, sharegpt_role_key_sft, sharegpt_content_key_sft, 
                      sharegpt_human_role_value_sft, sharegpt_ai_role_value_sft):
    """显示SFT数据"""
    st.subheader("SFT Prompt/Instruction:")
    current_prompt_text = ""
    current_output_text = ""
    
    if active_format == DATA_FORMAT_ALPACA_SFT:
        instruction_content = current_item.get(instruction_key_alpaca_sft, "")
        input_content = current_item.get(input_key_alpaca_sft, "")
        output_content = current_item.get(output_key_alpaca_sft, "")

        messages_for_prompt_template = []
        combined_prompt_raw = ""
        if instruction_content:
            combined_prompt_raw += instruction_content
        if input_content:
            if combined_prompt_raw:
                combined_prompt_raw += f"\n{input_content}"
            else:
                combined_prompt_raw = input_content
        
        if combined_prompt_raw:
            messages_for_prompt_template.append({"role": "user", "content": combined_prompt_raw})
        
        if tokenizer_enabled and messages_for_prompt_template:
            formatted_prompt = format_messages_with_tokenizer(messages_for_prompt_template, tokenizer_obj, add_generation_prompt=True)
            current_prompt_text = formatted_prompt if formatted_prompt is not None else combined_prompt_raw
        else:
            formatted_parts = []
            if instruction_content:
                formatted_parts.append(f"**{instruction_key_alpaca_sft}**: {instruction_content}")
            if input_content:
                formatted_parts.append(f"**{input_key_alpaca_sft}**: {input_content}")
            current_prompt_text = "\n\n".join(formatted_parts)
        
        current_output_text = output_content if output_content else ""

    elif active_format == DATA_FORMAT_SHAREGPT_SFT:
        if conversations_key_sharegpt_sft and conversations_key_sharegpt_sft in current_item and \
           isinstance(current_item[conversations_key_sharegpt_sft], list) and \
           sharegpt_role_key_sft and sharegpt_content_key_sft and \
           sharegpt_human_role_value_sft and sharegpt_ai_role_value_sft:

            conversation_list = current_item[conversations_key_sharegpt_sft]
            
            # 提取 Prompt (所有消息除了最后一个 AI 回复)
            prompt_messages_for_template = []
            simple_prompt_messages = []
            
            # 提取 Output (最后一个 AI 回复)
            output_message_content = ""

            if conversation_list:
                for i, msg in enumerate(conversation_list):
                    if not isinstance(msg, dict):
                        st.warning(f"跳过非字典类型的对话消息: {msg}")
                        continue
                    
                    msg_role = msg.get(sharegpt_role_key_sft)
                    msg_content = msg.get(sharegpt_content_key_sft, "")

                    standard_role = None
                    if msg_role == sharegpt_human_role_value_sft:
                        standard_role = "user"
                    elif msg_role == sharegpt_ai_role_value_sft:
                        standard_role = "assistant"

                    if standard_role:
                        if i == len(conversation_list) - 1 and standard_role == "assistant":
                            output_message_content = msg_content
                        else:
                            prompt_messages_for_template.append({"role": standard_role, "content": msg_content})
                            simple_prompt_messages.append(f"**{msg_role}**: {msg_content}")
                    else:
                        st.warning(f"SFT: 当前数据条目中的对话消息角色未识别: '{msg_role}'。已跳过此消息。")
            
            if tokenizer_enabled and prompt_messages_for_template:
                formatted_prompt = format_messages_with_tokenizer(prompt_messages_for_template, tokenizer_obj, add_generation_prompt=True)
                current_prompt_text = formatted_prompt if formatted_prompt is not None else "\n\n".join(simple_prompt_messages)
            else:
                current_prompt_text = "\n\n".join(simple_prompt_messages)
            
            current_output_text = output_message_content
        else:
            st.warning("ShareGPT SFT 数据配置不完整或数据结构异常。无法正确显示。")

    elif active_format == DATA_FORMAT_UNKNOWN:
        # 在未知格式下，将用户选择的字段直接显示
        st.info("当前为未知格式，请手动选择Prompt和Output字段进行显示。")
        instruction_key_unknown = st.sidebar.selectbox("选择Prompt/Instruction字段 (未知格式)", field_options, index=0, key="sft_unk_instr")
        output_key_unknown = st.sidebar.selectbox("选择Output字段 (未知格式)", field_options, index=0, key="sft_unk_output")

        if instruction_key_unknown and instruction_key_unknown != "不选择此字段" and instruction_key_unknown in current_item:
            current_prompt_text = current_item[instruction_key_unknown]
        if output_key_unknown and output_key_unknown != "不选择此字段" and output_key_unknown in current_item:
            current_output_text = current_item[output_key_unknown]
    else:
        st.info("此数据格式不适用于SFT数据视图。")


    if current_prompt_text:
        st.code(current_prompt_text, language="text")
    else:
        st.info("Prompt/Instruction 内容为空或未找到。请检查字段映射。") 

    st.subheader("SFT Output:") 
    if current_output_text:
        st.code(current_output_text, language="text")
    else:
        st.info("Output 内容为空或未找到。请检查字段映射。") 
    
    st.subheader("原始 JSON/JSONL 数据:")
    st.json(current_item)

    if image_key_final and image_key_final in current_item and final_image_folder_path and os.path.isdir(final_image_folder_path):
        st.subheader("图片:") 
        image_paths = []
        image_value = current_item[image_key_final]

        if isinstance(image_value, str):
            image_paths.append(get_image_path(final_image_folder_path, image_value))
        elif isinstance(image_value, list):
            for img_rel_path in image_value:
                image_paths.append(get_image_path(final_image_folder_path, img_rel_path))
        
        if image_paths:
            num_images = len(image_paths)
            st.session_state.current_image_index = st.session_state.current_image_index % num_images
            
            img_nav_col1, img_nav_col2, img_nav_col3 = st.columns([1, 2, 1])
            with img_nav_col1:
                if st.button("⬅️ 上一张", key="sft_prev_img", help="查看当前数据的上一张图片"):
                    st.session_state.current_image_index = (st.session_state.current_image_index - 1 + num_images) % num_images
            with img_nav_col3:
                if st.button("下一张 ➡️", key="sft_next_img", help="查看当前数据的下一张图片"):
                    st.session_state.current_image_index = (st.session_state.current_image_index + 1) % num_images

            st.markdown(f"<p style='text-align: center; color: grey; font-size: small;'>当前图片: {st.session_state.current_image_index + 1} / {num_images}</p>", unsafe_allow_html=True)
            
            current_image_path = image_paths[st.session_state.current_image_index]
            display_image(current_image_path, image_scale_factor)
        else:
            st.info("此条目中没有可显示的图片。")
    else:
        st.info("未配置图片文件夹或当前条目不包含图片。")


def display_preference_data(current_item, active_format, tokenizer_enabled, tokenizer_obj, image_key_final, final_image_folder_path, image_scale_factor, field_options,
                            instruction_key_alpaca_pref, input_key_alpaca_pref, chosen_key_alpaca_pref, rejected_key_alpaca_pref,
                            conversations_key_sharegpt_pref, chosen_key_sharegpt_pref, rejected_key_sharegpt_pref,
                            sharegpt_role_key_pref, sharegpt_content_key_pref, sharegpt_human_role_value_pref, sharegpt_ai_role_value_pref):
    """显示偏好数据"""
    st.subheader("Prompt:") 
    current_prompt_text = ""
    current_chosen_text = ""
    current_rejected_text = ""

    if active_format == DATA_FORMAT_ALPACA_PREF:
        instruction_content = current_item.get(instruction_key_alpaca_pref, "") 
        input_content = current_item.get(input_key_alpaca_pref, "")
        chosen_content = current_item.get(chosen_key_alpaca_pref, "") 
        rejected_content = current_item.get(rejected_key_alpaca_pref, "") 

        messages_for_prompt_template = []
        combined_prompt_raw = ""
        if instruction_content:
            combined_prompt_raw += instruction_content
        if input_content:
            if combined_prompt_raw:
                combined_prompt_raw += f"\n{input_content}"
            else:
                combined_prompt_raw = input_content
        
        if combined_prompt_raw:
            messages_for_prompt_template.append({"role": "user", "content": combined_prompt_raw})
        
        if tokenizer_enabled and messages_for_prompt_template:
            formatted_prompt = format_messages_with_tokenizer(messages_for_prompt_template, tokenizer_obj, add_generation_prompt=True)
            current_prompt_text = formatted_prompt if formatted_prompt is not None else combined_prompt_raw # 回退到原始
        else:
            formatted_parts = []
            if instruction_content:
                formatted_parts.append(f"**{instruction_key_alpaca_pref}**: {instruction_content}")
            if input_content:
                formatted_parts.append(f"**{input_key_alpaca_pref}**: {input_content}")
            current_prompt_text = "\n\n".join(formatted_parts)
        
        current_chosen_text = chosen_content if chosen_content else ""
        current_rejected_text = rejected_content if rejected_content else ""

    elif active_format == DATA_FORMAT_SHAREGPT_PREF:
        if conversations_key_sharegpt_pref and conversations_key_sharegpt_pref in current_item and \
           isinstance(current_item[conversations_key_sharegpt_pref], list) and \
           chosen_key_sharegpt_pref and chosen_key_sharegpt_pref in current_item and \
           rejected_key_sharegpt_pref and rejected_key_sharegpt_pref in current_item and \
           sharegpt_role_key_pref and sharegpt_content_key_pref and sharegpt_human_role_value_pref and sharegpt_ai_role_value_pref: 
            
            conversation_base = current_item[conversations_key_sharegpt_pref]
            chosen_data = current_item[chosen_key_sharegpt_pref] # 这是一个字典
            rejected_data = current_item[rejected_key_sharegpt_pref] # 这是一个字典

            prompt_messages_for_template = []
            simple_prompt_messages = []

            for msg in conversation_base:
                if not isinstance(msg, dict):
                    st.warning(f"跳过非字典类型的对话消息: {msg}")
                    continue
                
                msg_role = msg.get(sharegpt_role_key_pref)
                msg_content = msg.get(sharegpt_content_key_pref, "")

                if msg_role == sharegpt_human_role_value_pref:
                    prompt_messages_for_template.append({"role": "user", "content": msg_content})
                    simple_prompt_messages.append(f"**{sharegpt_human_role_value_pref}**: {msg_content}")
                elif msg_role == sharegpt_ai_role_value_pref:
                    prompt_messages_for_template.append({"role": "assistant", "content": msg_content})
                    simple_prompt_messages.append(f"**{sharegpt_ai_role_value_pref}**: {msg_content}")
                else:
                    st.warning(f"Preference: 当前数据条目中的对话消息角色未识别: '{msg_role}'。已跳过此消息。")

            if tokenizer_enabled and prompt_messages_for_template:
                # Prompt 使用 add_generation_prompt=True
                formatted_prompt = format_messages_with_tokenizer(prompt_messages_for_template, tokenizer_obj, add_generation_prompt=True)
                current_prompt_text = formatted_prompt if formatted_prompt is not None else "\n\n".join(simple_prompt_messages)
            else:
                current_prompt_text = "\n\n".join(simple_prompt_messages)

            # 提取 Chosen 和 Rejected (直接是字典)
            if isinstance(chosen_data, dict) and chosen_data.get(sharegpt_role_key_pref) == sharegpt_ai_role_value_pref:
                current_chosen_text = chosen_data.get(sharegpt_content_key_pref, "")
            else:
                st.warning("Chosen 数据不是预期的字典格式或角色不匹配。")
                current_chosen_text = ""

            if isinstance(rejected_data, dict) and rejected_data.get(sharegpt_role_key_pref) == sharegpt_ai_role_value_pref:
                current_rejected_text = rejected_data.get(sharegpt_content_key_pref, "")
            else:
                st.warning("Rejected 数据不是预期的字典格式或角色不匹配。")
                current_rejected_text = ""
        else:
            st.warning("ShareGPT偏好数据配置不完整或数据结构异常。无法正确显示。")
            pass # 如果配置不完整，则保持为空

    elif active_format == DATA_FORMAT_UNKNOWN: 
        st.info("当前为未知格式，请手动选择Prompt、Chosen和Rejected字段进行显示。")
        instruction_key_unknown = st.sidebar.selectbox("选择Prompt/Instruction字段 (未知格式)", field_options, index=0, key="pref_unk_instr_gen")
        chosen_key_unknown = st.sidebar.selectbox("选择Chosen文本字段 (未知格式)", field_options, index=0, key="pref_unk_chosen_gen")
        rejected_key_unknown = st.sidebar.selectbox("选择Rejected文本字段 (未知格式)", field_options, index=0, key="pref_unk_rejected_gen")

        if instruction_key_unknown and instruction_key_unknown != "不选择此字段" and instruction_key_unknown in current_item:
            current_prompt_text = current_item[instruction_key_unknown]
        if chosen_key_unknown and chosen_key_unknown != "不选择此字段" and chosen_key_unknown in current_item:
            current_chosen_text = current_item[chosen_key_unknown]
        if rejected_key_unknown and rejected_key_unknown != "不选择此字段" and rejected_key_unknown in current_item:
            current_rejected_text = current_item[rejected_key_unknown]
    else:
        st.info("此数据格式不适用于偏好数据视图。")


    if current_prompt_text:
        st.code(current_prompt_text, language="text")
    else:
        st.info("Prompt内容为空或未找到。请检查字段映射。") 

    st.subheader("Chosen Response:") 
    if current_chosen_text:
        st.code(current_chosen_text, language="text")
    else:
        st.info("Chosen Response内容为空或未找到。请检查字段映射。") 
    
    st.subheader("Rejected Response:") 
    if current_rejected_text:
        st.code(current_rejected_text, language="text")
    else:
        st.info("Rejected Response内容为空或未找到。请检查字段映射。") 

    st.subheader("原始 JSON/JSONL 数据:")
    st.json(current_item)

    if image_key_final and image_key_final in current_item and final_image_folder_path and os.path.isdir(final_image_folder_path):
        st.subheader("图片:") 
        image_paths = []
        image_value = current_item[image_key_final]

        if isinstance(image_value, str):
            image_paths.append(get_image_path(final_image_folder_path, image_value))
        elif isinstance(image_value, list):
            for img_rel_path in image_value:
                image_paths.append(get_image_path(final_image_folder_path, img_rel_path))
        
        if image_paths:
            num_images = len(image_paths)
            st.session_state.current_image_index = st.session_state.current_image_index % num_images
            
            img_nav_col1, img_nav_col2, img_nav_col3 = st.columns([1, 2, 1])
            with img_nav_col1:
                if st.button("⬅️ 上一张", key="pref_prev_img", help="查看当前数据的上一张图片"):
                    st.session_state.current_image_index = (st.session_state.current_image_index - 1 + num_images) % num_images
            with img_nav_col3:
                if st.button("下一张 ➡️", key="pref_next_img", help="查看当前数据的下一张图片"):
                    st.session_state.current_image_index = (st.session_state.current_image_index + 1) % num_images

            st.markdown(f"<p style='text-align: center; color: grey; font-size: small;'>当前图片: {st.session_state.current_image_index + 1} / {num_images}</p>", unsafe_allow_html=True)
            
            current_image_path = image_paths[st.session_state.current_image_index]
            display_image(current_image_path, image_scale_factor)
        else:
            st.info("此条目中没有可显示的图片。")
    else:
        st.info("未配置图片文件夹或当前条目不包含图片。")


# --- 主应用逻辑 ---
def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='font-size: 2.2em;'>数据集可视化工具</h1>", unsafe_allow_html=True)

    # --- 会话状态初始化 ---
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'tokenizer_enabled' not in st.session_state:
        st.session_state.tokenizer_enabled = False
    
    # Track paths to avoid re-loading on every rerun if user hasn't changed them
    if 'last_data_file_path' not in st.session_state:
        st.session_state.last_data_file_path = None
    if 'last_image_folder_path' not in st.session_state:
        st.session_state.last_image_folder_path = None
    if 'last_tokenizer_path' not in st.session_state:
        st.session_state.last_tokenizer_path = None

    # ShareGPT 特定键的会话状态
    if 'sharegpt_role_key_pref' not in st.session_state: st.session_state.sharegpt_role_key_pref = None
    if 'sharegpt_content_key_pref' not in st.session_state: st.session_state.sharegpt_content_key_pref = None
    if 'sharegpt_human_role_value_pref' not in st.session_state: st.session_state.sharegpt_human_role_value_pref = None
    if 'sharegpt_ai_role_value_pref' not in st.session_state: st.session_state.sharegpt_ai_role_value_pref = None
    if 'last_sharegpt_pref_config_keys' not in st.session_state: st.session_state.last_sharegpt_pref_config_keys = None
    if 'last_role_key_sidebar_sharegpt_inner_pref' not in st.session_state: st.session_state.last_role_key_sidebar_sharegpt_inner_pref = None

    if 'sharegpt_role_key_sft' not in st.session_state: st.session_state.sharegpt_role_key_sft = None
    if 'sharegpt_content_key_sft' not in st.session_state: st.session_state.sharegpt_content_key_sft = None
    if 'sharegpt_human_role_value_sft' not in st.session_state: st.session_state.sharegpt_human_role_value_sft = None
    if 'sharegpt_ai_role_value_sft' not in st.session_state: st.session_state.sharegpt_ai_role_value_sft = None
    if 'last_sharegpt_sft_config_keys' not in st.session_state: st.session_state.last_sharegpt_sft_config_keys = None
    if 'last_role_key_sidebar_sharegpt_inner_sft' not in st.session_state: st.session_state.last_role_key_sidebar_sharegpt_inner_sft = None

    # 从命令行参数获取路径 (如果提供)
    temp_data_file_path = None
    temp_image_folder_path = None
    temp_tokenizer_path = None 
    import sys
    args = sys.argv[1:] 
    if len(args) > 0: temp_data_file_path = args[0]
    if len(args) > 1: temp_image_folder_path = args[1]
    if len(args) > 2: temp_tokenizer_path = args[2]

    # --- 侧边栏：通用文件与路径配置 ---
    st.sidebar.header("文件与路径配置")
    st.sidebar.subheader("数据集")

    # 数据集文件选择 (命令行或上传)
    if not temp_data_file_path:
        uploaded_file = st.sidebar.file_uploader("选择您的JSON或JSONL数据集", type=["json", "jsonl"])
        if uploaded_file is not None:
            temp_upload_dir = "uploaded_datasets_temp_unified" # 统一的临时目录
            os.makedirs(temp_upload_dir, exist_ok=True)
            temp_data_file_path = os.path.join(temp_upload_dir, uploaded_file.name)
            with open(temp_data_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"已上传数据集: **`{uploaded_file.name}`**")
            st.sidebar.warning("⚠️ **请注意：** 您通过页面上传文件，无法自动获取数据集的父级目录作为图片文件夹和 Tokenizer 路径。**如果需要显示图片或使用 Tokenizer，请务必手动填写路径。**")
            # Force rerun to ensure data is loaded for format detection
            st.rerun() # Use st.rerun() to immediately re-evaluate the script with the new file

    if temp_data_file_path:
        st.sidebar.write(f"**数据集路径:** `{temp_data_file_path}`")
    else:
        st.sidebar.info("未选择数据集。")

    # 图片文件夹路径输入 (命令行或手动输入)
    st.sidebar.markdown("---")
    st.sidebar.subheader("图片文件夹 (可选)")
    if temp_data_file_path and not temp_image_folder_path and not st.session_state.last_image_folder_path:
        temp_image_folder_path = os.path.dirname(temp_data_file_path)
        st.sidebar.info(f"图片文件夹路径默认为数据集的父级目录: `{temp_image_folder_path}`")
    image_folder_input = st.sidebar.text_input("请输入图片文件夹路径", value=temp_image_folder_path if temp_image_folder_path else "")
    final_image_folder_path = None
    if image_folder_input:
        final_image_folder_path = image_folder_input
        if not os.path.isdir(final_image_folder_path):
            st.sidebar.warning(f"图片文件夹路径 `{final_image_folder_path}` 不是一个有效目录。")
    else:
        st.sidebar.info("未指定图片文件夹路径，将不展示图片。")
    
    # Tokenizer 路径输入 (命令行或手动输入)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tokenizer 目录 (可选)")
    tokenizer_path_input = st.sidebar.text_input("请输入Tokenizer目录路径", value=temp_tokenizer_path if temp_tokenizer_path else "")
    
    if tokenizer_path_input:
        # Only try to load if path changed or tokenizer not loaded
        if st.session_state.tokenizer is None or st.session_state.last_tokenizer_path != tokenizer_path_input:
            st.sidebar.info(f"正在尝试加载 Tokenizer 来自: `{tokenizer_path_input}`...")
            try:
                if not os.path.isdir(tokenizer_path_input):
                    st.sidebar.error(f"Tokenizer 路径 `{tokenizer_path_input}` 不是一个有效的目录。")
                    st.session_state.tokenizer = None
                else:
                    st.session_state.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_input)
                    st.session_state.last_tokenizer_path = tokenizer_path_input 
                    st.sidebar.success("Tokenizer 加载成功！")
                    st.session_state.tokenizer_enabled = True # 成功加载，默认启用
            except Exception as e:
                st.session_state.tokenizer = None
                st.session_state.tokenizer_enabled = False # 加载失败，禁用
                st.sidebar.error(f"加载 Tokenizer 失败: {e}")
                st.sidebar.warning("请确保输入的路径是一个完整的 Hugging Face Tokenizer 目录，且包含所有必要文件。")
    else: 
        st.session_state.tokenizer = None 
        st.session_state.last_tokenizer_path = None 
        st.session_state.tokenizer_enabled = False # 没有路径，禁用
        st.sidebar.info("未指定 Tokenizer 路径。")
    
    if st.session_state.tokenizer:
        st.sidebar.write(f"**Tokenizer 已加载:** `{st.session_state.last_tokenizer_path}`")
        st.session_state.tokenizer_enabled = st.sidebar.checkbox(
            "启用 Tokenizer 格式化 Prompt", 
            value=st.session_state.tokenizer_enabled,
            help="勾选此项将使用加载的 Tokenizer 对 Prompt 进行格式化 (如 `apply_chat_template`)。取消勾选将显示原始字段内容。"
        )
    else:
        st.sidebar.write(f"**Tokenizer 状态:** 未加载")
        st.sidebar.checkbox(
            "启用 Tokenizer 格式化 Prompt", 
            value=False, 
            disabled=True, 
            help="未加载 Tokenizer，此功能不可用。"
        )

    # --- 通用的图片缩放滑块 (放在这里确保在任何视图下都可用) ---
    st.sidebar.markdown("---")
    st.sidebar.header("图片设置")
    image_scale_factor = st.sidebar.slider(
        "选择图片缩放倍数",
        min_value=0.1, max_value=2.0, value=1.0, step=0.1,
        help="调整图片显示大小。1.0为原始大小，大于1.0放大，小于1.0缩小。",
        key="universal_image_scale" # 添加一个唯一的key
    )
    
    # --- 视图选择 ---
    st.sidebar.markdown("---")
    view_mode = st.sidebar.radio(
        "选择数据视图模式:",
        ["偏好数据", "SFT 数据", "预训练数据"],
        index=0 # Default to Preference Data
    )
    st.sidebar.markdown("---")

    data_file_path = temp_data_file_path 
    
    if data_file_path:
        data = load_data(data_file_path)
        if data is None:
            return

        if not data:
            st.warning("数据集中没有数据。")
            return

        # 自动判断数据格式
        detected_format = detect_data_format(data[0])
        st.sidebar.header(f"当前文件检测格式: **`{detected_format}`**")
        
        # 定义所有可能的字段，用于通用选择
        all_keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []
        field_options = ["不选择此字段"] + all_keys

        # 通用的图片字段选择（无论哪个模式都可能用到图片）
        # 放在这里避免重复定义，并且确保在任何视图下都能被配置
        image_key = st.sidebar.selectbox("选择Image字段 (可选)", field_options, index=get_default_index(field_options, "image"), key="universal_image_key_selector")


        # 根据不同的视图模式显示不同的字段映射
        st.sidebar.subheader("字段映射")

        # 初始化所有模式的字段变量
        # Preference Data (Alpaca)
        instruction_key_alpaca_pref = ""
        input_key_alpaca_pref = ""
        chosen_key_alpaca_pref = ""
        rejected_key_alpaca_pref = ""
        # Preference Data (ShareGPT)
        conversations_key_sharegpt_pref = ""
        chosen_key_sharegpt_pref = ""
        rejected_key_sharegpt_pref = ""

        # SFT Data (Alpaca)
        instruction_key_alpaca_sft = ""
        input_key_alpaca_sft = ""
        output_key_alpaca_sft = ""
        # SFT Data (ShareGPT)
        conversations_key_sharegpt_sft = ""
        

        # 仅在需要时显示和配置字段映射
        if view_mode == "偏好数据":
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 偏好数据字段配置")
            selected_format = st.sidebar.selectbox(
                "手动选择偏好数据格式 (若自动检测不准确)",
                [DATA_FORMAT_UNKNOWN, DATA_FORMAT_ALPACA_PREF, DATA_FORMAT_SHAREGPT_PREF],
                index=get_default_index([DATA_FORMAT_UNKNOWN, DATA_FORMAT_ALPACA_PREF, DATA_FORMAT_SHAREGPT_PREF], detected_format),
                key="pref_format_select"
            )
            active_format = selected_format # Override detected for specific view

            if active_format == DATA_FORMAT_ALPACA_PREF:
                st.sidebar.markdown("**Alpaca偏好格式字段**")
                instruction_key_alpaca_pref = st.sidebar.selectbox("选择Instruction字段 (如: instruction)", field_options, index=get_default_index(field_options, "instruction"), key="pref_alp_instr")
                input_key_alpaca_pref = st.sidebar.selectbox("选择Input字段 (可选, 如: input)", field_options, index=get_default_index(field_options, "input"), key="pref_alp_input")
                chosen_key_alpaca_pref = st.sidebar.selectbox("选择Chosen字段 (如: chosen)", field_options, index=get_default_index(field_options, "chosen"), key="pref_alp_chosen")
                rejected_key_alpaca_pref = st.sidebar.selectbox("选择Rejected字段 (如: rejected)", field_options, index=get_default_index(field_options, "rejected"), key="pref_alp_rejected")
            
            elif active_format == DATA_FORMAT_SHAREGPT_PREF:
                st.sidebar.markdown("**ShareGPT偏好格式字段**")
                conversations_key_sharegpt_pref = st.sidebar.selectbox("1. 选择对话列表字段 (通常是: conversations)", field_options, index=get_default_index(field_options, "conversations"), key="pref_sgpt_conv")
                chosen_key_sharegpt_pref = st.sidebar.selectbox("2. 选择Chosen对话字段 (如: chosen)", field_options, index=get_default_index(field_options, "chosen"), key="pref_sgpt_chosen")
                rejected_key_sharegpt_pref = st.sidebar.selectbox("3. 选择Rejected对话字段 (如: rejected)", field_options, index=get_default_index(field_options, "rejected"), key="pref_sgpt_rejected")

                current_sharegpt_pref_config_keys = (conversations_key_sharegpt_pref, chosen_key_sharegpt_pref, rejected_key_sharegpt_pref)
                if st.session_state.get('last_sharegpt_pref_config_keys') != current_sharegpt_pref_config_keys:
                    st.session_state.sharegpt_role_key_pref = None
                    st.session_state.sharegpt_content_key_pref = None
                    st.session_state.sharegpt_human_role_value_pref = None
                    st.session_state.sharegpt_ai_role_value_pref = None
                    st.session_state.last_sharegpt_pref_config_keys = current_sharegpt_pref_config_keys

                if (conversations_key_sharegpt_pref and conversations_key_sharegpt_pref != "不选择此字段") or \
                   (chosen_key_sharegpt_pref and chosen_key_sharegpt_pref != "不选择此字段") or \
                   (rejected_key_sharegpt_pref and rejected_key_sharegpt_pref != "不选择此字段"):
                    all_inner_message_keys = set()
                    all_roles_in_data = set()
                    sample_current_item = data[0] 

                    if conversations_key_sharegpt_pref and conversations_key_sharegpt_pref != "不选择此字段" \
                       and conversations_key_sharegpt_pref in sample_current_item \
                       and isinstance(sample_current_item[conversations_key_sharegpt_pref], list):
                        for msg in sample_current_item[conversations_key_sharegpt_pref]:
                            if isinstance(msg, dict):
                                all_inner_message_keys.update(msg.keys())
                                if "from" in msg: all_roles_in_data.add(msg["from"])
                    
                    for dataset_key in [chosen_key_sharegpt_pref, rejected_key_sharegpt_pref]:
                        if dataset_key and dataset_key != "不选择此字段" and dataset_key in sample_current_item \
                           and isinstance(sample_current_item[dataset_key], dict):
                            msg = sample_current_item[dataset_key]
                            all_inner_message_keys.update(msg.keys())
                            if "from" in msg: all_roles_in_data.add(msg["from"])

                    if all_inner_message_keys:
                        role_key_options = ["不选择此字段"] + list(all_inner_message_keys)
                        role_key_sidebar_sharegpt_inner_pref = st.sidebar.selectbox("4. 选择角色标识键 (如: from)", role_key_options, index=get_default_index(role_key_options, "from"), key="pref_sgpt_role_key")
                        
                        if st.session_state.get('last_role_key_sidebar_sharegpt_inner_pref') != role_key_sidebar_sharegpt_inner_pref:
                            st.session_state.sharegpt_content_key_pref = None 
                            st.session_state.sharegpt_human_role_value_pref = None
                            st.session_state.sharegpt_ai_role_value_pref = None
                            st.session_state.last_role_key_sidebar_sharegpt_inner_pref = role_key_sidebar_sharegpt_inner_pref

                        if role_key_sidebar_sharegpt_inner_pref and role_key_sidebar_sharegpt_inner_pref != "不选择此字段":
                            other_keys = [k for k in all_inner_message_keys if k != role_key_sidebar_sharegpt_inner_pref]
                            auto_content_key = None
                            if "value" in other_keys: auto_content_key = "value"
                            elif "content" in other_keys: auto_content_key = "content"
                            elif len(other_keys) > 0: auto_content_key = other_keys[0] 

                            if auto_content_key:
                                st.session_state.sharegpt_role_key_pref = role_key_sidebar_sharegpt_inner_pref
                                st.session_state.sharegpt_content_key_pref = auto_content_key
                                st.sidebar.info(f"对话内容键自动识别为: **`{st.session_state.sharegpt_content_key_pref}`**")
                            else:
                                st.sidebar.error("无法识别内容键。请检查消息结构。")
                                st.session_state.sharegpt_role_key_pref = None
                                st.session_state.sharegpt_content_key_pref = None
                            
                            if st.session_state.sharegpt_role_key_pref and st.session_state.sharegpt_content_key_pref:
                                role_value_options = ["不选择此字段"] + [val for val in all_roles_in_data if val is not None] 
                                human_role_value_sidebar_sharegpt_inner_pref = st.sidebar.selectbox(f"5. 选择用户角色标识 ({st.session_state.sharegpt_role_key_pref}字段的值, 如: human)", role_value_options, index=get_default_index(role_value_options, "human"), key="pref_sgpt_human_role")
                                ai_role_value_sidebar_sharegpt_inner_pref = st.sidebar.selectbox(f"6. 选择AI角色标识 ({st.session_state.sharegpt_role_key_pref}字段的值, 如: gpt 或 assistant)", role_value_options, index=get_default_index(role_value_options, "gpt") if "gpt" in role_value_options else get_default_index(role_value_options, "assistant"), key="pref_sgpt_ai_role")
                                st.session_state.sharegpt_human_role_value_pref = human_role_value_sidebar_sharegpt_inner_pref
                                st.session_state.sharegpt_ai_role_value_pref = ai_role_value_sidebar_sharegpt_inner_pref
                            else:
                                st.sidebar.warning("无法识别角色键或内容键，无法继续配置角色值。请检查ShareGPT消息结构。")
                        else:
                            st.sidebar.warning("请选择 '角色标识键' 以继续配置角色值。")
                    else:
                        st.sidebar.warning("无法从对话或偏好列表中提取内部消息键。请检查数据集结构。")
                else:
                    st.sidebar.warning("请至少选择 '对话列表字段'、'Chosen对话字段' 或 'Rejected对话字段' 中的一个，以开始配置 ShareGPT 格式。")

            elif active_format == DATA_FORMAT_UNKNOWN: 
                st.sidebar.markdown("**通用/未知格式字段**")
                instruction_key_alpaca_pref = st.sidebar.selectbox("选择Prompt/Instruction字段", field_options, index=0, key="pref_unk_instr_gen")
                chosen_key_alpaca_pref = st.sidebar.selectbox("选择Chosen文本字段", field_options, index=0, key="pref_unk_chosen_gen")
                rejected_key_alpaca_pref = st.sidebar.selectbox("选择Rejected文本字段", field_options, index=0, key="pref_unk_rejected_gen")
                st.sidebar.info("未知格式下，字段解释可能不准确。请尝试手动选择对应字段。")

            # Finalize preference keys
            instruction_key_final = instruction_key_alpaca_pref if instruction_key_alpaca_pref != "不选择此字段" else ""
            input_key_final = input_key_alpaca_pref if input_key_alpaca_pref != "不选择此字段" else ""
            chosen_key_final = chosen_key_alpaca_pref if chosen_key_alpaca_pref != "不选择此字段" else ""
            rejected_key_final = rejected_key_alpaca_pref if rejected_key_alpaca_pref != "不选择此字段" else ""
            
            conversations_key_final = conversations_key_sharegpt_pref if conversations_key_sharegpt_pref != "不选择此字段" else ""
            chosen_key_sharegpt_final = chosen_key_sharegpt_pref if chosen_key_sharegpt_pref != "不选择此字段" else ""
            rejected_key_sharegpt_final = rejected_key_sharegpt_pref if rejected_key_sharegpt_pref != "不选择此字段" else ""

            # Use st.session_state for ShareGPT internal keys
            role_key_final_pref = st.session_state.sharegpt_role_key_pref
            content_key_final_pref = st.session_state.sharegpt_content_key_pref
            human_role_value_final_pref = st.session_state.sharegpt_human_role_value_pref
            ai_role_value_final_pref = st.session_state.sharegpt_ai_role_value_pref

            # Display preference data
            is_alpaca_pref_ready = active_format == DATA_FORMAT_ALPACA_PREF and instruction_key_final and chosen_key_final and rejected_key_final
            is_sharegpt_pref_ready = active_format == DATA_FORMAT_SHAREGPT_PREF and \
                                     conversations_key_final and chosen_key_sharegpt_final and rejected_key_sharegpt_final and \
                                     role_key_final_pref and content_key_final_pref and human_role_value_final_pref and ai_role_value_final_pref
            is_unknown_pref_ready = active_format == DATA_FORMAT_UNKNOWN and (instruction_key_final or chosen_key_final or rejected_key_final) # For unknown, any field is "ready"
            
            is_ready_to_display = is_alpaca_pref_ready or is_sharegpt_pref_ready or is_unknown_pref_ready

            if not is_ready_to_display:
                st.info("请完成偏好数据字段配置。")
            else:
                display_preference_data(
                    current_item=data[st.session_state.current_index],
                    active_format=active_format,
                    tokenizer_enabled=st.session_state.tokenizer_enabled,
                    tokenizer_obj=st.session_state.tokenizer,
                    image_key_final=image_key, # Image key is universal
                    final_image_folder_path=final_image_folder_path,
                    image_scale_factor=image_scale_factor,
                    field_options=field_options,
                    instruction_key_alpaca_pref=instruction_key_final,
                    input_key_alpaca_pref=input_key_final,
                    chosen_key_alpaca_pref=chosen_key_final,
                    rejected_key_alpaca_pref=rejected_key_final,
                    conversations_key_sharegpt_pref=conversations_key_final,
                    chosen_key_sharegpt_pref=chosen_key_sharegpt_final,
                    rejected_key_sharegpt_pref=rejected_key_sharegpt_final,
                    sharegpt_role_key_pref=role_key_final_pref,
                    sharegpt_content_key_pref=content_key_final_pref,
                    sharegpt_human_role_value_pref=human_role_value_final_pref,
                    sharegpt_ai_role_value_pref=ai_role_value_final_pref
                )

        elif view_mode == "SFT 数据":
            st.sidebar.markdown("---")
            st.sidebar.markdown("### SFT 数据字段配置")
            selected_format = st.sidebar.selectbox(
                "手动选择 SFT 数据格式 (若自动检测不准确)",
                [DATA_FORMAT_UNKNOWN, DATA_FORMAT_ALPACA_SFT, DATA_FORMAT_SHAREGPT_SFT],
                index=get_default_index([DATA_FORMAT_UNKNOWN, DATA_FORMAT_ALPACA_SFT, DATA_FORMAT_SHAREGPT_SFT], detected_format),
                key="sft_format_select"
            )
            active_format = selected_format

            if active_format == DATA_FORMAT_ALPACA_SFT:
                st.sidebar.markdown("**Alpaca SFT 格式字段**")
                instruction_key_alpaca_sft = st.sidebar.selectbox("选择Instruction字段 (如: instruction)", field_options, index=get_default_index(field_options, "instruction"), key="sft_alp_instr")
                input_key_alpaca_sft = st.sidebar.selectbox("选择Input字段 (可选, 如: input)", field_options, index=get_default_index(field_options, "input"), key="sft_alp_input")
                output_key_alpaca_sft = st.sidebar.selectbox("选择Output字段 (如: output)", field_options, index=get_default_index(field_options, "output"), key="sft_alp_output")

            elif active_format == DATA_FORMAT_SHAREGPT_SFT:
                st.sidebar.markdown("**ShareGPT SFT 格式字段**")
                conversations_key_sharegpt_sft = st.sidebar.selectbox("1. 选择对话列表字段 (通常是: conversations)", field_options, index=get_default_index(field_options, "conversations"), key="sft_sgpt_conv")

                current_sharegpt_sft_config_keys = (conversations_key_sharegpt_sft,)
                if st.session_state.get('last_sharegpt_sft_config_keys') != current_sharegpt_sft_config_keys:
                    st.session_state.sharegpt_role_key_sft = None
                    st.session_state.sharegpt_content_key_sft = None
                    st.session_state.sharegpt_human_role_value_sft = None
                    st.session_state.sharegpt_ai_role_value_sft = None
                    st.session_state.last_sharegpt_sft_config_keys = current_sharegpt_sft_config_keys

                if conversations_key_sharegpt_sft and conversations_key_sharegpt_sft != "不选择此字段":
                    all_inner_message_keys = set()
                    all_roles_in_data = set()
                    sample_current_item = data[0]

                    if conversations_key_sharegpt_sft in sample_current_item and \
                       isinstance(sample_current_item[conversations_key_sharegpt_sft], list):
                        for msg in sample_current_item[conversations_key_sharegpt_sft]:
                            if isinstance(msg, dict):
                                all_inner_message_keys.update(msg.keys())
                                if "from" in msg: all_roles_in_data.add(msg["from"])

                    if all_inner_message_keys:
                        role_key_options = ["不选择此字段"] + list(all_inner_message_keys)
                        role_key_sidebar_sharegpt_inner_sft = st.sidebar.selectbox("2. 选择角色标识键 (如: from)", role_key_options, index=get_default_index(role_key_options, "from"), key="sft_sgpt_role_key")
                        
                        if st.session_state.get('last_role_key_sidebar_sharegpt_inner_sft') != role_key_sidebar_sharegpt_inner_sft:
                            st.session_state.sharegpt_content_key_sft = None 
                            st.session_state.sharegpt_human_role_value_sft = None
                            st.session_state.sharegpt_ai_role_value_sft = None
                            st.session_state.last_role_key_sidebar_sharegpt_inner_sft = role_key_sidebar_sharegpt_inner_sft

                        if role_key_sidebar_sharegpt_inner_sft and role_key_sidebar_sharegpt_inner_sft != "不选择此字段":
                            other_keys = [k for k in all_inner_message_keys if k != role_key_sidebar_sharegpt_inner_sft]
                            auto_content_key = None
                            if "value" in other_keys: auto_content_key = "value"
                            elif "content" in other_keys: auto_content_key = "content"
                            elif len(other_keys) > 0: auto_content_key = other_keys[0] 

                            if auto_content_key:
                                st.session_state.sharegpt_role_key_sft = role_key_sidebar_sharegpt_inner_sft
                                st.session_state.sharegpt_content_key_sft = auto_content_key
                                st.sidebar.info(f"对话内容键自动识别为: **`{st.session_state.sharegpt_content_key_sft}`**")
                            else:
                                st.sidebar.error("无法识别内容键。请检查消息结构。")
                                st.session_state.sharegpt_role_key_sft = None
                                st.session_state.sharegpt_content_key_sft = None
                            
                            if st.session_state.sharegpt_role_key_sft and st.session_state.sharegpt_content_key_sft:
                                role_value_options = ["不选择此字段"] + [val for val in all_roles_in_data if val is not None] 
                                human_role_value_sidebar_sharegpt_inner_sft = st.sidebar.selectbox(f"3. 选择用户角色标识 ({st.session_state.sharegpt_role_key_sft}字段的值, 如: human)", role_value_options, index=get_default_index(role_value_options, "human"), key="sft_sgpt_human_role")
                                ai_role_value_sidebar_sharegpt_inner_sft = st.sidebar.selectbox(f"4. 选择AI角色标识 ({st.session_state.sharegpt_role_key_sft}字段的值, 如: gpt 或 assistant)", role_value_options, index=get_default_index(role_value_options, "gpt") if "gpt" in role_value_options else get_default_index(role_value_options, "assistant"), key="sft_sgpt_ai_role")
                                st.session_state.sharegpt_human_role_value_sft = human_role_value_sidebar_sharegpt_inner_sft
                                st.session_state.sharegpt_ai_role_value_sft = ai_role_value_sidebar_sharegpt_inner_sft
                            else:
                                st.sidebar.warning("无法识别角色键或内容键，无法继续配置角色值。请检查ShareGPT消息结构。")
                        else:
                            st.sidebar.warning("请选择 '角色标识键' 以继续配置角色值。")
                    else:
                        st.sidebar.warning("无法从对话列表中提取内部消息键。请检查数据集结构。")
                else:
                    st.sidebar.warning("请选择 '对话列表字段' 以开始配置 ShareGPT 格式。")

            elif active_format == DATA_FORMAT_UNKNOWN: 
                st.sidebar.markdown("**通用/未知格式字段**")
                instruction_key_alpaca_sft = st.sidebar.selectbox("选择Prompt/Instruction字段", field_options, index=0, key="sft_unk_instr_gen")
                output_key_alpaca_sft = st.sidebar.selectbox("选择Output字段", field_options, index=0, key="sft_unk_output_gen")
                st.sidebar.info("未知格式下，字段解释可能不准确。请尝试手动选择对应字段。")

            # Finalize SFT keys
            instruction_key_final = instruction_key_alpaca_sft if instruction_key_alpaca_sft != "不选择此字段" else ""
            input_key_final = input_key_alpaca_sft if input_key_alpaca_sft != "不选择此字段" else ""
            output_key_final = output_key_alpaca_sft if output_key_alpaca_sft != "不选择此字段" else ""
            conversations_key_final = conversations_key_sharegpt_sft if conversations_key_sharegpt_sft != "不选择此字段" else ""

            # Use st.session_state for ShareGPT internal keys
            role_key_final_sft = st.session_state.sharegpt_role_key_sft
            content_key_final_sft = st.session_state.sharegpt_content_key_sft
            human_role_value_final_sft = st.session_state.sharegpt_human_role_value_sft
            ai_role_value_final_sft = st.session_state.sharegpt_ai_role_value_sft

            # Display SFT data
            is_alpaca_sft_ready = active_format == DATA_FORMAT_ALPACA_SFT and instruction_key_final and output_key_final
            is_sharegpt_sft_ready = active_format == DATA_FORMAT_SHAREGPT_SFT and \
                                   conversations_key_final and role_key_final_sft and content_key_final_sft and \
                                   human_role_value_final_sft and ai_role_value_final_sft
            is_unknown_sft_ready = active_format == DATA_FORMAT_UNKNOWN and (instruction_key_final or output_key_final)

            is_ready_to_display = is_alpaca_sft_ready or is_sharegpt_sft_ready or is_unknown_sft_ready

            if not is_ready_to_display:
                st.info("请完成SFT数据字段配置。")
            else:
                display_sft_data(
                    current_item=data[st.session_state.current_index],
                    active_format=active_format,
                    tokenizer_enabled=st.session_state.tokenizer_enabled,
                    tokenizer_obj=st.session_state.tokenizer,
                    image_key_final=image_key, # Image key is universal
                    final_image_folder_path=final_image_folder_path,
                    image_scale_factor=image_scale_factor,
                    field_options=field_options,
                    instruction_key_alpaca_sft=instruction_key_final,
                    input_key_alpaca_sft=input_key_final,
                    output_key_alpaca_sft=output_key_final,
                    conversations_key_sharegpt_sft=conversations_key_final,
                    sharegpt_role_key_sft=role_key_final_sft,
                    sharegpt_content_key_sft=content_key_final_sft,
                    sharegpt_human_role_value_sft=human_role_value_final_sft,
                    sharegpt_ai_role_value_sft=ai_role_value_final_sft
                )

        elif view_mode == "预训练数据":
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 预训练数据字段配置")
            selected_format = st.sidebar.selectbox(
                "手动选择预训练数据格式 (若自动检测不准确)",
                [DATA_FORMAT_UNKNOWN, DATA_FORMAT_PRETRAIN],
                index=get_default_index([DATA_FORMAT_UNKNOWN, DATA_FORMAT_PRETRAIN], detected_format),
                key="pretrain_format_select"
            )
            active_format = selected_format

            is_ready_to_display = (active_format == DATA_FORMAT_PRETRAIN) or \
                                  (active_format == DATA_FORMAT_UNKNOWN and len(field_options) > 1) # If UNKNOWN and user has options to select

            if not is_ready_to_display:
                st.info("请完成预训练数据字段配置。")
            else:
                display_pretrain_data(
                    current_item=data[st.session_state.current_index],
                    active_format=active_format,
                    image_key_final=image_key, # Image key is universal
                    final_image_folder_path=final_image_folder_path,
                    image_scale_factor=image_scale_factor,
                    field_options=field_options
                )
        
        # --- 核心显示逻辑 ---
        # 统一的数据导航按钮
        st.markdown("---") 
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ 上一条数据", disabled=(st.session_state.current_index == 0), key="nav_prev_data"):
                st.session_state.current_index = max(0, st.session_state.current_index - 1)
                st.session_state.current_image_index = 0 # Reset image index when changing data entry
        with col3:
            if st.button("下一条数据 ➡️", disabled=(st.session_state.current_index >= len(data) - 1), key="nav_next_data"):
                st.session_state.current_index = min(len(data) - 1, st.session_state.current_index + 1)
                st.session_state.current_image_index = 0 # Reset image index when changing data entry

        st.markdown(f"<h3 style='font-size: 1.5rem; margin-bottom: 0.5rem;'>数据详情 (条目 {st.session_state.current_index + 1} / {len(data)})</h3>", unsafe_allow_html=True)


    else: # If no data file path is provided
        st.info("请在左侧边栏选择您的数据集文件。")

if __name__ == '__main__':
    main()