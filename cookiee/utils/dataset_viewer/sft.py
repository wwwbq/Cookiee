import streamlit as st
import json
import os
from PIL import Image, UnidentifiedImageError
import base64
from io import BytesIO
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer

# 定义数据格式常量
DATA_FORMAT_UNKNOWN = "未知格式"
DATA_FORMAT_ALPACA = "Alpaca格式 (instruction/output)"
DATA_FORMAT_SHAREGPT = "ShareGPT格式 (conversations)"

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
            # Use LANCZOS for high-quality downsampling/upsampling
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
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
    尝试检测数据集格式 (Alpaca 或 ShareGPT)，根据数据类型和结构。
    如果无法识别，默认为未知格式。
    """
    if not sample_data or not isinstance(sample_data, dict):
        return DATA_FORMAT_UNKNOWN

    # ShareGPT 格式判断: 顶级是字典，且包含 'conversations' 键，其值是一个列表
    # 并且列表的第一个元素是字典，且包含 'from' 键
    if "conversations" in sample_data and \
       isinstance(sample_data["conversations"], list) and \
       len(sample_data["conversations"]) > 0 and \
       isinstance(sample_data["conversations"][0], dict) and \
       "from" in sample_data["conversations"][0]: 
        
        first_msg_keys = list(sample_data["conversations"][0].keys())
        if len(first_msg_keys) != 2:
            st.sidebar.warning(f"警告: 自动检测到ShareGPT数据中对话消息包含 {len(first_msg_keys)} 个键（如：`{first_msg_keys}`），期望只有两个键。这可能不是标准的ShareGPT格式。")
        
        return DATA_FORMAT_SHAREGPT

    # Alpaca 格式判断: 顶级是字典，且且同时包含 'instruction' 和 'output' 键
    # 进一步，如果存在 'input' 键，也认为是 Alpaca 格式的变体。
    if "instruction" in sample_data and "output" in sample_data:
        return DATA_FORMAT_ALPACA

    return DATA_FORMAT_UNKNOWN # 无法识别时返回未知

def main():
    st.set_page_config(layout="wide")
    
    # 调整主标题字体大小
    st.markdown("<h1 style='font-size: 2.2em;'>SFT数据集可视化工具</h1>", unsafe_allow_html=True)

    # 初始化会话状态
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0
    if 'sharegpt_role_key' not in st.session_state:
        st.session_state.sharegpt_role_key = None
    if 'sharegpt_content_key' not in st.session_state:
        st.session_state.sharegpt_content_key = None
    if 'sharegpt_human_role_value' not in st.session_state:
        st.session_state.sharegpt_human_role_value = None
    if 'sharegpt_ai_role_value' not in st.session_state:
        st.session_state.sharegpt_ai_role_value = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'last_data_file_path' not in st.session_state:
        st.session_state.last_data_file_path = None
    if 'last_image_folder_path' not in st.session_state:
        st.session_state.last_image_folder_path = None
    if 'last_tokenizer_path' not in st.session_state:
        st.session_state.last_tokenizer_path = None
    # 新增 tokenizer 启用状态
    if 'tokenizer_enabled' not in st.session_state:
        st.session_state.tokenizer_enabled = False # 默认关闭

    # Temp variables for paths, will be processed into final variables
    temp_data_file_path = None
    temp_image_folder_path = None
    temp_tokenizer_path = None 
    
    # 检查命令行参数
    import sys
    args = sys.argv[1:] 
    
    if len(args) > 0:
        temp_data_file_path = args[0]
        if len(args) > 1:
            temp_image_folder_path = args[1]
        if len(args) > 2: 
            temp_tokenizer_path = args[2]

    st.sidebar.header("文件与路径配置")

    # --- 数据集文件选择 (命令行或上传) ---
    st.sidebar.subheader("数据集")
    if not temp_data_file_path:
        uploaded_file = st.sidebar.file_uploader("选择您的JSON或JSONL数据集", type=["json", "jsonl"])
        if uploaded_file is not None:
            temp_upload_dir = "uploaded_datasets_temp"
            os.makedirs(temp_upload_dir, exist_ok=True)
            temp_data_file_path = os.path.join(temp_upload_dir, uploaded_file.name)
            with open(temp_data_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"已上传数据集: **`{uploaded_file.name}`**")
            st.sidebar.warning("⚠️ **请注意：** 您通过页面上传文件，无法自动获取数据集的父级目录作为图片文件夹和 Tokenizer 路径。**如果需要显示图片或使用 Tokenizer，请务必手动填写路径。**")
    
    # Display current data file path
    if temp_data_file_path:
        st.sidebar.write(f"**数据集路径:** `{temp_data_file_path}`")
    else:
        st.sidebar.info("未选择数据集。")


    # --- 图片文件夹路径输入 (命令行或手动输入) ---
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
    

    # --- Tokenizer 路径输入 (命令行或手动输入) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Tokenizer 目录 (可选)")
    tokenizer_path_input = st.sidebar.text_input("请输入Tokenizer目录路径", value=temp_tokenizer_path if temp_tokenizer_path else "")
    
    tokenizer_loaded_successfully = False

    if tokenizer_path_input:
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
                    tokenizer_loaded_successfully = True
                    # 如果成功加载，默认启用
                    st.session_state.tokenizer_enabled = True 
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
    
    # Display current tokenizer status
    if st.session_state.tokenizer:
        st.sidebar.write(f"**Tokenizer 已加载:** `{st.session_state.last_tokenizer_path}`")
        # 添加启用/禁用按钮
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
            disabled=True, # 没有 tokenizer 时禁用按钮
            help="未加载 Tokenizer，此功能不可用。"
        )

    # Finalize data_file_path for processing
    data_file_path = temp_data_file_path 
    
    # --- End of File Selection & Config ---
    st.sidebar.markdown("---") 

    if data_file_path:
        data = load_data(data_file_path)
        if data is None:
            return

        if not data:
            st.warning("数据集中没有数据。")
            return

        # 自动判断数据格式
        detected_format = detect_data_format(data[0])
        st.sidebar.header(f"数据格式识别与字段映射") 
        st.sidebar.subheader(f"检测到的格式: **`{detected_format}`**")
        selected_format = st.sidebar.selectbox(
            "手动选择数据格式 (若自动检测不准确)",
            [DATA_FORMAT_UNKNOWN, DATA_FORMAT_ALPACA, DATA_FORMAT_SHAREGPT],
            index=[DATA_FORMAT_UNKNOWN, DATA_FORMAT_ALPACA, DATA_FORMAT_SHAREGPT].index(detected_format)
        )
        
        active_format = selected_format

        all_keys = list(data[0].keys()) if data else []
        field_options = ["不选择此字段"] + all_keys

        st.sidebar.markdown("---") 
        st.sidebar.subheader("字段映射") 

        instruction_key_selected = "" 
        input_key_selected = ""       
        label_key_selected = ""       
        image_key = ""
        
        conversations_key_sidebar = ""
        role_key_sidebar = ""
        human_role_value_sidebar = ""
        ai_role_value_sidebar = ""


        if active_format == DATA_FORMAT_ALPACA:
            st.sidebar.markdown("**Alpaca格式字段**")
            instruction_key_selected = st.sidebar.selectbox(
                "选择Instruction字段 (如: instruction)", 
                field_options, 
                index=field_options.index("instruction") if "instruction" in all_keys else 0
            )
            input_key_selected = st.sidebar.selectbox( 
                "选择Input字段 (可选, 如: input)", 
                field_options, 
                index=field_options.index("input") if "input" in all_keys else 0
            )
            label_key_selected = st.sidebar.selectbox(
                "选择Output字段 (如: output)", 
                field_options, 
                index=field_options.index("output") if "output" in all_keys else 0
            )
        
        elif active_format == DATA_FORMAT_SHAREGPT:
            st.sidebar.markdown("**ShareGPT格式字段**")
            
            conversations_key_sidebar = st.sidebar.selectbox(
                "1. 选择对话列表字段 (通常是: conversations)", 
                field_options, 
                index=field_options.index("conversations") if "conversations" in all_keys else 0
            )
            
            if st.session_state.get('last_conversations_key_sidebar') != conversations_key_sidebar:
                st.session_state.sharegpt_role_key = None
                st.session_state.sharegpt_content_key = None
                st.session_state.sharegpt_human_role_value = None
                st.session_state.sharegpt_ai_role_value = None
                st.session_state.last_conversations_key_sidebar = conversations_key_sidebar 

            if conversations_key_sidebar and conversations_key_sidebar != "不选择此字段":
                first_conversation_entry_list = data[0].get(conversations_key_sidebar, [])
                
                if first_conversation_entry_list and isinstance(first_conversation_entry_list[0], dict):
                    first_message_keys = list(first_conversation_entry_list[0].keys())
                    
                    role_key_options = ["不选择此字段"] + first_message_keys
                    role_key_sidebar = st.sidebar.selectbox(
                        "2. 选择角色标识键 (如: from)", 
                        role_key_options,
                        index=role_key_options.index("from") if "from" in role_key_options else 0
                    )
                    
                    if st.session_state.get('last_role_key_sidebar') != role_key_sidebar:
                        st.session_state.sharegpt_content_key = None 
                        st.session_state.sharegpt_human_role_value = None
                        st.session_state.sharegpt_ai_role_value = None
                        st.session_state.last_role_key_sidebar = role_key_sidebar

                    if role_key_sidebar and role_key_sidebar != "不选择此字段":
                        other_keys = [k for k in first_message_keys if k != role_key_sidebar]

                        if len(first_message_keys) == 2 and len(other_keys) == 1:
                            st.session_state.sharegpt_role_key = role_key_sidebar
                            st.session_state.sharegpt_content_key = other_keys[0]
                            st.sidebar.info(f"对话内容键自动识别为: **`{st.session_state.sharegpt_content_key}`**")
                        elif len(first_message_keys) > 2:
                            st.sidebar.warning(f"警告: 消息字典包含多于两个键（`{first_message_keys}`）。期望只有角色键(`{role_key_sidebar}`)和内容键。将使用除角色键外的第一个键作为内容键。")
                            st.session_state.sharegpt_role_key = role_key_sidebar
                            st.session_state.sharegpt_content_key = other_keys[0] if other_keys else None 
                            if st.session_state.sharegpt_content_key:
                                st.sidebar.info(f"对话内容键根据规则识别为: **`{st.session_state.sharegpt_content_key}`**")
                            else:
                                st.sidebar.error("无法识别内容键。请检查消息结构。")
                        elif len(first_message_keys) < 2:
                            st.sidebar.error(f"消息字典键数量异常（只有 {len(first_message_keys)} 个键）。期望至少两个键：角色键和内容键。")
                            st.session_state.sharegpt_role_key = None
                            st.session_state.sharegpt_content_key = None
                        
                        if st.session_state.sharegpt_role_key and st.session_state.sharegpt_content_key:
                            roles_in_conversation_values = list(set([msg.get(st.session_state.sharegpt_role_key) for msg in first_conversation_entry_list if st.session_state.sharegpt_role_key in msg]))
                            role_value_options = ["不选择此字段"] + [val for val in roles_in_conversation_values if val is not None] 
                            
                            human_role_value_sidebar = st.sidebar.selectbox(
                                f"3. 选择用户角色标识 ({st.session_state.sharegpt_role_key}字段的值, 如: human)", 
                                role_value_options,
                                index=role_value_options.index("human") if "human" in role_value_options else 0
                            )
                            ai_role_value_sidebar = st.sidebar.selectbox(
                                f"4. 选择AI角色标识 ({st.session_state.sharegpt_role_key}字段的值, 如: gpt 或 assistant)", 
                                role_value_options,
                                index=role_value_options.index("gpt") if "gpt" in role_value_options else (role_value_options.index("assistant") if "assistant" in role_value_options else 0)
                            )
                            st.session_state.sharegpt_human_role_value = human_role_value_sidebar
                            st.session_state.sharegpt_ai_role_value = ai_role_value_sidebar

                        else:
                            st.sidebar.warning("无法识别角色键或内容键，无法继续配置角色值。请检查ShareGPT消息结构。")
                    else:
                        st.sidebar.warning("请选择 '角色标识键' 以继续配置角色值。")
                else:
                    st.sidebar.warning("ShareGPT对话列表为空或格式不正确。请检查数据集。")
            else:
                st.sidebar.warning("请选择 '对话列表字段' 以开始配置 ShareGPT 格式。")
        
        elif active_format == DATA_FORMAT_UNKNOWN: 
            st.sidebar.markdown("**通用/未知格式字段**")
            instruction_key_selected = st.sidebar.selectbox( 
                "选择Prompt字段", 
                field_options, 
                index=0 
            )
            label_key_selected = st.sidebar.selectbox(
                "选择Label字段", 
                field_options, 
                index=0 
            )

        image_key = st.sidebar.selectbox("选择Image字段 (可选)", field_options, index=field_options.index("image") if "image" in all_keys else 0)

        # 最终使用的键名
        instruction_key = instruction_key_selected if instruction_key_selected != "不选择此字段" else ""
        input_key = input_key_selected if input_key_selected != "不选择此字段" else ""
        label_key = label_key_selected if label_key_selected != "不选择此字段" else ""
        image_key = image_key if image_key != "不选择此字段" else ""
        
        conversations_key = conversations_key_sidebar if conversations_key_sidebar != "不选择此字段" else ""
        human_role_value = st.session_state.sharegpt_human_role_value if st.session_state.sharegpt_human_role_value != "不选择此字段" else ""
        ai_role_value = st.session_state.sharegpt_ai_role_value if st.session_state.sharegpt_ai_role_value != "不选择此字段" else ""


        st.sidebar.markdown("---")
        st.sidebar.header("图片设置")
        image_scale_factor = st.sidebar.slider(
            "选择图片缩放倍数",
            min_value=0.1, max_value=2.0, value=1.0, step=0.1,
            help="调整图片显示大小。1.0为原始大小，大于1.0放大，小于1.0缩小。"
        )
        
        # --- 核心显示逻辑 ---
        # 字段未选择完全时的提示
        # Alpaca 格式需要 Instruction 和 Output 字段
        is_alpaca_ready = active_format == DATA_FORMAT_ALPACA and instruction_key and label_key
        # ShareGPT 格式需要 conversations 键，以及角色和内容键的推断（session_state中）
        is_sharegpt_ready = active_format == DATA_FORMAT_SHAREGPT and conversations_key and \
                            st.session_state.sharegpt_role_key and st.session_state.sharegpt_content_key and \
                            human_role_value and ai_role_value
        # 未知格式需要至少一个字段 (Prompt/Instruction 或 Label 或 Image)
        is_unknown_ready = active_format == DATA_FORMAT_UNKNOWN and (instruction_key or label_key or image_key)

        is_ready_to_display_main_content = is_alpaca_ready or is_sharegpt_ready or is_unknown_ready

        if not is_ready_to_display_main_content:
            st.info("请通过左侧边栏选择数据集，并完成数据格式及字段映射配置，以便开始可视化。")
            return # 未配置完成，直接返回，不显示下方内容

        st.markdown("---") 

        # 调整字体大小
        st.markdown(f"<h3 style='font-size: 1.5rem; margin-bottom: 0.5rem;'>数据详情 (条目 {st.session_state.current_index + 1} / {len(data)})</h3>", unsafe_allow_html=True)

        current_item = data[st.session_state.current_index]

        current_prompt_text = ""
        current_label_text = ""
        
        # 真正的 tokenizer 启用状态取决于是否加载成功并且用户勾选了
        use_tokenizer_for_prompt = st.session_state.tokenizer is not None and st.session_state.tokenizer_enabled
        
        if active_format == DATA_FORMAT_ALPACA:
            instruction_content = current_item.get(instruction_key, "") 
            input_content = current_item.get(input_key, "")   
            original_label_text = current_item.get(label_key, "") 

            if use_tokenizer_for_prompt:
                messages_for_template = []
                combined_content = ""
                if instruction_content:
                    combined_content += instruction_content
                if input_content:
                    if combined_content:
                        combined_content += f"\n{input_content}"
                    else:
                        combined_content = input_content
                
                if combined_content:
                    messages_for_template.append({"role": "user", "content": combined_content})
                
                if not messages_for_template: 
                    messages_for_template.append({"role": "user", "content": ""})

                try:
                    current_prompt_text = st.session_state.tokenizer.apply_chat_template(
                        messages_for_template, 
                        tokenize=False, 
                        add_generation_prompt=True 
                    )
                except Exception as e:
                    st.error(f"使用 Tokenizer 的 `apply_chat_template` 处理 Alpaca Prompt 失败，将回退到简单文本格式。错误: {e}") 
                    
                    # 回退到原始格式，带有字段名称
                    formatted_parts = []
                    if instruction_content:
                        formatted_parts.append(f"**{instruction_key}**: {instruction_content}")
                    if input_content:
                        formatted_parts.append(f"**{input_key}**: {input_content}")
                    current_prompt_text = "\n\n".join(formatted_parts)
            else: 
                # 未加载 Tokenizer 或未启用 Tokenizer 时，按照指定字段名显示
                formatted_parts = []
                if instruction_content:
                    formatted_parts.append(f"**{instruction_key}**: {instruction_content}")
                if input_content:
                    formatted_parts.append(f"**{input_key}**: {input_content}")
                current_prompt_text = "\n\n".join(formatted_parts)
            
            current_label_text = original_label_text if original_label_text else ""


        elif active_format == DATA_FORMAT_SHAREGPT:
            if conversations_key and conversations_key in current_item and \
               isinstance(current_item[conversations_key], list) and \
               human_role_value and ai_role_value and \
               st.session_state.sharegpt_role_key and st.session_state.sharegpt_content_key: 
                
                conversation_raw = current_item[conversations_key]
                determined_role_key = st.session_state.sharegpt_role_key
                determined_content_key = st.session_state.sharegpt_content_key

                messages_for_template = []
                simple_prompt_messages = []
                simple_label_message = ""
                
                for i, msg in enumerate(conversation_raw):
                    msg_keys = list(msg.keys())
                    if len(msg_keys) != 2 or determined_role_key not in msg_keys or determined_content_key not in msg_keys:
                        st.warning(f"当前数据条目中的第 {i+1} 条对话消息结构异常。已跳过此消息。请检查侧边栏配置。") 
                        continue 

                    role = msg.get(determined_role_key)
                    content = msg.get(determined_content_key, "")

                    hf_role = None
                    if role == human_role_value:
                        hf_role = "user" 
                    elif role == ai_role_value:
                        hf_role = "assistant" 
                    
                    if hf_role:
                        messages_for_template.append({"role": hf_role, "content": content})
                    else:
                        st.warning(f"未识别的对话角色: '{role}'。此消息将不会被包含在 Prompt 中。") 
                        pass

                    if i < len(conversation_raw) - 1: 
                        if role == human_role_value:
                            simple_prompt_messages.append(f"**{human_role_value}**: {content}")
                        elif role == ai_role_value:
                            simple_prompt_messages.append(f"**{ai_role_value}**: {content}")
                    else: 
                        if role == ai_role_value:
                            simple_label_message = content
                        else:
                            st.warning(f"ShareGPT格式数据中最后一段消息不是来自AI角色 ('{ai_role_value}')，无法识别为Label。") 

                if use_tokenizer_for_prompt and messages_for_template:
                    try:
                        prompt_for_tokenizer = messages_for_template[:-1] if len(messages_for_template) > 0 and messages_for_template[-1]['role'] == 'assistant' else messages_for_template
                        label_for_tokenizer_content = messages_for_template[-1]['content'] if len(messages_for_template) > 0 and messages_for_template[-1]['role'] == 'assistant' else ""

                        current_prompt_text = st.session_state.tokenizer.apply_chat_template(
                            prompt_for_tokenizer, 
                            tokenize=False, 
                            add_generation_prompt=True 
                        )
                        current_label_text = label_for_tokenizer_content 
                    except Exception as e:
                        st.error(f"使用 Tokenizer 的 `apply_chat_template` 处理 ShareGPT 数据失败，将回退到简单文本格式。错误: {e}") 
                        current_prompt_text = "\n\n".join(simple_prompt_messages)
                        current_label_text = simple_label_message
                else: 
                    current_prompt_text = "\n\n".join(simple_prompt_messages)
                    current_label_text = simple_label_message

            else:
                pass 
        
        elif active_format == DATA_FORMAT_UNKNOWN: 
            if instruction_key and instruction_key in current_item: # 此时 instruction_key 作为通用 Prompt Key
                current_prompt_text = current_item[instruction_key]
            
            if label_key and label_key in current_item:
                current_label_text = current_item[label_key]


        st.subheader("Prompt:") 
        if current_prompt_text:
            st.code(current_prompt_text, language="text")
        else:
            st.info("Prompt内容为空或未找到。请检查字段映射。") 

        st.subheader("Label:") 
        if current_label_text:
            st.code(current_label_text, language="text")
        else:
            st.info("Label内容为空或未找到。请检查字段映射。") 


        st.subheader("图片:") 
        if image_key and image_key in current_item and final_image_folder_path and os.path.isdir(final_image_folder_path):
            image_paths = []
            image_value = current_item[image_key]

            if isinstance(image_value, str):
                image_paths.append(get_image_path(final_image_folder_path, image_value))
            elif isinstance(image_value, list):
                for img_rel_path in image_value:
                    image_paths.append(get_image_path(final_image_folder_path, img_rel_path))
            else:
                pass 
            
            if image_paths:
                num_images = len(image_paths)
                st.session_state.current_image_index = st.session_state.current_image_index % num_images
                
                img_nav_col1, img_nav_col2, img_nav_col3 = st.columns([1, 2, 1])
                with img_nav_col1:
                    if st.button("⬅️ 上一张", key="prev_img", help="查看当前数据的上一张图片"):
                        st.session_state.current_image_index = (st.session_state.current_image_index - 1 + num_images) % num_images
                with img_nav_col3:
                    if st.button("下一张 ➡️", key="next_img", help="查看当前数据的下一张图片"):
                        st.session_state.current_image_index = (st.session_state.current_image_index + 1) % num_images

                st.markdown(f"<p style='text-align: center; color: grey; font-size: small;'>当前图片: {st.session_state.current_image_index + 1} / {num_images}</p>", unsafe_allow_html=True)
                
                current_image_path = image_paths[st.session_state.current_image_index]
                display_image(current_image_path, image_scale_factor)
            else:
                st.info("未找到有效图片路径或图片字段值类型不正确。") 
        else:
            if image_key:
                st.info("图片内容为空或未找到，请检查图片字段和文件夹路径。") 
            else:
                st.info("未选择图片字段。") 
            
        st.markdown("---") 

        st.markdown("### 浏览更多数据")
        data_nav_col_bottom1, data_nav_col_bottom2, data_nav_col_bottom3 = st.columns([1, 2, 1])
        with data_nav_col_bottom1:
            if st.button("⏪ 上一条数据", key="prev_data_bottom", help="查看上一条数据"):
                if st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
                    st.session_state.current_image_index = 0 
        with data_nav_col_bottom2:
            st.markdown(f"<h3 style='text-align: center; color: grey;'>{st.session_state.current_index + 1} / {len(data)}</h3>", unsafe_allow_html=True)
        with data_nav_col_bottom3:
            if st.button("下一条数据 ⏩", key="next_data_bottom", help="查看下一条数据"):
                if st.session_state.current_index < len(data) - 1:
                    st.session_state.current_index += 1
                    st.session_state.current_image_index = 0 

    else:
        st.info("请通过左侧边栏选择您的数据集或通过命令行参数指定。")

if __name__ == "__main__":
    main()