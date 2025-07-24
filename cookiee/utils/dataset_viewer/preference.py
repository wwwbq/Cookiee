import streamlit as st
import json
import os
from PIL import Image, UnidentifiedImageError
import base64
from io import BytesIO
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer

# 定义数据格式常量
DATA_FORMAT_UNKNOWN = "未知格式"
DATA_FORMAT_ALPACA_PREF = "Alpaca偏好格式 (instruction/chosen/rejected)"
DATA_FORMAT_SHAREGPT_PREF = "ShareGPT偏好格式 (conversations/chosen/rejected)"

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
    尝试检测数据集格式 (Alpaca Preference 或 ShareGPT Preference)，根据数据类型和结构。
    如果无法识别，默认为未知格式。
    """
    if not sample_data or not isinstance(sample_data, dict):
        return DATA_FORMAT_UNKNOWN

    # ShareGPT Preference 格式判断: 顶级是字典，且包含 'conversations', 'chosen', 'rejected' 键
    # 并且 'conversations' 是列表，'chosen' 和 'rejected' 是字典
    if "conversations" in sample_data and \
       isinstance(sample_data.get("conversations"), list) and \
       "chosen" in sample_data and isinstance(sample_data.get("chosen"), dict) and \
       "rejected" in sample_data and isinstance(sample_data.get("rejected"), dict) and \
       len(sample_data["conversations"]) > 0 and isinstance(sample_data["conversations"][0], dict) and "from" in sample_data["conversations"][0]:
        return DATA_FORMAT_SHAREGPT_PREF

    # Alpaca Preference 格式判断: 顶级是字典，且包含 'instruction', 'chosen', 'rejected' 键
    if "instruction" in sample_data and "chosen" in sample_data and "rejected" in sample_data:
        return DATA_FORMAT_ALPACA_PREF

    return DATA_FORMAT_UNKNOWN # 无法识别时返回未知

def main():
    st.set_page_config(layout="wide")
    
    # 调整主标题字体大小
    st.markdown("<h1 style='font-size: 2.2em;'>偏好数据可视化工具</h1>", unsafe_allow_html=True)

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
    if 'tokenizer_enabled' not in st.session_state:
        st.session_state.tokenizer_enabled = False 

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
            temp_upload_dir = "uploaded_datasets_temp_preference" # 更改临时目录名称
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
    # 修正：移除错误的 "鹰嘴豆"
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
            [DATA_FORMAT_UNKNOWN, DATA_FORMAT_ALPACA_PREF, DATA_FORMAT_SHAREGPT_PREF],
            index=[DATA_FORMAT_UNKNOWN, DATA_FORMAT_ALPACA_PREF, DATA_FORMAT_SHAREGPT_PREF].index(detected_format)
        )
        
        active_format = selected_format

        all_keys = list(data[0].keys()) if data else []
        field_options = ["不选择此字段"] + all_keys

        st.sidebar.markdown("---") 
        st.sidebar.subheader("字段映射") 

        # 偏好数据 Alpaca 格式的键
        instruction_key_alpaca_pref = ""
        input_key_alpaca_pref = ""
        chosen_key_alpaca_pref = ""
        rejected_key_alpaca_pref = ""

        # 偏好数据 ShareGPT 格式的键
        conversations_key_sidebar_sharegpt_pref = ""
        chosen_key_sharegpt_pref = ""
        rejected_key_sharegpt_pref = ""
        
        # ShareGPT 内部对话的键
        role_key_sidebar_sharegpt_inner = ""
        human_role_value_sidebar_sharegpt_inner = ""
        ai_role_value_sidebar_sharegpt_inner = ""

        # 通用图片键
        image_key = ""


        if active_format == DATA_FORMAT_ALPACA_PREF:
            st.sidebar.markdown("**Alpaca偏好格式字段**")
            instruction_key_alpaca_pref = st.sidebar.selectbox(
                "选择Instruction字段 (如: instruction)", 
                field_options, 
                index=field_options.index("instruction") if "instruction" in all_keys else 0
            )
            input_key_alpaca_pref = st.sidebar.selectbox( 
                "选择Input字段 (可选, 如: input)", 
                field_options, 
                index=field_options.index("input") if "input" in all_keys else 0
            )
            chosen_key_alpaca_pref = st.sidebar.selectbox(
                "选择Chosen字段 (如: chosen)", 
                field_options, 
                index=field_options.index("chosen") if "chosen" in all_keys else 0
            )
            rejected_key_alpaca_pref = st.sidebar.selectbox(
                "选择Rejected字段 (如: rejected)", 
                field_options, 
                index=field_options.index("rejected") if "rejected" in all_keys else 0
            )
        
        elif active_format == DATA_FORMAT_SHAREGPT_PREF:
            st.sidebar.markdown("**ShareGPT偏好格式字段**")
            
            conversations_key_sidebar_sharegpt_pref = st.sidebar.selectbox(
                "1. 选择对话列表字段 (通常是: conversations)", 
                field_options, 
                index=field_options.index("conversations") if "conversations" in all_keys else 0
            )
            chosen_key_sharegpt_pref = st.sidebar.selectbox(
                "2. 选择Chosen对话字段 (如: chosen)",
                field_options,
                index=field_options.index("chosen") if "chosen" in all_keys else 0
            )
            rejected_key_sharegpt_pref = st.sidebar.selectbox(
                "3. 选择Rejected对话字段 (如: rejected)",
                field_options,
                index=field_options.index("rejected") if "rejected" in all_keys else 0
            )

            # SFT ShareGPT 模式的内部对话字段逻辑也在这里复用，但更名为 sharegpt_inner
            # 由于 sharegpt_role_key 和 sharegpt_content_key 存储在 session_state，
            # 需要判断是否更新以避免重置用户已选值
            current_sharegpt_pref_config_keys = (
                conversations_key_sidebar_sharegpt_pref,
                chosen_key_sharegpt_pref,
                rejected_key_sharegpt_pref
            )
            if st.session_state.get('last_sharegpt_pref_config_keys') != current_sharegpt_pref_config_keys:
                st.session_state.sharegpt_role_key = None
                st.session_state.sharegpt_content_key = None
                st.session_state.sharegpt_human_role_value = None
                st.session_state.sharegpt_ai_role_value = None
                st.session_state.last_sharegpt_pref_config_keys = current_sharegpt_pref_config_keys

            # 只有当 conversations_key_sidebar_sharegpt_pref, chosen_key_sharegpt_pref, rejected_key_sharegpt_pref
            # 中至少一个有效时，才尝试提取内部对话键
            if (conversations_key_sidebar_sharegpt_pref and conversations_key_sidebar_sharegpt_pref != "不选择此字段") or \
               (chosen_key_sharegpt_pref and chosen_key_sharegpt_pref != "不选择此字段") or \
               (rejected_key_sharegpt_pref and rejected_key_sharegpt_pref != "不选择此字段"):

                # 收集所有可能的内部消息键和角色值
                all_inner_message_keys = set()
                all_roles_in_data = set()

                sample_current_item = data[0] # 使用第一个数据条目进行采样

                # 遍历所有可能的对话列表字段来收集键和角色
                for dataset_key in [conversations_key_sidebar_sharegpt_pref]:
                    if dataset_key and dataset_key != "不选择此字段" and dataset_key in sample_current_item and isinstance(sample_current_item[dataset_key], list):
                        for msg in sample_current_item[dataset_key]:
                            if isinstance(msg, dict): # 确保msg是字典
                                all_inner_message_keys.update(msg.keys())
                                if "from" in msg: # 假设角色键是 "from"
                                    all_roles_in_data.add(msg["from"])
                
                # 额外处理 chosen 和 rejected (它们是字典，不是列表)
                for dataset_key in [chosen_key_sharegpt_pref, rejected_key_sharegpt_pref]:
                    if dataset_key and dataset_key != "不选择此字段" and dataset_key in sample_current_item and isinstance(sample_current_item[dataset_key], dict):
                        msg = sample_current_item[dataset_key] # 直接是字典
                        all_inner_message_keys.update(msg.keys())
                        if "from" in msg:
                            all_roles_in_data.add(msg["from"])

                if all_inner_message_keys:
                    role_key_options = ["不选择此字段"] + list(all_inner_message_keys)
                    role_key_sidebar_sharegpt_inner = st.sidebar.selectbox(
                        "4. 选择角色标识键 (如: from)", 
                        role_key_options,
                        index=role_key_options.index("from") if "from" in role_key_options else 0
                    )
                    
                    if st.session_state.get('last_role_key_sidebar_sharegpt_inner') != role_key_sidebar_sharegpt_inner:
                        st.session_state.sharegpt_content_key = None 
                        st.session_state.sharegpt_human_role_value = None
                        st.session_state.sharegpt_ai_role_value = None
                        st.session_state.last_role_key_sidebar_sharegpt_inner = role_key_sidebar_sharegpt_inner

                    if role_key_sidebar_sharegpt_inner and role_key_sidebar_sharegpt_inner != "不选择此字段":
                        other_keys = [k for k in all_inner_message_keys if k != role_key_sidebar_sharegpt_inner]
                        
                        # 尝试自动识别内容键
                        auto_content_key = None
                        if "value" in other_keys: # Common for ShareGPT
                            auto_content_key = "value"
                        elif "content" in other_keys: # Also common
                            auto_content_key = "content"
                        elif len(other_keys) > 0:
                            auto_content_key = other_keys[0] # Fallback to first other key

                        if auto_content_key:
                            st.session_state.sharegpt_role_key = role_key_sidebar_sharegpt_inner
                            st.session_state.sharegpt_content_key = auto_content_key
                            st.sidebar.info(f"对话内容键自动识别为: **`{st.session_state.sharegpt_content_key}`**")
                        else:
                            st.sidebar.error("无法识别内容键。请检查消息结构。")
                            st.session_state.sharegpt_role_key = None
                            st.session_state.sharegpt_content_key = None
                        
                        if st.session_state.sharegpt_role_key and st.session_state.sharegpt_content_key:
                            role_value_options = ["不选择此字段"] + [val for val in all_roles_in_data if val is not None] 
                            
                            human_role_value_sidebar_sharegpt_inner = st.sidebar.selectbox(
                                f"5. 选择用户角色标识 ({st.session_state.sharegpt_role_key}字段的值, 如: human)", 
                                role_value_options,
                                index=role_value_options.index("human") if "human" in role_value_options else 0
                            )
                            ai_role_value_sidebar_sharegpt_inner = st.sidebar.selectbox(
                                f"6. 选择AI角色标识 ({st.session_state.sharegpt_role_key}字段的值, 如: gpt 或 assistant)", 
                                role_value_options,
                                index=role_value_options.index("gpt") if "gpt" in role_value_options else (role_value_options.index("assistant") if "assistant" in role_value_options else 0)
                            )
                            st.session_state.sharegpt_human_role_value = human_role_value_sidebar_sharegpt_inner
                            st.session_state.sharegpt_ai_role_value = ai_role_value_sidebar_sharegpt_inner

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
            # 通用 Prompt/Instruction 字段
            instruction_key_alpaca_pref = st.sidebar.selectbox( 
                "选择Prompt/Instruction字段 (对于Alpaca偏好数据的instruction)", 
                field_options, 
                index=0 
            )
            # 通用 Chosen 字段
            chosen_key_alpaca_pref = st.sidebar.selectbox(
                "选择Chosen文本字段 (对于Alpaca偏好数据的chosen)", 
                field_options, 
                index=0 
            )
            # 通用 Rejected 字段
            rejected_key_alpaca_pref = st.sidebar.selectbox(
                "选择Rejected文本字段 (对于Alpaca偏好数据的rejected)", 
                field_options, 
                index=0 
            )
            st.sidebar.info("未知格式下，字段解释可能不准确。请尝试手动选择对应字段。")


        image_key = st.sidebar.selectbox("选择Image字段 (可选)", field_options, index=field_options.index("image") if "image" in all_keys else 0)

        # 最终使用的键名
        # Alpaca Preference 模式
        instruction_key = instruction_key_alpaca_pref if instruction_key_alpaca_pref != "不选择此字段" else ""
        input_key = input_key_alpaca_pref if input_key_alpaca_pref != "不选择此字段" else ""
        chosen_key = chosen_key_alpaca_pref if chosen_key_alpaca_pref != "不选择此字段" else ""
        rejected_key = rejected_key_alpaca_pref if rejected_key_alpaca_pref != "不选择此字段" else ""
        
        # ShareGPT Preference 模式
        conversations_key = conversations_key_sidebar_sharegpt_pref if conversations_key_sidebar_sharegpt_pref != "不选择此字段" else ""
        chosen_key_sharegpt = chosen_key_sharegpt_pref if chosen_key_sharegpt_pref != "不选择此字段" else ""
        rejected_key_sharegpt = rejected_key_sharegpt_pref if rejected_key_sharegpt_pref != "不选择此字段" else ""

        # ShareGPT 内部对话的最终键名
        role_key_final = st.session_state.sharegpt_role_key
        content_key_final = st.session_state.sharegpt_content_key
        human_role_value_final = st.session_state.sharegpt_human_role_value
        ai_role_value_final = st.session_state.sharegpt_ai_role_value

        image_key_final = image_key if image_key != "不选择此字段" else ""
        
        st.sidebar.markdown("---")
        st.sidebar.header("图片设置")
        image_scale_factor = st.sidebar.slider(
            "选择图片缩放倍数",
            min_value=0.1, max_value=2.0, value=1.0, step=0.1,
            help="调整图片显示大小。1.0为原始大小，大于1.0放大，小于1.0缩小。"
        )
        
        # --- 核心显示逻辑 ---
        # 字段未选择完全时的提示
        is_alpaca_pref_ready = active_format == DATA_FORMAT_ALPACA_PREF and instruction_key and chosen_key and rejected_key
        is_sharegpt_pref_ready = active_format == DATA_FORMAT_SHAREGPT_PREF and \
                                 conversations_key and chosen_key_sharegpt and rejected_key_sharegpt and \
                                 role_key_final and content_key_final and human_role_value_final and ai_role_value_final
        is_unknown_ready = active_format == DATA_FORMAT_UNKNOWN and (instruction_key or chosen_key or rejected_key or image_key_final)

        is_ready_to_display_main_content = is_alpaca_pref_ready or is_sharegpt_pref_ready or is_unknown_ready

        if not is_ready_to_display_main_content:
            st.info("请通过左侧边栏选择数据集，并完成数据格式及字段映射配置，以便开始可视化。")
            return # 未配置完成，直接返回，不显示下方内容

        st.markdown("---") 

        # 调整字体大小
        st.markdown(f"<h3 style='font-size: 1.5rem; margin-bottom: 0.5rem;'>数据详情 (条目 {st.session_state.current_index + 1} / {len(data)})</h3>", unsafe_allow_html=True)

        current_item = data[st.session_state.current_index]

        current_prompt_text = ""
        current_chosen_text = ""
        current_rejected_text = ""
        
        use_tokenizer_for_prompt = st.session_state.tokenizer is not None and st.session_state.tokenizer_enabled

        def format_messages_with_tokenizer(messages, tokenizer_obj, add_generation_prompt=True):
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

        if active_format == DATA_FORMAT_ALPACA_PREF:
            instruction_content = current_item.get(instruction_key, "") 
            input_content = current_item.get(input_key, "")   
            chosen_content = current_item.get(chosen_key, "") 
            rejected_content = current_item.get(rejected_key, "") 

            # 构建 Prompt
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
            
            if use_tokenizer_for_prompt and messages_for_prompt_template:
                formatted_prompt = format_messages_with_tokenizer(messages_for_prompt_template, st.session_state.tokenizer, add_generation_prompt=True)
                current_prompt_text = formatted_prompt if formatted_prompt is not None else combined_prompt_raw # 回退到原始
            else:
                formatted_parts = []
                if instruction_content:
                    formatted_parts.append(f"**{instruction_key}**: {instruction_content}")
                if input_content:
                    formatted_parts.append(f"**{input_key}**: {input_content}")
                current_prompt_text = "\n\n".join(formatted_parts)
            
            current_chosen_text = chosen_content if chosen_content else ""
            current_rejected_text = rejected_content if rejected_content else ""


        elif active_format == DATA_FORMAT_SHAREGPT_PREF:
            if conversations_key and conversations_key in current_item and \
               isinstance(current_item[conversations_key], list) and \
               chosen_key_sharegpt and chosen_key_sharegpt in current_item and \
               rejected_key_sharegpt and rejected_key_sharegpt in current_item and \
               role_key_final and content_key_final and human_role_value_final and ai_role_value_final: 
                
                conversation_base = current_item[conversations_key]
                chosen_data = current_item[chosen_key_sharegpt] # 这是一个字典
                rejected_data = current_item[rejected_key_sharegpt] # 这是一个字典

                # 提取 Prompt (通常是 conversations 的除最后一条 AI 回复之外的部分)
                prompt_messages_for_template = []
                simple_prompt_messages = []

                for msg in conversation_base:
                    if not isinstance(msg, dict):
                        st.warning(f"跳过非字典类型的对话消息: {msg}")
                        continue
                    if msg.get(role_key_final) == human_role_value_final:
                        prompt_messages_for_template.append({"role": "user", "content": msg.get(content_key_final, "")})
                        simple_prompt_messages.append(f"**{human_role_value_final}**: {msg.get(content_key_final, '')}")
                    elif msg.get(role_key_final) == ai_role_value_final:
                        prompt_messages_for_template.append({"role": "assistant", "content": msg.get(content_key_final, "")})
                        simple_prompt_messages.append(f"**{ai_role_value_final}**: {msg.get(content_key_final, '')}")
                    else:
                        st.warning(f"当前数据条目中的对话消息角色未识别: '{msg.get(role_key_final)}'。已跳过此消息。")

                if use_tokenizer_for_prompt and prompt_messages_for_template:
                    # Prompt 使用 add_generation_prompt=True
                    formatted_prompt = format_messages_with_tokenizer(prompt_messages_for_template, st.session_state.tokenizer, add_generation_prompt=True)
                    current_prompt_text = formatted_prompt if formatted_prompt is not None else "\n\n".join(simple_prompt_messages)
                else:
                    current_prompt_text = "\n\n".join(simple_prompt_messages)

                # 提取 Chosen 和 Rejected (直接是字典)
                # Chosen 和 Rejected 只显示它们自己的内容，不使用 tokenizer.apply_chat_template
                if isinstance(chosen_data, dict) and chosen_data.get(role_key_final) == ai_role_value_final:
                    current_chosen_text = chosen_data.get(content_key_final, "")
                else:
                    st.warning("Chosen 数据不是预期的字典格式或角色不匹配。")
                    current_chosen_text = ""

                if isinstance(rejected_data, dict) and rejected_data.get(role_key_final) == ai_role_value_final:
                    current_rejected_text = rejected_data.get(content_key_final, "")
                else:
                    st.warning("Rejected 数据不是预期的字典格式或角色不匹配。")
                    current_rejected_text = ""

            else:
                st.warning("ShareGPT偏好数据配置不完整或数据结构异常。无法正确显示。")
                pass # 如果配置不完整，则保持为空


        elif active_format == DATA_FORMAT_UNKNOWN: 
            # 在未知格式下，将用户选择的字段直接显示
            if instruction_key and instruction_key in current_item:
                current_prompt_text = current_item[instruction_key]
            if chosen_key and chosen_key in current_item:
                current_chosen_text = current_item[chosen_key]
            if rejected_key and rejected_key in current_item:
                current_rejected_text = current_item[rejected_key]


        st.subheader("Prompt:") 
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


        st.subheader("图片:") 
        if image_key_final and image_key_final in current_item and final_image_folder_path and os.path.isdir(final_image_folder_path):
            image_paths = []
            image_value = current_item[image_key_final]

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
            if image_key_final:
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