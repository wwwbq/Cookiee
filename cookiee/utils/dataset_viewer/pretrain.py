import streamlit as st
import json
import os
from PIL import Image, UnidentifiedImageError
import base64
from io import BytesIO
from transformers import AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizer

# 定义数据格式常量 (这里实际上只有一个显式格式)
DATA_FORMAT_PRETRAIN = "预训练格式 (text)"

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

def main():
    st.set_page_config(layout="wide")
    
    # 调整主标题字体大小
    st.markdown("<h1 style='font-size: 2.2em;'>预训练数据可视化工具</h1>", unsafe_allow_html=True)

    # 初始化会话状态
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'current_image_index' not in st.session_state:
        st.session_state.current_image_index = 0
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
            temp_upload_dir = "uploaded_datasets_temp_pretrain" # 更改临时目录名称
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
            "启用 Tokenizer 处理文本", 
            value=st.session_state.tokenizer_enabled,
            help="勾选此项将使用加载的 Tokenizer 对文本进行处理（例如编码/解码，尽管通常预训练数据不会使用apply_chat_template）。取消勾选将显示原始字段内容。"
        )
    else:
        st.sidebar.write(f"**Tokenizer 状态:** 未加载")
        st.sidebar.checkbox(
            "启用 Tokenizer 处理文本", 
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

        st.sidebar.header(f"数据格式识别与字段映射") 
        st.sidebar.subheader(f"当前模式: **`{DATA_FORMAT_PRETRAIN}`**") # 固定显示预训练格式

        all_keys = list(data[0].keys()) if data else []
        field_options = ["不选择此字段"] + all_keys

        st.sidebar.markdown("---") 
        st.sidebar.subheader("字段映射") 
        
        # 预训练相关的键
        text_key_selected = st.sidebar.selectbox(
            "选择Text字段 (如: text)",
            field_options,
            index=field_options.index("text") if "text" in all_keys else 0
        )
        image_key_selected = st.sidebar.selectbox("选择Image字段 (可选)", field_options, index=field_options.index("image") if "image" in all_keys else 0)

        # 最终使用的键名
        text_key = text_key_selected if text_key_selected != "不选择此字段" else ""
        image_key = image_key_selected if image_key_selected != "不选择此字段" else ""
        
        st.sidebar.markdown("---")
        st.sidebar.header("图片设置")
        image_scale_factor = st.sidebar.slider(
            "选择图片缩放倍数",
            min_value=0.1, max_value=2.0, value=1.0, step=0.1,
            help="调整图片显示大小。1.0为原始大小，大于1.0放大，小于1.0缩小。"
        )
        
        # --- 核心显示逻辑 ---
        # 预训练格式需要 Text 字段
        is_ready_to_display_main_content = bool(text_key)

        if not is_ready_to_display_main_content:
            st.info("请通过左侧边栏选择数据集，并完成数据格式及字段映射配置（至少选择Text字段），以便开始可视化。")
            return # 未配置完成，直接返回，不显示下方内容

        st.markdown("---") 

        # 调整字体大小
        st.markdown(f"<h3 style='font-size: 1.5rem; margin-bottom: 0.5rem;'>数据详情 (条目 {st.session_state.current_index + 1} / {len(data)})</h3>", unsafe_allow_html=True)

        current_item = data[st.session_state.current_index]

        display_text_content = ""
        
        # 真正的 tokenizer 启用状态取决于是否加载成功并且用户勾选了
        use_tokenizer_for_text = st.session_state.tokenizer is not None and st.session_state.tokenizer_enabled
        
        text_content_raw = current_item.get(text_key, "")

        if use_tokenizer_for_text:
            # 对于预训练数据，tokenizer 通常只用于编码/解码，而非 apply_chat_template
            # 这里简单演示编码再解码，如果实际需要更复杂的处理，可以修改
            try:
                # 假设预训练数据只需要简单的tokenize/decode，如果需要chat template则不适用
                # 简单地将文本传递给 tokenizer，然后解码回来，这可能不会改变文本，
                # 但表示 tokenizer 正在“处理”它
                tokenized_ids = st.session_state.tokenizer.encode(text_content_raw, add_special_tokens=False)
                display_text_content = st.session_state.tokenizer.decode(tokenized_ids, skip_special_tokens=True)
                st.info(f"文本已通过 Tokenizer 处理。原始长度: {len(text_content_raw)}，处理后长度: {len(display_text_content)}")
            except Exception as e:
                st.warning(f"使用 Tokenizer 处理文本失败: {e}。将显示原始文本。")
                display_text_content = text_content_raw
        else: 
            display_text_content = text_content_raw


        st.subheader("Text:") 
        if display_text_content:
            st.code(display_text_content, language="text")
        else:
            st.info("Text内容为空或未找到。请检查字段映射。") 


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