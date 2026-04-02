from __future__ import annotations

import os
import sys
import logging
import subprocess
import gradio as gr
from modules import script_callbacks
from modules import scripts
from modules import shared
from modules.processing import StableDiffusionProcessing

# 创建logger
logger = logging.getLogger("See-Through")
logger.setLevel(logging.INFO)

# 添加See-Through项目路径
see_through_path = os.path.join(os.path.dirname(__file__), "..", "see-through")
sys.path.insert(0, see_through_path)

def install_dependencies():
    """自动安装依赖"""
    try:
        import importlib
        
        dependencies = {
            'psd_tools': 'psd-tools',
            'pycocotools': 'pycocotools'
        }
        
        missing_deps = []
        for module_name, package_name in dependencies.items():
            try:
                importlib.import_module(module_name)
                logger.info(f"{package_name} 已安装")
            except ImportError:
                logger.warning(f"{package_name} 未安装，正在安装...")
                missing_deps.append(package_name)
        
        if missing_deps:
            logger.info("正在安装缺失的依赖...")
            for package in missing_deps:
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", package],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    logger.info(f"{package} 安装成功")
                except subprocess.CalledProcessError as e:
                    logger.error(f"{package} 安装失败: {e}")
    except Exception as e:
        logger.error(f"依赖安装检查失败: {e}")

def on_app_started(demo, app):
    """WebUI启动时自动安装依赖"""
    logger.info("See-Through: 检查依赖...")
    install_dependencies()

class SeeThroughScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        see_through_path = os.path.join(os.path.dirname(__file__), "..", "see-through")
        self.output_dir = os.path.join(see_through_path, "workspace", "layerdiff_output")
        
    def title(self) -> str:
        return "See-Through Layer Decomposition"

    def show(self, is_img2img) -> scripts.AlwaysVisible:
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group(), gr.Accordion("See-Through 图层分离为PSD文件", open=False):
            
            with gr.Row():
                see_through_enabled = gr.Checkbox(label="启用See-Through", value=False)
            
            with gr.Row(visible=False) as see_through_options:
                with gr.Column():
                    gr.Markdown("### 图像输入")
                    input_mode = gr.Radio(
                        label="输入模式",
                        choices=["使用生成的图像", "上传图像", "指定图像路径"],
                        value="使用生成的图像"
                    )
                    image_upload = gr.Image(
                        label="上传图像",
                        type="pil",
                        visible=False
                    )
                    image_path = gr.Textbox(
                        label="图像路径",
                        placeholder="D:\\path\\to\\image.png",
                        visible=False
                    )
                    
                with gr.Column():
                    gr.Markdown("### 处理选项")
                    save_to_psd = gr.Checkbox(label="保存为PSD文件", value=True)
                    
                with gr.Column():
                    gr.Markdown("### 高级选项")
                    resolution = gr.Slider(
                        label="处理分辨率",
                        minimum=512,
                        maximum=1536,
                        value=1024,
                        step=64,
                        info="较低的分辨率可减少显存占用"
                    )
                    use_layerdiff3d = gr.Checkbox(label="使用LayerDiff 3D", value=True)
                    use_marigold = gr.Checkbox(label="使用Marigold深度估计", value=True)
                    use_sam = gr.Checkbox(label="使用SAM分割", value=True)
            
            with gr.Row(visible=False) as output_options:
                with gr.Column():
                    output_info = gr.Textbox(
                        label="处理状态",
                        value="等待处理...",
                        interactive=False
                    )
                    open_dir_btn = gr.Button("打开输出目录", variant="secondary")
            
            def update_input_mode(mode):
                return gr.update(visible=mode == "上传图像"), gr.update(visible=mode == "指定图像路径")
            
            def toggle_options(enabled):
                return gr.update(visible=enabled), gr.update(visible=enabled)
            
            def open_output_dir():
                output_dir = os.path.join(os.path.dirname(__file__), "..", "see-through", "workspace", "layerdiff_output")
                output_dir = os.path.abspath(output_dir)
                if os.path.exists(output_dir):
                    subprocess.run(['explorer', output_dir])
                    return f"已打开目录: {output_dir}"
                else:
                    os.makedirs(output_dir, exist_ok=True)
                    subprocess.run(['explorer', output_dir])
                    return f"已创建并打开目录: {output_dir}"
            
            input_mode.change(
                update_input_mode,
                inputs=[input_mode],
                outputs=[image_upload, image_path]
            )
            
            see_through_enabled.change(
                toggle_options,
                inputs=[see_through_enabled],
                outputs=[see_through_options, output_options]
            )
            
            open_dir_btn.click(
                open_output_dir,
                inputs=[],
                outputs=[output_info]
            )
            
            def process_image(mode, uploaded_image, path, save_psd, resolution, 
                            use_layerdiff, use_marigold_depth, use_sam_seg):
                input_image = None
                
                try:
                    logger.info("=" * 50)
                    logger.info("See-Through: 开始处理图像")
                    logger.info("=" * 50)
                    
                    # 确定输入图像路径
                    see_through_path = os.path.join(os.path.dirname(__file__), "..", "see-through")
                    temp_dir = os.path.join(see_through_path, "workspace", "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    logger.info(f"临时目录: {temp_dir}")
                    
                    if mode == "使用生成的图像":
                        logger.error("错误：请使用其他输入模式")
                        return "错误：请使用其他输入模式"
                    elif mode == "上传图像":
                        if not uploaded_image:
                            logger.error("错误：请上传图像")
                            return "错误：请上传图像"
                        # 保存上传的图像
                        temp_path = os.path.join(temp_dir, "uploaded_image.png")
                        uploaded_image.save(temp_path)
                        input_image = temp_path
                        logger.info(f"上传图像已保存: {temp_path}")
                    elif mode == "指定图像路径":
                        if not path:
                            logger.error("错误：请输入图像路径")
                            return "错误：请输入图像路径"
                        input_image = path
                        logger.info(f"使用指定图像: {path}")
                    else:
                        logger.error("错误：请选择有效的输入模式")
                        return "错误：请选择有效的输入模式"
                    
                    logger.info(f"输入图像: {input_image}")
                    logger.info(f"输出目录: {self.output_dir}")
                    logger.info(f"处理分辨率: {resolution}")
                    logger.info(f"保存PSD: {save_psd}")
                    
                    # 清理显存
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info(f"GPU显存已清理，当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    
                    # 构建命令
                    cmd = [
                        sys.executable,
                        os.path.join(see_through_path, "inference", "scripts", "inference_psd.py"),
                        "--srcp", input_image,
                        "--resolution", str(resolution)
                    ]
                    
                    if save_psd:
                        cmd.append("--save_to_psd")
                    
                    logger.info(f"执行命令: {' '.join(cmd)}")
                    logger.info("开始执行处理脚本...")
                    
                    # 执行处理 - 实时输出日志
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd=see_through_path,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    # 实时读取输出并记录到日志
                    all_output = []
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            line = line.rstrip()
                            all_output.append(line)
                            logger.info(f"[See-Through] {line}")
                    
                    process.wait()
                    output_text = '\n'.join(all_output)
                    
                    if process.returncode == 0:
                        logger.info("=" * 50)
                        logger.info("See-Through: 处理完成!")
                        logger.info(f"输出文件保存在: {self.output_dir}")
                        logger.info("=" * 50)
                        return f"处理成功！输出文件保存在: {self.output_dir}"
                    else:
                        logger.error("=" * 50)
                        logger.error("See-Through: 处理失败!")
                        logger.error(f"错误输出: {output_text}")
                        logger.error("=" * 50)
                        return f"处理失败：{output_text}"
                        
                except Exception as e:
                    logger.error("=" * 50)
                    logger.error(f"See-Through: 发生异常 - {str(e)}")
                    logger.error("=" * 50)
                    import traceback
                    logger.error(traceback.format_exc())
                    return f"错误：{str(e)}"
            
            process_btn = gr.Button("开始处理", variant="primary")
            process_btn.click(
                process_image,
                inputs=[input_mode, image_upload, image_path, save_to_psd, resolution, 
                       use_layerdiff3d, use_marigold, use_sam],
                outputs=[output_info]
            )
    
    def process(self, p: StableDiffusionProcessing, *args):
        if not self.enabled:
            return
        
        # 这里可以添加对生成图像的自动处理
        pass
    
    def after_component(self, component: gr.components.Component, **kwargs):
        pass

def on_ui_settings():
    shared.opts.add_option(
        key="see_through_enabled",
        info=shared.OptionInfo(
            default=False,
            label="启用See-Through插件",
            section=("see_through", "See-Through")
        )
    )
    see_through_path = os.path.join(os.path.dirname(__file__), "..", "see-through")
    default_output_dir = os.path.join(see_through_path, "workspace", "layerdiff_output")
    shared.opts.add_option(
        key="see_through_output_dir",
        info=shared.OptionInfo(
            default=default_output_dir,
            label="See-Through输出目录",
            section=("see_through", "See-Through")
        )
    )

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(on_app_started)
