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
import torch
import numpy as np
from PIL import Image

logger = logging.getLogger("See-Through")
logger.setLevel(logging.INFO)

see_through_path = os.path.join(os.path.dirname(__file__), "..", "see-through")
see_through_path = os.path.abspath(see_through_path)
sys.path.insert(0, see_through_path)

inference_path = os.path.join(see_through_path, "inference")
inference_path = os.path.abspath(inference_path)
if inference_path not in sys.path:
    sys.path.insert(0, inference_path)
    logger.info(f"[Path] Added to sys.path: {inference_path}")
    
if not os.path.exists(inference_path):
    logger.error(f"[Path Error] Inference path does not exist: {inference_path}")
else:
    logger.info(f"[Path OK] Inference path exists: {inference_path}")

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
                    image_upload = gr.Image(
                        label="上传图像",
                        type="pil"
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
                        info="较低的分辨率可减少显存占用，提高处理速度"
                    )
                    num_inference_steps = gr.Slider(
                        label="推理步数",
                        minimum=10,
                        maximum=50,
                        value=30,
                        step=5,
                        info="较低的步数可加快速度，但可能降低质量（推荐20-30）"
                    )
                    batch_size = gr.Slider(
                        label="批处理大小",
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                        info="较小的批处理大小可减少显存占用"
                    )
                    seed = gr.Number(
                        label="随机种子",
                        value=-1,
                        precision=0,
                        info="设置随机种子以控制生成结果的可重复性（-1表示随机）"
                    )
                    
                with gr.Column():
                    gr.Markdown("### 分割模式")
                    segmentation_mode = gr.Radio(
                        label="分割模式",
                        choices=[
                            "人物分割",
                            "场景分割 (SAM)"
                        ],
                        value="人物分割",
                        info="选择要使用的分割模式，两种模式不能同时启用"
                    )
                    
                    with gr.Column(visible=True) as human_segmentation_options:
                        gr.Markdown("**人物分割选项**")
                        use_layerdiff3d = gr.Checkbox(label="使用LayerDiff 3D", value=True)
                        use_marigold = gr.Checkbox(label="使用Marigold深度估计", value=True)
                        use_sam = gr.Checkbox(label="使用SAM分割", value=True)
                        
                        with gr.Row():
                            enable_lr_split = gr.Checkbox(label="启用左右分离", value=False, info="将手、脚、耳等部位分为左右两部分")
                            enable_hair_split = gr.Checkbox(label="启用头发分割", value=False, info="将头发分为前发、后发、左发、右发")
                            enable_accessories = gr.Checkbox(label="处理饰品图层", value=False, info="优化饰品和配饰的透明度")
                            enable_equipment = gr.Checkbox(label="处理装备图层", value=False, info="优化武器、护甲等装备的显示")
                    
                    with gr.Column(visible=False) as scene_segmentation_options:
                        gr.Markdown("**场景分割选项**")
                        # 添加分割模式选择
                        scene_segmentation_mode = gr.Radio(
                            label="分割模式",
                            choices=[("限制数量", "limited"), ("分割所有目标", "all")],
                            value="limited",
                            interactive=True
                        )
                        scene_max_masks = gr.Number(
                            label="最大分割数量",
                            value=10,
                            precision=0,
                            interactive=True
                        )
                        scene_min_area = gr.Slider(
                            label="最小区域大小",
                            minimum=100,
                            maximum=10000,
                            value=1000,
                            step=100,
                            info="过滤小于此面积的分割区域（像素）"
                        )
                        scene_model_type = gr.Dropdown(
                            label="SAM 模型类型",
                            choices=["vit_b", "vit_l", "vit_h"],
                            value="vit_b",
                            info="vit_b: 小模型（~1.2GB）, vit_l: 中模型（~2.5GB）, vit_h: 大模型（~3.9GB）"
                        )
                    

                    
                with gr.Column():
                    gr.Markdown("### 内存优化选项")
                    use_nf4_quantization = gr.Checkbox(label="使用NF4量化 (8GB GPU)", value=True, info="使用4位量化模型权重，峰值显存~8GB")
                    cache_tag_embeds = gr.Checkbox(label="缓存文本嵌入", value=True, info="节省约2GB显存且无速度损失")
                
                with gr.Column():
                    gr.Markdown("### 模型路径与下载")
                    gr.Markdown("**人物分割模型位置**:")
                    gr.Markdown("- `sd-webui-forge-neo-v2\\webui\\models\\diffusers`")
                    gr.Markdown("  - `models--24yearsold--seethroughv0.0.1_marigold_nf4`")
                    gr.Markdown("  - `models--24yearsold--seethroughv0.0.2_layerdiff3d_nf4`")
                    gr.Markdown("")
                    gr.Markdown("**场景分割模型位置**:")
                    gr.Markdown("- `sd-webui-forge-neo-v2\\webui\\models\\sams`")
                    gr.Markdown("  - `sam_vit_b_01ec64.pth`")
                    gr.Markdown("  - `sam_vit_h_4b8939.pth`")
                    gr.Markdown("  - `sam_vitl0b3195.pth`")
                    gr.Markdown("")
                    gr.Markdown("**模型下载说明**:")
                    gr.Markdown("1. 所有模型在网盘中，请从群主网盘中获取")
                    gr.Markdown("2. 8g显存使用NF4量化模型，16g可使用全参数模型")
            
            with gr.Row(visible=False) as output_options:
                with gr.Column():
                    output_info = gr.Textbox(
                        label="处理状态",
                        value="等待处理...",
                        interactive=False
                    )
                    open_dir_btn = gr.Button("打开输出目录", variant="secondary")
            
            def toggle_options(enabled):
                return gr.update(visible=enabled), gr.update(visible=enabled)

            def toggle_semantic_mode(use_semantic):
                return gr.update(visible=use_semantic)

            def open_output_dir(segmentation_mode):
                """打开输出目录"""
                try:
                    see_through_path = os.path.join(os.path.dirname(__file__), "..", "see-through")
                    if segmentation_mode == "场景分割 (SAM)":
                        output_dir = os.path.join(see_through_path, "workspace", "scene_output")
                    else:
                        output_dir = os.path.join(see_through_path, "workspace", "layerdiff_output")
                    output_dir = os.path.abspath(output_dir)
                    if os.path.exists(output_dir):
                        subprocess.run(['explorer', output_dir])
                        return f"已打开目录: {output_dir}"
                    else:
                        os.makedirs(output_dir, exist_ok=True)
                        subprocess.run(['explorer', output_dir])
                        return f"已创建并打开目录: {output_dir}"
                except Exception as e:
                    error_msg = f"打开目录失败: {str(e)}"
                    print(error_msg)
                    return error_msg
            
            segmentation_mode.change(
                fn=lambda mode: {
                    human_segmentation_options: gr.Column(visible=mode == "人物分割"),
                    scene_segmentation_options: gr.Column(visible=mode == "场景分割 (SAM)")
                },
                inputs=[segmentation_mode],
                outputs=[human_segmentation_options, scene_segmentation_options]
            )
            
            see_through_enabled.change(
                toggle_options,
                inputs=[see_through_enabled],
                outputs=[see_through_options, output_options]
            )
            
            open_dir_btn.click(
                open_output_dir,
                inputs=[segmentation_mode],
                outputs=[output_info]
            )
            
            def process_image(uploaded_image, save_psd, resolution, num_inference_steps, batch_size, seed,
                            use_layerdiff, use_marigold_depth, use_sam_seg,
                            enable_lr_split, enable_hair_split, enable_accessories, enable_equipment,
                            use_nf4_quantization, cache_tag_embeds,
                            segmentation_mode, scene_segmentation_mode, scene_max_masks, scene_min_area, scene_model_type):
                input_image = None
                
                try:
                    logger.info("=" * 50)
                    logger.info("See-Through: 开始处理图像")
                    logger.info("=" * 50)
                    
                    if not uploaded_image:
                        logger.error("错误：请上传图像")
                        return "错误：请上传图像"

                    see_through_path = os.path.join(os.path.dirname(__file__), "..", "see-through")
                    temp_dir = os.path.join(see_through_path, "workspace", "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    logger.info(f"临时目录: {temp_dir}")
                    
                    import time
                    timestamp = int(time.time() * 1000)
                    unique_filename = f"uploaded_image_{timestamp}.png"
                    temp_path = os.path.join(temp_dir, unique_filename)
                    uploaded_image.save(temp_path)
                    input_image = temp_path
                    logger.info(f"上传图像已保存: {temp_path}")
                    
                    logger.info(f"输入图像: {input_image}")
                    logger.info(f"输出目录: {self.output_dir}")
                    logger.info(f"处理分辨率: {resolution}")
                    logger.info(f"推理步数: {num_inference_steps}")
                    logger.info(f"随机种子: {seed}")
                    logger.info(f"保存PSD: {save_psd}")
                    logger.info(f"使用NF4量化: {use_nf4_quantization}")
                    
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info(f"GPU显存已清理，当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    
                    try:
                        if segmentation_mode == "场景分割 (SAM)":
                            logger.info("[Scene Segmentation] 开始场景分割")
                            logger.info(f"[Scene Segmentation] 输入图像: {input_image}")
                            logger.info(f"[Scene Segmentation] 模型类型: {scene_model_type}")
                            logger.info(f"[Scene Segmentation] 最大分割数量: {scene_max_masks}")
                            logger.info(f"[Scene Segmentation] 最小区域大小: {scene_min_area}")
                            
                            import importlib.util
                            scene_segmenter_path = os.path.join(see_through_path, "inference", "scripts", "scene_segmenter.py")
                            spec = importlib.util.spec_from_file_location("scene_segmenter", scene_segmenter_path)
                            scene_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(scene_module)
                            
                            SceneSegmenter = scene_module.SceneSegmenter
                            
                            import time
                            timestamp = int(time.time() * 1000)
                            output_dir = os.path.join(see_through_path, "workspace", "scene_output", f"scene_{timestamp}")
                            os.makedirs(output_dir, exist_ok=True)
                            
                            segmenter = SceneSegmenter(model_type=scene_model_type)
                            
                            # 根据分割模式决定最大分割数量
                            if scene_segmentation_mode == "all":
                                # 分割所有目标，设置一个很大的数值
                                max_masks = 9999
                                logger.info("[Scene Segmentation] 分割所有目标模式")
                            else:
                                # 限制数量模式
                                try:
                                    max_masks = int(scene_max_masks)
                                    if max_masks <= 0:
                                        max_masks = 10  # 默认值
                                        logger.info("[Scene Segmentation] 最大分割数量必须大于0，使用默认值10")
                                except (ValueError, TypeError):
                                    max_masks = 10  # 默认值
                                    logger.info("[Scene Segmentation] 无效的最大分割数量，使用默认值10")
                                logger.info(f"[Scene Segmentation] 限制数量模式，最多 {max_masks} 个目标")
                            
                            masks = segmenter.segment_image(
                                image_path=input_image,
                                min_area=scene_min_area,
                                max_masks=max_masks
                            )
                            
                            output_paths = segmenter.create_layer_images(
                                image_path=input_image,
                                masks=masks,
                                output_dir=output_dir
                            )
                            
                            try:
                                keywords = [f"layer_{i+1}" for i in range(len(masks))]
                                psd_path = segmenter.create_psd(
                                    image_path=input_image,
                                    masks=masks,
                                    output_dir=output_dir,
                                    keywords=keywords
                                )
                                if psd_path:
                                    logger.info(f"PSD 文件已生成: {psd_path}")
                                else:
                                    logger.warning("PSD 文件生成失败（可能缺少 psd-tools）")
                            except Exception as e:
                                logger.error(f"生成 PSD 文件失败: {e}")
                                return f"场景分割成功，但生成 PSD 文件失败: {e}"
                            
                            logger.info("=" * 50)
                            logger.info("See-Through: 场景分割完成!")
                            logger.info(f"输出文件保存在: {output_dir}")
                            logger.info(f"生成的图层: {len(output_paths)}")
                            for path in output_paths:
                                logger.info(f"  - {os.path.basename(path)}")
                            logger.info(f"PSD 文件: {os.path.basename(psd_path)}")
                            logger.info("=" * 50)
                            return f"场景分割成功！\n输出文件保存在: {output_dir}\nPSD 文件: {psd_path}"
                        else:
                            if use_nf4_quantization:
                                script_name = "inference_psd_optimized.py"
                                
                                # 处理随机种子：-1表示随机生成
                                import random
                                actual_seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
                                logger.info(f"实际使用种子: {actual_seed}")
                                
                                cmd = [
                                    sys.executable,
                                    os.path.join(see_through_path, "inference", "scripts", script_name),
                                    "--srcp", input_image,
                                    "--resolution", str(resolution),
                                    "--num_inference_steps", str(num_inference_steps),
                                    "--seed", str(actual_seed),
                                    "--quant_mode", "nf4"
                                ]
                                
                                if save_psd:
                                    cmd.append("--save_to_psd")
                                
                                if enable_lr_split:
                                    cmd.append("--tblr_split")
                                
                                if cache_tag_embeds:
                                    cmd.append("--cache_tag_embeds")
                                
                                logger.info(f"使用优化版NF4量化流水线 (支持模型缓存)")
                                logger.info(f"深度估计分辨率: {resolution} (与主分辨率一致)")
                                logger.info(f"推理步数: {num_inference_steps}")
                                logger.info(f"随机种子: {actual_seed}")
                            else:
                                script_name = "inference_psd_optimized.py"
                                
                                # 处理随机种子：-1表示随机生成
                                import random
                                actual_seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
                                logger.info(f"实际使用种子: {actual_seed}")
                                
                                cmd = [
                                    sys.executable,
                                    os.path.join(see_through_path, "inference", "scripts", script_name),
                                    "--srcp", input_image,
                                    "--resolution", str(resolution),
                                    "--num_inference_steps", str(num_inference_steps),
                                    "--seed", str(actual_seed),
                                    "--quant_mode", "none"
                                ]
                                
                                if save_psd:
                                    cmd.append("--save_to_psd")
                                
                                if enable_lr_split:
                                    cmd.append("--tblr_split")
                                
                                if cache_tag_embeds:
                                    cmd.append("--cache_tag_embeds")
                                
                                logger.info(f"使用优化版标准流水线 (支持模型缓存)")
                                logger.info(f"推理步数: {num_inference_steps}")
                                logger.info(f"随机种子: {actual_seed}")
                            logger.info(f"执行命令: {' '.join(cmd)}")
                            logger.info(f"图层分割选项 - 左右分离: {enable_lr_split}, 头发分割: {enable_hair_split}, 饰品处理: {enable_accessories}, 装备处理: {enable_equipment}")
                            logger.info(f"内存优化选项 - NF4量化: {use_nf4_quantization}, 缓存文本嵌入: {cache_tag_embeds}")
                            logger.info(f"推理选项 - 步数: {num_inference_steps}")
                            logger.info("开始执行处理脚本...")
                            
                            timeout_seconds = 30 * 60
                            
                            process = subprocess.Popen(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                cwd=see_through_path,
                                bufsize=1,
                                universal_newlines=True
                            )
                            
                            all_output = []
                            start_time = time.time()
                            
                            try:
                                while process.poll() is None:
                                    if time.time() - start_time > timeout_seconds:
                                        process.kill()
                                        return f"处理超时（超过{timeout_seconds//60}分钟），请检查模型加载情况"
                                    
                                    line = process.stdout.readline()
                                    if line:
                                        line = line.rstrip()
                                        all_output.append(line)
                                        logger.info(f"[See-Through] {line}")
                                    else:
                                        time.sleep(0.1)
                            except Exception as e:
                                process.kill()
                                logger.error(f"处理过程中发生异常: {e}")
                                return f"处理过程中发生异常: {e}"
                            finally:
                                if process.poll() is None:
                                    process.kill()
                            
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
                        logger.error(f"See-Through: 处理失败! 错误: {str(e)}")
                        logger.error("=" * 50)
                        import traceback
                        logger.error(traceback.format_exc())
                        return f"处理失败: {str(e)}"
                        
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
                inputs=[image_upload, save_to_psd, resolution, num_inference_steps, batch_size, seed,
                       use_layerdiff3d, use_marigold, use_sam,
                       enable_lr_split, enable_hair_split, enable_accessories, enable_equipment,
                       use_nf4_quantization, cache_tag_embeds,
                       segmentation_mode, scene_segmentation_mode, scene_max_masks, scene_min_area, scene_model_type],
                outputs=[output_info]
            )
    
    def process(self, p: StableDiffusionProcessing, *args):
        if not self.enabled:
            return
        
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
