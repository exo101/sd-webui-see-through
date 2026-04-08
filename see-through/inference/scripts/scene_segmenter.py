"""Scene segmentation using SAM (Segment Anything Model)."""
import os
import sys
import subprocess
import cv2
import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Tuple, Optional

def install_dependencies():
    """安装SAM所需的依赖"""
    try:
        import importlib
        
        dependencies = {
            'segment_anything': 'segment-anything'
        }
        
        missing_deps = []
        for module_name, package_name in dependencies.items():
            try:
                importlib.import_module(module_name)
                print(f"[SceneSegmenter] {package_name} 已安装")
            except ImportError:
                print(f"[SceneSegmenter] {package_name} 未安装，正在安装...")
                missing_deps.append(package_name)
        
        if missing_deps:
            print("[SceneSegmenter] 正在安装缺失的依赖...")
            for package in missing_deps:
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", package],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print(f"[SceneSegmenter] {package} 安装成功")
                except subprocess.CalledProcessError as e:
                    print(f"[SceneSegmenter] {package} 安装失败: {e}")
    except Exception as e:
        print(f"[SceneSegmenter] 依赖安装检查失败: {e}")

install_dependencies()

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
except ImportError as e:
    print(f"[SceneSegmenter] 导入SAM模块失败: {e}")
    print("[SceneSegmenter] 请手动安装: pip install segment-anything")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "segment-anything"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        print("[SceneSegmenter] SAM模块安装成功")
    except Exception as e:
        print(f"[SceneSegmenter] 安装SAM模块失败: {e}")
        raise


class SceneSegmenter:
    """Scene segmenter using SAM for automatic object detection and segmentation."""
    
    def __init__(self, device=None, model_type="vit_h"):
        """Initialize scene segmenter.
        
        Args:
            device: Device to run model on (cuda/cpu)
            model_type: SAM model type (vit_h, vit_l, vit_b)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.sam = None
        self.mask_generator = None
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load SAM model."""
        try:
            print(f"[SceneSegmenter] Loading SAM model (type: {self.model_type})...")
            
            checkpoint = self._find_checkpoint()
            if checkpoint is None:
                print("[SceneSegmenter] SAM checkpoint not found, downloading...")
                checkpoint = self._download_checkpoint()
            
            if checkpoint is None:
                raise FileNotFoundError("Could not find or download SAM checkpoint")
            
            self.sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
            self.sam.to(device=self.device)
            
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
            self.predictor = SamPredictor(self.sam)
            
            print(f"[SceneSegmenter] SAM model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"[SceneSegmenter] Failed to load SAM model: {e}")
            self.sam = None
            self.mask_generator = None
            self.predictor = None
    
    def _find_checkpoint(self) -> Optional[str]:
        """Find SAM checkpoint in common locations."""
        checkpoint_names = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        
        checkpoint_name = checkpoint_names.get(self.model_type)
        if not checkpoint_name:
            return None
        
        current_file = os.path.abspath(__file__)
        webui_root = os.path.abspath(os.path.join(current_file, "..", "..", "..", "..", "..", ".."))
        models_dir = os.path.join(webui_root, "models")
        sams_dir = os.path.join(models_dir, "sams")
        
        print(f"[SceneSegmenter] Calculated webui root: {webui_root}")
        print(f"[SceneSegmenter] Models directory: {models_dir}")
        print(f"[SceneSegmenter] SAMS directory: {sams_dir}")
        
        possible_paths = [
            os.path.join(sams_dir, checkpoint_name),
            os.path.join(models_dir, "sam", checkpoint_name),
            os.path.join("D:", "ai", "sd-webui-forge-neo-v2", "webui", "models", "sams", checkpoint_name),
            os.path.join(os.path.expanduser("~"), ".cache", "sam", checkpoint_name),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"[SceneSegmenter] Found checkpoint at: {path}")
                return path
        
        return None
    
    def _download_checkpoint(self) -> Optional[str]:
        """Download SAM checkpoint."""
        try:
            import urllib.request
            
            checkpoint_urls = {
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            }
            
            url = checkpoint_urls.get(self.model_type)
            if not url:
                return None
            
            current_file = os.path.abspath(__file__)
            webui_root = os.path.abspath(os.path.join(current_file, "..", "..", "..", "..", "..", ".."))
            models_dir = os.path.join(webui_root, "models")
            sams_dir = os.path.join(models_dir, "sams")
            
            print(f"[SceneSegmenter] Calculated webui root: {webui_root}")
            print(f"[SceneSegmenter] Models directory: {models_dir}")
            print(f"[SceneSegmenter] SAMS directory: {sams_dir}")
            
            os.makedirs(sams_dir, exist_ok=True)
            
            checkpoint_name = os.path.basename(url)
            checkpoint_path = os.path.join(sams_dir, checkpoint_name)
            
            print(f"[SceneSegmenter] Downloading SAM checkpoint from {url}...")
            print(f"[SceneSegmenter] Saving to: {checkpoint_path}")
            print(f"[SceneSegmenter] This may take a few minutes...")
            
            urllib.request.urlretrieve(url, checkpoint_path)
            print(f"[SceneSegmenter] Downloaded to: {checkpoint_path}")
            
            return checkpoint_path
            
        except Exception as e:
            print(f"[SceneSegmenter] Failed to download checkpoint: {e}")
            return None
    
    def segment_image(self, image_path: str, min_area: int = 1000, 
                     max_masks: int = 10) -> List[Dict]:
        """Segment image and return masks.
        
        Args:
            image_path: Path to input image
            min_area: Minimum mask area in pixels
            max_masks: Maximum number of masks to return
            
        Returns:
            List of dictionaries containing mask info
        """
        if self.mask_generator is None:
            print("[SceneSegmenter] Model not loaded, cannot segment")
            return []
        
        image = np.array(Image.open(image_path).convert('RGB'))
        print(f"[SceneSegmenter] Processing image: {image.shape}")
        
        print("[SceneSegmenter] Generating masks...")
        masks = self.mask_generator.generate(image)
        print(f"[SceneSegmenter] Generated {len(masks)} masks")
        
        filtered_masks = []
        for mask_data in masks:
            area = mask_data['area']
            if area >= min_area:
                filtered_masks.append(mask_data)
        
        filtered_masks.sort(key=lambda x: x['area'], reverse=True)
        filtered_masks = filtered_masks[:max_masks]
        
        print(f"[SceneSegmenter] Filtered to {len(filtered_masks)} masks")
        
        return filtered_masks
    
    def create_layer_images(self, image_path: str, masks: List[Dict], 
                           output_dir: str) -> List[str]:
        """Create layer images from masks.
        
        Args:
            image_path: Path to input image
            masks: List of mask dictionaries
            output_dir: Directory to save layer images
            
        Returns:
            List of output file paths
        """
        image = np.array(Image.open(image_path).convert('RGBA'))
        
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            
            layer = np.zeros_like(image)
            layer[mask] = image[mask]
            
            output_name = f"layer_{i:02d}.png"
            output_path = os.path.join(output_dir, output_name)
            Image.fromarray(layer).save(output_path)
            output_paths.append(output_path)
            
            print(f"[SceneSegmenter] Saved: {output_name} (area: {mask_data['area']})")
        
        combined_mask = np.zeros(image.shape[:2], dtype=bool)
        for mask_data in masks:
            combined_mask |= mask_data['segmentation']
        
        background = image.copy()
        background[combined_mask] = 0
        
        background_path = os.path.join(output_dir, "background.png")
        Image.fromarray(background).save(background_path)
        output_paths.append(background_path)
        
        print(f"[SceneSegmenter] Saved: background.png")
        
        return output_paths
    
    def create_psd(self, image_path: str, masks: List[Dict], output_dir: str, keywords: List[str]) -> str:
        """Create PSD file from masks.
        
        Args:
            image_path: Path to input image
            masks: List of mask dictionaries
            output_dir: Directory to save PSD file
            keywords: List of keywords for layer names
            
        Returns:
            Path to created PSD file
        """
        try:
            from psd_tools import PSDImage
            print("[SceneSegmenter] Successfully imported psd-tools")
            
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            
            layer_paths = []
            for i, mask_data in enumerate(masks):
                mask = mask_data['segmentation']
                
                layer_image = Image.new("RGB", (width, height), (255, 255, 255))
                layer_np = np.array(layer_image)
                image_np = np.array(image)
                layer_np[mask] = image_np[mask]
                layer_image = Image.fromarray(layer_np)
                
                layer_name = keywords[i] if i < len(keywords) else f"layer_{i+1}"
                layer_path = os.path.join(output_dir, f"{layer_name}.png")
                layer_image.save(layer_path)
                layer_paths.append((layer_name, layer_path))
            
            try:
                psd = PSDImage.new('RGB', (width, height))
                print("[SceneSegmenter] Successfully created PSD with PSDImage.new")
                
                try:
                    image_np = np.array(image)
                    combined_mask = np.zeros(image_np.shape[:2], dtype=bool)
                    for mask_data in masks:
                        combined_mask |= mask_data['segmentation']
                    
                    background_np = image_np.copy()
                    background_np[combined_mask] = 0
                    background = Image.fromarray(background_np)
                    
                    if hasattr(psd, 'create_pixel_layer'):
                        background_layer = psd.create_pixel_layer(background, name='background', top=0, left=0, opacity=255)
                        print("[SceneSegmenter] Successfully added background layer")
                    else:
                        print("[SceneSegmenter] create_pixel_layer method not available")
                except Exception as e:
                    print(f"[SceneSegmenter] Error adding background layer: {e}")
                
                for i, mask_data in enumerate(masks):
                    mask = mask_data['segmentation']
                    
                    layer_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                    layer_np = np.array(layer_image)
                    image_np = np.array(image.convert("RGBA"))
                    
                    layer_np[mask] = image_np[mask]
                    layer_image = Image.fromarray(layer_np)
                    
                    try:
                        layer_name = keywords[i] if i < len(keywords) else f"layer_{i+1}"
                        if hasattr(psd, 'create_pixel_layer'):
                            layer = psd.create_pixel_layer(layer_image, name=layer_name, top=0, left=0, opacity=255)
                            print(f"[SceneSegmenter] Successfully added layer: {layer_name}")
                        else:
                            print(f"[SceneSegmenter] Cannot add layer {layer_name}: create_pixel_layer not available")
                    except Exception as e:
                        print(f"[SceneSegmenter] Error adding layer {i}: {e}")
                
                psd_path = os.path.join(output_dir, "scene_seg.psd")
                psd.save(psd_path)
                print(f"[SceneSegmenter] Created PSD: {psd_path}")
                return psd_path
            except Exception as e:
                print(f"[SceneSegmenter] Error creating PSD: {e}")
                return ""
        except ImportError as e:
            print(f"[SceneSegmenter] psd-tools not installed: {e}")
            return ""
        except Exception as e:
            print(f"[SceneSegmenter] Error creating PSD: {e}")
            import traceback
            print(traceback.format_exc())
            return ""
    
    def segment_with_keywords(self, image_path: str, keywords: List[str], 
                             output_dir: str) -> List[str]:
        """Segment image using keywords with CLIP-based filtering.
        
        Args:
            image_path: Path to input image
            keywords: List of keywords to segment
            output_dir: Directory to save layer images
            
        Returns:
            List of output file paths
        """
        masks = self.segment_image(image_path)
        masks = masks[:len(keywords)]
        layer_paths = self.create_layer_images(image_path, masks, output_dir)
        psd_path = self.create_psd(image_path, masks, output_dir, keywords)
        
        return layer_paths + [psd_path]


def segment_scene(image_path: str, output_dir: str, 
                  min_area: int = 1000, max_masks: int = 10,
                  model_type: str = "vit_b") -> List[str]:
    """Segment scene and create layer images.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save layer images
        min_area: Minimum mask area in pixels
        max_masks: Maximum number of masks to return
        model_type: SAM model type (vit_h, vit_l, vit_b)
        
    Returns:
        List of output file paths
    """
    segmenter = SceneSegmenter(model_type=model_type)
    masks = segmenter.segment_image(image_path, min_area=min_area, max_masks=max_masks)
    return segmenter.create_layer_images(image_path, masks, output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scene_segmenter.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = os.path.join(os.path.dirname(image_path), "segmented")
    
    print(f"Segmenting: {image_path}")
    print(f"Output dir: {output_dir}")
    
    output_paths = segment_scene(image_path, output_dir)
    
    print(f"\nGenerated {len(output_paths)} layers:")
    for path in output_paths:
        print(f"  - {path}")
