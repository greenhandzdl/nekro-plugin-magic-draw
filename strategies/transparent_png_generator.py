import uuid
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from PIL import Image

from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core import logger
from nekro_agent.core.config import config as global_config

from ..utils import (
    decode_base64_image,
    download_image,
    generate_image_via_chat,
    prepare_reference_image,
)
from .base import DrawingStrategy

if TYPE_CHECKING:
    from ..plugin import MagicDrawConfig


class TransparentPngStrategy(DrawingStrategy):
    """透明 PNG 生成策略

    通过指导 AI 使用对比色作为背景，然后自动提取并替换为透明背景。
    """

    def __init__(self, config: "MagicDrawConfig"):
        super().__init__(config)

    def get_description(self) -> str:
        return dedent(
            """
            ### 生成透明背景 PNG 图片
            使用此策略可以生成带透明背景的 PNG 图片，适合用于制作贴纸、图标等。
            
            **参数**:
            - `content`: 图片主体内容的详细描述。
            - `style`: (可选) 绘画风格，默认为"卡通风格"。
            - `size`: (可选) 图片尺寸，默认为"1024x1024"。
            - `reference_images`: (可选) 参考图片列表，每项包含 `image_path`（图片路径）和 `description`（图片描述说明）
            
            **示例**:
            ```python
            # 不使用参考图
            magic_draw(strategy_name="transparent_png", content="一只可爱的橘猫在伸懒腰", style="卡通风格")
            
            # 使用参考图
            magic_draw(
                strategy_name="transparent_png",
                content="一只可爱的猫咪在伸懒腰",
                style="卡通风格",
                reference_images=[
                    {"image_path": "shared/cat_style.png", "description": "猫咪风格参考"}
                ]
            )
            ```
        """,
        ).strip()

    async def execute(
        self,
        ctx: AgentCtx,
        content: str,
        style: str = "卡通风格",
        size: str = "1024x1024",
        reference_images: Optional[List[Dict[str, Any]]] = None,
        **kwargs,  # noqa: ARG002
    ) -> str:
        """执行透明 PNG 生成流程

        Args:
            ctx: Agent 上下文
            content: 图片主体内容描述
            style: 绘画风格
            size: 图片尺寸
            reference_images: 参考图片列表，每项包含 image_path 和 description

        Returns:
            str: 生成的透明 PNG 文件路径（沙盒路径）
        """

        # 1. 准备参考图片
        ref_images_data: List[Tuple[str, str]] = []
        if reference_images:
            for ref_img in reference_images:
                image_path = ref_img.get("image_path", "")
                description = ref_img.get("description", "")
                if image_path:
                    img_data = await prepare_reference_image(
                        image_path,
                        ctx.chat_key,
                        ctx.container_key or "",
                    )
                    if img_data:
                        ref_images_data.append((img_data, description))

        # 2. 构造特殊提示词
        prompt = self._build_prompt(content, style, size, bool(ref_images_data))
        logger.info(f"透明 PNG 生成提示词: {prompt}")

        # 3. 调用基础模型生成图片
        model_group_name = self.config.BASIC_MODEL_GROUP
        if model_group_name not in global_config.MODEL_GROUPS:
            raise ValueError(f"未找到配置的模型组: {model_group_name}")

        model_group = global_config.MODEL_GROUPS[model_group_name]
        image_url_or_b64 = await generate_image_via_chat(
            model_group,
            prompt,
            timeout=self.config.TIMEOUT,
            reference_images=ref_images_data if ref_images_data else None,
            stream_mode=self.config.STREAM_MODE,
        )

        # 4. 获取图片对象
        if image_url_or_b64.startswith("http"):
            image = await download_image(image_url_or_b64)
        else:
            image = decode_base64_image(image_url_or_b64)

        # DEBUG: 发送原始图片
        if self.config.DEBUG:
            raw_path = f"/tmp/{uuid.uuid4()}.png"
            image.save(raw_path)
            sandbox_raw_file = await ctx.fs.mixed_forward_file(raw_path)
            await ctx.ms.send_image(ctx.chat_key, sandbox_raw_file, ctx=ctx)
            Path(raw_path).unlink()

        # 5. 提取边缘背景色
        background_color = self._extract_edge_color(image)
        logger.info(f"检测到的背景色: RGB{background_color}")

        # 6. 将背景色替换为透明
        transparent_image = self._make_background_transparent(image, background_color)

        # 7. 保存并返回
        temp_png_path = f"/tmp/{uuid.uuid4()}.png"
        transparent_image.save(temp_png_path, "PNG")

        sandbox_png_file = await ctx.fs.mixed_forward_file(temp_png_path)
        Path(temp_png_path).unlink()

        logger.info("透明 PNG 生成完成")
        return sandbox_png_file

    def _build_prompt(self, content: str, style: str, size: str, has_reference: bool) -> str:
        """构建增强的提示词"""
        if has_reference:
            return dedent(
                f"""
                【专业透明背景图制作任务】
                
                基于提供的参考图片，绘制一张 {style} 风格的图片，内容：{content}

                ## 输出规格
                - 尺寸：{size}
                - 格式：适合后期透明处理的 PNG 图片

                ## 主体绘制要求
                1. **构图**：主体内容清晰、完整、居中放置
                2. **风格一致**：严格保持参考图片的视觉风格、色彩方案、细节特征
                3. **边缘处理**：主体边缘必须**清晰锐利**，禁止使用羽化、模糊、渐变边缘
                4. **完整性**：确保主体所有部分都在画面内，不要裁切关键部位
                5. **细节质量**：保持高精度细节，特别是边缘区域

                ## 背景处理要求（关键！）
                背景必须是**完全统一的纯色**，用于后续自动抠图处理：

                **背景色选择原则：**
                - 主体暖色调（红、橙、黄、棕）→ 使用冷色背景（推荐：#0047AB 深蓝、#006400 深绿）
                - 主体冷色调（蓝、青、紫）→ 使用暖色背景（推荐：#8B0000 深红、#FF6600 橙色）
                - 主体多彩/白色/浅色 → 使用深色背景（推荐：#2C2C2C 深灰、#654321 深棕）
                - 主体深色/黑色 → 使用亮色背景（推荐：#00FF00 亮绿、#FFFF00 黄色）

                **背景质量标准：**
                - ✅ 整个背景区域填充**单一纯色**
                - ✅ 无渐变、无纹理、无阴影、无装饰元素
                - ✅ 色值完全统一（每个像素的 RGB 值完全相同）
                - ✅ 与主体颜色形成**强烈对比**（色差 > 100）
                - ❌ 禁止背景与主体颜色相近或融合
                - ❌ 禁止背景中出现任何图案或细节

                ## 主体与背景分离
                - 主体边缘与背景之间必须有**清晰的颜色分界线**
                - 不要使用抗锯齿效果使边缘与背景混合
                - 主体的阴影、高光等效果应该包含在主体内部，不要延伸到背景

                ## 质量检查
                请确保最终图片满足：
                1. 主体清晰可辨，细节丰富
                2. 背景绝对纯色，颜色与主体对比明显
                3. 边缘锐利，便于后期抠图
                4. 整体构图合理，主体居中且完整

                输出符合上述所有要求的高质量图片。
            """,
            ).strip()

        return dedent(
            f"""
            【专业透明背景图制作任务】
            
            绘制一张 {style} 风格的图片，内容：{content}

            ## 输出规格
            - 尺寸：{size}
            - 格式：适合后期透明处理的 PNG 图片

            ## 主体绘制要求
            1. **构图**：主体内容清晰、完整、居中放置
            2. **风格表现**：充分体现 {style} 的视觉特点和艺术风格
            3. **边缘处理**：主体边缘必须**清晰锐利**，禁止使用羽化、模糊、渐变边缘
            4. **完整性**：确保主体所有部分都在画面内，不要裁切关键部位
            5. **细节质量**：保持高精度细节，特别是边缘区域

            ## 背景处理要求（关键！）
            背景必须是**完全统一的纯色**，用于后续自动抠图处理：

            **背景色选择原则：**
            - 主体暖色调（红、橙、黄、棕）→ 使用冷色背景（推荐：#0047AB 深蓝、#006400 深绿）
            - 主体冷色调（蓝、青、紫）→ 使用暖色背景（推荐：#8B0000 深红、#FF6600 橙色）
            - 主体多彩/白色/浅色 → 使用深色背景（推荐：#2C2C2C 深灰、#654321 深棕）
            - 主体深色/黑色 → 使用亮色背景（推荐：#00FF00 亮绿、#FFFF00 黄色）

            **背景质量标准：**
            - ✅ 整个背景区域填充**单一纯色**
            - ✅ 无渐变、无纹理、无阴影、无装饰元素
            - ✅ 色值完全统一（每个像素的 RGB 值完全相同）
            - ✅ 与主体颜色形成**强烈对比**（色差 > 100）
            - ❌ 禁止背景与主体颜色相近或融合
            - ❌ 禁止背景中出现任何图案或细节

            ## 主体与背景分离
            - 主体边缘与背景之间必须有**清晰的颜色分界线**
            - 不要使用抗锯齿效果使边缘与背景混合
            - 主体的阴影、高光等效果应该包含在主体内部，不要延伸到背景

            ## 质量检查
            请确保最终图片满足：
            1. 主体清晰可辨，细节丰富
            2. 背景绝对纯色，颜色与主体对比明显
            3. 边缘锐利，便于后期抠图
            4. 整体构图合理，主体居中且完整

            输出符合上述所有要求的高质量图片。
        """,
        ).strip()

    def _extract_edge_color(self, image: Image.Image, tolerance: int = 16) -> Tuple[int, int, int]:
        """提取图片边缘最常见的颜色

        Args:
            image: 输入图片
            tolerance: 颜色容差，小于此值视为同一颜色

        Returns:
            Tuple[int, int, int]: RGB 颜色值
        """
        # 确保是 RGB 模式
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size
        edge_pixels = []

        # 收集四条边的所有像素
        # 顶边
        for x in range(width):
            edge_pixels.append(image.getpixel((x, 0)))
        # 底边
        for x in range(width):
            edge_pixels.append(image.getpixel((x, height - 1)))
        # 左边（排除角落已采集的）
        for y in range(1, height - 1):
            edge_pixels.append(image.getpixel((0, y)))
        # 右边（排除角落已采集的）
        for y in range(1, height - 1):
            edge_pixels.append(image.getpixel((width - 1, y)))

        # 颜色聚类：将相似颜色归为一类
        color_clusters = []
        for pixel in edge_pixels:
            r, g, b = pixel
            # 查找是否已有相近颜色簇
            found = False
            for cluster in color_clusters:
                cr, cg, cb = cluster["center"]
                if abs(r - cr) <= tolerance and abs(g - cg) <= tolerance and abs(b - cb) <= tolerance:
                    cluster["pixels"].append(pixel)
                    found = True
                    break
            if not found:
                color_clusters.append({"center": pixel, "pixels": [pixel]})

        # 找出像素数量最多的簇
        if not color_clusters:
            # 兜底：返回图片四角的平均色
            corners: list[Tuple[int, int, int]] = []
            for x, y in [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]:
                pixel = image.getpixel((x, y))
                if isinstance(pixel, int):
                    corners.append((pixel, pixel, pixel))
                elif isinstance(pixel, tuple) and len(pixel) >= 3:
                    # 确保是 RGB 元组
                    r = int(pixel[0]) if not isinstance(pixel[0], int) else pixel[0]
                    g = int(pixel[1]) if not isinstance(pixel[1], int) else pixel[1]
                    b = int(pixel[2]) if not isinstance(pixel[2], int) else pixel[2]
                    corners.append((r, g, b))

            if not corners:
                return (0, 0, 0)

            avg_r = sum(c[0] for c in corners) // len(corners)
            avg_g = sum(c[1] for c in corners) // len(corners)
            avg_b = sum(c[2] for c in corners) // len(corners)
            return (avg_r, avg_g, avg_b)

        largest_cluster = max(color_clusters, key=lambda c: len(c["pixels"]))

        # 计算簇内颜色的平均值作为最终背景色
        pixels_in_cluster: list[Tuple[int, int, int]] = largest_cluster["pixels"]
        avg_r = sum(p[0] for p in pixels_in_cluster) // len(pixels_in_cluster)
        avg_g = sum(p[1] for p in pixels_in_cluster) // len(pixels_in_cluster)
        avg_b = sum(p[2] for p in pixels_in_cluster) // len(pixels_in_cluster)

        return (avg_r, avg_g, avg_b)

    def _make_background_transparent(
        self,
        image: Image.Image,
        bg_color: Tuple[int, int, int],
        tolerance: int = 16,
    ) -> Image.Image:
        """将背景色替换为透明

        Args:
            image: 输入图片
            bg_color: 要替换的背景色
            tolerance: 颜色容差

        Returns:
            Image.Image: 处理后的 RGBA 图片
        """
        # 转换为 RGBA
        rgba_image = image.convert("RGBA")
        width, height = rgba_image.size

        new_data: list[Tuple[int, int, int, int]] = []
        br, bg, bb = bg_color

        # 逐像素处理
        for y in range(height):
            for x in range(width):
                pixel = rgba_image.getpixel((x, y))
                if isinstance(pixel, tuple) and len(pixel) == 4:
                    r, g, b, a = int(pixel[0]), int(pixel[1]), int(pixel[2]), int(pixel[3])
                    # 判断是否在容差范围内
                    if abs(r - br) <= tolerance and abs(g - bg) <= tolerance and abs(b - bb) <= tolerance:
                        # 设为完全透明
                        new_data.append((255, 255, 255, 0))
                    else:
                        # 保持原样
                        new_data.append((r, g, b, a))

        rgba_image.putdata(new_data)
        return rgba_image
