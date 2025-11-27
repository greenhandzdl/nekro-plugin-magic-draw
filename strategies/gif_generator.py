import uuid
from collections import Counter
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from PIL import Image

from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core import logger
from nekro_agent.core.config import config as global_config

from ..utils import (
    create_gif_from_frames,
    decode_base64_image,
    download_image,
    generate_image_via_chat,
    get_average_color,
    prepare_reference_image,
    split_sprite_sheet,
)
from .base import DrawingStrategy

if TYPE_CHECKING:
    from ..plugin import MagicDrawConfig


class GifGenerationStrategy(DrawingStrategy):
    """GIF 动画生成策略"""

    def __init__(self, config: "MagicDrawConfig"):
        super().__init__(config)

    def get_description(self) -> str:
        return dedent(
            """
            ### 生成 GIF 动画
            生成像素风格或其他风格的**循环播放** GIF 动画，支持透明背景。
            
            **重要：你需要提供非常详细的动画描述！**
            
            **参数**:
            - `content`: **动画内容的极其详细描述**（关键！）
              - 必须描述清楚：主体是什么、做什么动作、动作如何变化
              - 必须分解动作过程：从什么状态开始 → 中间经过哪些变化 → 最后回到什么状态
              - 必须说明循环方式：如何首尾衔接形成循环
              - 示例（好）："一只橘猫，从站立姿势开始，逐渐抬起右前爪挥动，头部微微摆动，然后放下爪子回到站立姿势，形成循环挥手动画"
              - 示例（差）："一只猫在挥手" ❌ 太简略，无法生成流畅动画
            
            - `style`: (可选) 画面风格，默认为"pixel art"
            
            - `fps`: (可选) 帧率，单位：帧每秒（FPS）
              - 建议范围：6-15
              - 快速动作（如奔跑、挥动）：12-15 FPS
              - 中速动作（如走路、呼吸）：8-10 FPS
              - 慢速动作（如飘动、闪烁）：6-8 FPS
              - 未指定时使用系统默认值
            
            - `transparent_background`: (可选) 是否生成透明背景，默认为 False
              - True: 要求 AI 使用纯色背景，自动识别并移除背景色，生成透明 GIF
              - False: 不做背景处理，直接输出不透明 GIF 动画
            
            - `reference_images`: (可选) 参考图片列表
              - 每项包含 `image_path`（图片路径）和 `description`（该图片提供什么参考）
            
            **调用示例**:
            ```python
            # 详细描述动作过程
            magic_draw(
                strategy_name="gif_generation",
                content="一个像素风格的小人，完整的奔跑动画循环：第1-3帧左腿向前迈步，右臂向后摆；第4-6帧右腿向前迈步，左臂向后摆；第7-9帧左腿再次向前，重复循环。身体略有上下起伏，头部保持稳定，背景是城市街道",
                style="8-bit pixel art",
                fps=12  # 快速动作用较高帧率
            )
            
            # 使用参考图
            magic_draw(
                strategy_name="gif_generation",
                content="基于参考图中的角色形象，制作一个跳跃动画：从蹲下姿势开始蓄力（第1-3帧），然后向上跳起身体伸展（第4-7帧），到达最高点（第8帧），然后下落（第9-12帧），最后落地缓冲回到蹲姿（第13-16帧），形成循环",
                style="pixel art",
                fps=8,
                reference_images=[
                    {"image_path": "shared/character.png", "description": "角色的外观、服装、配色参考"}
                ]
            )
            
            # 生成透明背景的 GIF（慢速动画）
            magic_draw(
                strategy_name="gif_generation",
                content="一个燃烧的火焰效果，火焰从底部向上窜起，火苗摇曳，顶部火焰逐渐散开，然后循环",
                style="pixel art",
                fps=6,  # 慢速飘动效果
                transparent_background=True  # 自动移除背景色
            )
            ```
            
            **记住**：content 描述越详细，生成的动画越流畅！必须清楚说明每个阶段的动作变化。
        """,
        ).strip()

    async def execute(
        self,
        ctx: AgentCtx,
        content: str,
        style: str = "pixel art",
        fps: Optional[int] = None,
        transparent_background: bool = False,
        reference_images: Optional[List[Dict[str, Any]]] = None,
        **kwargs,  # noqa: ARG002
    ) -> str:
        """执行 GIF 生成流程

        Args:
            ctx: Agent 上下文
            content: 动画内容描述
            style: 画面风格
            fps: 帧率（每秒帧数），未指定时使用配置的默认值
            transparent_background: 是否生成透明背景（自动识别并移除背景色）
            reference_images: 参考图片列表，每项包含 image_path 和 description

        Returns:
            str: 生成的 GIF 文件路径（沙盒路径）
        """

        # 确定帧率
        actual_fps = fps if fps is not None else self.config.GIF_DEFAULT_FPS
        if actual_fps < 1 or actual_fps > 30:
            logger.warning(f"帧率 {actual_fps} 超出合理范围 (1-30)，使用默认值 {self.config.GIF_DEFAULT_FPS}")
            actual_fps = self.config.GIF_DEFAULT_FPS

        frame_duration_ms = int(1000 / actual_fps)
        logger.info(f"GIF 帧率: {actual_fps} FPS (每帧 {frame_duration_ms}ms)")

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

        # 2. 构造提示词
        ref_prefix = "基于提供的参考图片，" if ref_images_data else ""
        ref_style_note = "- 严格保持参考图片的视觉风格、色彩方案、角色特征" if ref_images_data else ""

        # 根据透明背景模式设置差异化参数
        if transparent_background:
            background_requirement = dedent(
                """
                **背景必须是纯色**：选择与主体颜色**强烈对比**的单一纯色作为背景
                - 主体暖色调 → 使用冷色背景（如深蓝、深绿）
                - 主体冷色调 → 使用暖色背景（如深红、橙色）
                - 主体多彩 → 使用深灰或深棕背景
                - 背景在**所有 16 帧**中保持**完全相同的纯色**
                - 背景无渐变、无纹理、无装饰、无阴影
            """,
            ).strip()
            summary_note = "，背景必须是纯色"
        else:
            background_requirement = "背景元素（环境、装饰物）在所有 16 帧中保持**完全一致**"
            summary_note = ""

        prompt = dedent(
            f"""
            【专业动画序列帧制作任务】
            
            {ref_prefix}创作一个 {style} 风格的**循环动画**。
            
            ## 动画内容要求
            {content}
            
            ## 输出格式规范
            输出一张**正方形画布**，精确划分为 4×4 共 16 个**完全相等**的格子：
            - 格子排列：从左到右、从上到下依次为第 1-16 格
            - 间距要求：格子之间**严格 0 像素间隙**，无分割线、无边框、无空隙
            - 尺寸要求：每个格子的宽高必须完全相同（画布尺寸除以 4）
            
            ## 动画帧制作要求（全部 16 格）
            
            ### 核心原则
            🎬 **这是逐帧动画，每一帧必须展示不同的动作画面！**
            🔄 **这是循环动画，第 16 帧必须能自然衔接回第 1 帧！**
            
            ### 动画连贯性要求
            1. **禁止重复帧**：
               - ❌ 严禁任意两帧画面完全相同或高度相似
               - ✅ 每一帧都必须展示动作的不同阶段
               - ✅ 相邻帧之间必须有清晰可见的变化
            
            2. **动作流畅过渡**：
               - 人物/物体的**位置、姿态、肢体角度**在每一帧都应不同
               - 动作变化应该**渐进式过渡**，不要跳跃式变化
               - 细节元素（头发、衣服、配饰等）也应随主体动作产生自然的次级运动
            
            3. **循环衔接设计**：
               - 第 1 帧：动作循环的起点状态
               - 第 2-15 帧：动作连续变化过程（严格按内容描述绘制）
               - 第 16 帧：回到起点状态（应能自然过渡到第 1 帧，但不是重复第 1 帧）
               - **测试标准**：将第 16 帧与第 1 帧连接时，动作应该流畅连贯无跳跃
            
            4. **视觉稳定性要求**：
               - {background_requirement}
               - 光照、色调、画面构图保持统一
               {ref_style_note}
            
            ### 质量标准
            - 每帧都是完整的独立画面（不是草图或分解图）
            - 动作变化清晰可辨，不模糊不含糊
            - 整体动画连贯流畅，符合物理规律和视觉习惯
            
            ## 技术规格
            - 画布：正方形 1:1 比例
            - 分辨率：每格尺寸 = 画布尺寸 ÷ 4
            - 帧率：{actual_fps} FPS（每帧 {frame_duration_ms}ms）
            - 对齐：像素级精准对齐
            
            请严格按照上述要求制作 16 帧**各不相同**且能**循环播放**的动画序列图{summary_note}。
        """,
        ).strip()

        logger.info(f"GIF 生成提示词: {prompt}")

        # 3. 调用高级模型生成图片
        model_group_name = self.config.ADVANCED_MODEL_GROUP
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

        # 5. 切割图片
        frames = split_sprite_sheet(image, rows=4, cols=4)
        if len(frames) != 16:
            raise ValueError(f"图片切割失败，期望 16 帧但获得 {len(frames)} 帧")

        # 6. 处理帧并生成 GIF
        edge_filter = self.config.GIF_EDGE_FILTER_PIXELS

        # 先裁剪边缘（移除可能的分割栅格）
        animation_frames = [self._filter_frame_edges(frame, edge_filter) for frame in frames]
        logger.info(f"已裁剪所有帧的边缘 {edge_filter} 像素")

        # 根据模式决定是否处理透明背景
        if transparent_background:
            # 透明模式：提取背景色并替换为透明
            transparency_color = self._extract_common_background_color(
                animation_frames,
                edge_filter,
            )
            if transparency_color:
                logger.info(
                    f"透明模式：从 {len(animation_frames)} 帧边缘提取背景色 RGB{transparency_color}",
                )
            else:
                logger.warning("未能提取到有效的背景色，将不使用透明处理")
                transparency_color = None
        else:
            # 非透明模式：不做任何背景处理，直接输出不透明 GIF
            transparency_color = None
            logger.info(f"非透明模式：直接输出不透明 GIF，共 {len(animation_frames)} 帧")

        # 7. 生成 GIF
        temp_gif_path = f"/tmp/{uuid.uuid4()}.gif"

        create_gif_from_frames(
            animation_frames,
            temp_gif_path,
            duration=frame_duration_ms,  # 根据帧率计算的每帧时长
            transparency_color=transparency_color,
            tolerance=10,  # 容差
        )

        # 8. 转换为沙盒文件并返回
        sandbox_gif_file = await ctx.fs.mixed_forward_file(temp_gif_path)
        Path(temp_gif_path).unlink()

        logger.info("GIF 动画生成完成")
        return sandbox_gif_file

    @staticmethod
    def _filter_frame_edges(frame: Image.Image, edge_pixels: int) -> Image.Image:
        """过滤帧的边缘像素，移除可能的分割栅格

        Args:
            frame: 原始帧
            edge_pixels: 要过滤的边缘像素数

        Returns:
            过滤边缘后的帧
        """
        if edge_pixels <= 0:
            return frame

        width, height = frame.size
        # 裁剪掉边缘
        crop_box = (
            edge_pixels,
            edge_pixels,
            width - edge_pixels,
            height - edge_pixels,
        )
        return frame.crop(crop_box)

    @staticmethod
    def _normalize_pixel(
        pixel: float | tuple[int, ...] | tuple[float, ...] | None,
    ) -> Tuple[int, int, int]:
        """将 getpixel 返回的各种类型统一转换为 RGB 元组

        Args:
            pixel: getpixel 返回值，可能是 int/float（灰度）或 tuple（RGB/RGBA）

        Returns:
            RGB 元组
        """
        if pixel is None:
            return (0, 0, 0)
        if isinstance(pixel, (int, float)):
            val = int(pixel)
            return (val, val, val)
        # tuple 类型
        if len(pixel) >= 3:
            r = int(pixel[0]) if not isinstance(pixel[0], int) else pixel[0]
            g = int(pixel[1]) if not isinstance(pixel[1], int) else pixel[1]
            b = int(pixel[2]) if not isinstance(pixel[2], int) else pixel[2]
            return (r, g, b)
        return (0, 0, 0)

    def _extract_common_background_color(
        self,
        frames: List[Image.Image],
        filtered_edge_pixels: int = 0,
        tolerance: int = 16,
    ) -> Optional[Tuple[int, int, int]]:
        """从所有帧的边缘提取共同的背景色

        Args:
            frames: 帧列表（PIL Image 对象，应该已经过边缘过滤）
            filtered_edge_pixels: 边缘过滤像素数（用于日志记录）
            tolerance: 颜色容差，小于此值视为同一颜色

        Returns:
            RGB 元组，如果未能提取到有效背景色则返回 None
        """
        logger.info(
            f"开始从 {len(frames)} 帧中提取共同背景色（已过滤边缘 {filtered_edge_pixels} 像素，容差={tolerance}）",
        )

        # 收集所有帧边缘像素
        all_edge_pixels: List[Tuple[int, int, int]] = []

        for idx, frame in enumerate(frames):
            if frame.mode != "RGB":
                frame = frame.convert("RGB")

            width, height = frame.size
            edge_pixels: List[Tuple[int, int, int]] = []

            # 提取四边边缘
            # 上边缘
            for x in range(width):
                pixel = frame.getpixel((x, 0))
                rgb_pixel = self._normalize_pixel(pixel)
                edge_pixels.append(rgb_pixel)

            # 下边缘
            for x in range(width):
                pixel = frame.getpixel((x, height - 1))
                rgb_pixel = self._normalize_pixel(pixel)
                edge_pixels.append(rgb_pixel)

            # 左边缘（不包括角点，避免重复）
            for y in range(1, height - 1):
                pixel = frame.getpixel((0, y))
                rgb_pixel = self._normalize_pixel(pixel)
                edge_pixels.append(rgb_pixel)

            # 右边缘（不包括角点，避免重复）
            for y in range(1, height - 1):
                pixel = frame.getpixel((width - 1, y))
                rgb_pixel = self._normalize_pixel(pixel)
                edge_pixels.append(rgb_pixel)

            all_edge_pixels.extend(edge_pixels)
            logger.debug(f"第 {idx + 1} 帧提取了 {len(edge_pixels)} 个边缘像素")

        logger.info(f"总共提取了 {len(all_edge_pixels)} 个边缘像素")

        if not all_edge_pixels:
            logger.warning("未提取到任何边缘像素")
            return None

        # 聚类：将相似的颜色归为一组
        color_clusters: Dict[Tuple[int, int, int], int] = {}

        for pixel in all_edge_pixels:
            # 查找是否有接近的聚类中心
            found_cluster = False
            for center in color_clusters:
                # 计算颜色距离
                distance = sum(abs(pixel[i] - center[i]) for i in range(3))
                if distance < tolerance:
                    color_clusters[center] += 1
                    found_cluster = True
                    break

            # 没有找到接近的聚类，创建新聚类
            if not found_cluster:
                color_clusters[pixel] = 1

        # 找出出现频率最高的聚类
        if not color_clusters:
            logger.warning("颜色聚类失败")
            return None

        most_common_color = max(color_clusters.items(), key=lambda x: x[1])
        bg_color, count = most_common_color

        percentage = (count / len(all_edge_pixels)) * 100
        logger.info(
            f"最常见背景色 RGB{bg_color}，出现 {count} 次（占边缘像素的 {percentage:.1f}%）",
        )

        # 验证：如果最常见的颜色占比太低，可能不是有效的背景色
        if percentage < 30:
            logger.warning(
                f"最常见颜色占比仅 {percentage:.1f}%，可能不是纯色背景，建议检查图像",
            )

        return bg_color
