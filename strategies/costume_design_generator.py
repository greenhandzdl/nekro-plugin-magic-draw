import uuid
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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


class CostumeDesignStrategy(DrawingStrategy):
    """角色概念设定图生成策略

    生成全景式角色深度概念分解图（Character Sheet），
    包含中心立绘、服装分层、表情集、材质特写和生活化物品展示。
    """

    def __init__(self, config: "MagicDrawConfig"):
        super().__init__(config)

    def get_description(self) -> str:
        return dedent(
            """
            ### 生成角色概念设定图 (Character Sheet)
            生成专业的角色深度概念分解图，适用于游戏美术、动漫设计、角色设定等场景。
            
            **核心特点**：
            - 全景式角色深度概念分解
            - 中心立绘 + 周边拆解元素环绕布局
            - 服装分层、表情集、材质特写
            - 关联物品、生活细节深度展示
            
            **参数**：
            - `reference_image`: **必需** 参考图片信息，包含：
              - `image_path`: 角色参考图片路径
              - `description`: 参考图说明（如"角色外观参考"）
            
            - `design_requirements`: (可选) 设定图的具体要求或补充说明
              - 描述想要强调的特征、风格定位、性格暗示
              - 如："强调角色的成熟职业感，展示其日常通勤风格"
              - 如："突出角色的神秘气质，添加神秘学相关物品"
              - 未提供时将生成标准的角色概念设定图
            
            - `style`: (可选) 整体美术风格，默认为"anime"
              - 可选值：anime（动漫）、realistic（写实）、concept art（概念艺术）等
            
            **调用示例**：
            ```python
            
            # 使用默认模板（自动分析角色特征）
            magic_draw(
                strategy_name="costume_design",
                reference_image={
                    "image_path": "shared/character.png",
                    "description": "角色参考"
                },
                style="concept art"
            )
            ```
            
            **输出内容**：
            - 中心角色全身立绘（高清补全细节）
            - 环绕式拆解展示：服装分层、内着细节、表情集、材质特写
            - 生活化物品展示：随身包袋内容、美妆护理、个人物件
            - 专业注释说明：材质标注、设计细节、物品说明
        """,
        ).strip()

    async def execute(
        self,
        ctx: AgentCtx,
        reference_image: Dict[str, str],
        design_requirements: Optional[str] = None,
        style: str = "anime",
        **kwargs,  # noqa: ARG002
    ) -> str:
        """执行角色概念设定图生成流程

        Args:
            ctx: Agent 上下文
            reference_image: 参考图片信息，包含 image_path 和 description
            design_requirements: 角色设定的补充要求描述
            style: 整体美术风格

        Returns:
            str: 生成的角色概念设定图文件路径（沙盒路径）
        """

        # 1. 验证并准备参考图片
        if not reference_image or not reference_image.get("image_path"):
            raise ValueError("必须提供参考图片（reference_image）")

        image_path = reference_image.get("image_path", "")
        ref_description = reference_image.get("description", "参考图片")

        img_data = await prepare_reference_image(
            image_path,
            ctx.chat_key,
            ctx.container_key or "",
        )

        if not img_data:
            raise ValueError(f"无法加载参考图片: {image_path}")

        logger.info(f"已加载参考图片: {ref_description}")

        # 2. 构造提示词
        prompt = self._build_prompt(
            ref_description=ref_description,
            design_requirements=design_requirements,
            style=style,
        )

        logger.info(f"角色概念设定图生成提示词: {prompt}")

        # 3. 调用高级模型生成图片
        model_group_name = self.config.ADVANCED_MODEL_GROUP
        if model_group_name not in global_config.MODEL_GROUPS:
            raise ValueError(f"未找到配置的模型组: {model_group_name}")

        model_group = global_config.MODEL_GROUPS[model_group_name]
        image_url_or_b64 = await generate_image_via_chat(
            model_group,
            prompt,
            timeout=self.config.TIMEOUT,
            reference_images=[(img_data, ref_description)],
            stream_mode=self.config.STREAM_MODE,
        )

        # 4. 处理生成的图片
        if image_url_or_b64.startswith("http"):
            image = await download_image(image_url_or_b64)
        else:
            image = decode_base64_image(image_url_or_b64)

        # 5. 保存并返回
        temp_image_path = f"/tmp/{uuid.uuid4()}.png"
        image.save(temp_image_path, "PNG")

        sandbox_image_file = await ctx.fs.mixed_forward_file(temp_image_path)
        Path(temp_image_path).unlink()

        logger.info("角色概念设定图生成完成")
        return sandbox_image_file

    def _build_prompt(
        self,
        ref_description: str,
        design_requirements: Optional[str],
        style: str,
    ) -> str:
        """构造角色概念设定图生成的提示词

        Args:
            ref_description: 参考图说明
            design_requirements: 用户提供的补充要求
            style: 美术风格

        Returns:
            完整的提示词
        """

        # 根据是否有补充要求构造不同的提示词段落
        if design_requirements:
            requirements_section = dedent(
                f"""
                ## 角色定位与补充要求
                {design_requirements}
                
                请基于上述要求深入挖掘角色的性格特质、生活方式和审美品味，并在拆解元素中体现这些特征。
                """,
            ).strip()
        else:
            requirements_section = dedent(
                """
                ## 角色定位
                基于参考图深度分析角色的核心特征、气质、可能的职业背景和生活方式，
                并通过拆解元素全方位展现角色的性格与审美品味。
                """,
            ).strip()

        return dedent(
            f"""
            【角色概念设定图制作任务 - Character Sheet】
            
            你是一位顶尖的游戏与动漫概念美术设计大师，擅长制作详尽的角色设定图。
            基于提供的参考图片（{ref_description}），创作一张**全景式角色深度概念分解图**。
            
            {requirements_section}
            
            ## 构图布局 (Layout)
            
            ### 中心位 (Center)
            - 放置角色的**全身立绘**（高清化并补齐细节）
            - 姿势自然优雅，清晰展示整体形象
            - 双足完整可见（赤足或穿鞋均可，根据角色设定）
            - 作为整个设计图的视觉锚点
            
            ### 环绕位 (Surroundings)
            - 在中心人物四周空白处，有序排列拆解后的各类元素
            - 使用**手绘箭头或引导线**连接周边元素与中心人物的对应部位
            - 保持整体布局的平衡感和专业性
            
            ## 拆解内容 (Deconstruction Details)
            
            ### 1. 服装分层 (Clothing Layers)
            - 将角色服装拆分为单品展示（外套、上衣、下装、鞋履等）
            - 如果是多层穿搭，展示脱下外层后的内层状态
            - **私密内着拆解**：独立展示内层衣物设计
              - 内衣裤（展示设计风格、蕾丝花纹、剪裁细节）
              - 丝袜/打底裤（展示透肉感、袜口设计、材质）
              - 其他贴身衣物（塑身衣、安全裤等）
            - 每个单品旁标注材质说明（如"柔软蕾丝"、"磨砂皮革"）
            
            ### 2. 表情集 (Expression Sheet)
            - 在画面角落绘制 **6-8 个头部特写**
            - 展示不同情绪：冷漠、害羞、惊讶、失神、开心、沉思、娇羞等
            - 每个表情都要清晰可辨，体现角色性格的多面性
            
            ### 3. 材质特写 (Texture & Zoom)
            - 选取 **2-3 个关键部位**进行放大特写
            - 服装材质：布料褶皱、皮革纹理、金属光泽、透明材质层次
            - 皮肤细节：肤质纹理、手部细节、足部细节
            - 物品质感：口红膏体的润泽感、皮革包包的颗粒纹理、化妆品的精致感
            
            ### 4. 关联物品 (Related Items) - 生活切片展示
            
            #### 随身包袋与内容物 (Bag & Contents)
            - 绘制角色的日常包包（手提包、通勤包、挎包等）
            - 将包包"打开"，展示散落在旁的物品：
              - 钱包、手机、钥匙、充电线
              - 纸巾、湿巾、镜子
              - 个人特色物品（反映性格的小物件）
            
            #### 美妆与护理 (Beauty & Grooming)
            - 常用化妆品组合展示：
              - 口红/唇釉（特写展示色号和膏体质感，如"常用色号 #520"）
              - 粉饼盒（带镜子，展示开合状态）
              - 香水瓶（展示瓶身设计和品牌暗示）
              - 护手霜、面霜等护理品
            - 美妆工具：化妆刷、睫毛夹、眉笔等
            
            #### 私密生活物件 (Lifestyle & Personal Items)
            - 根据角色性格设定，展示更私密的生活物品：
              - 私密日记本或笔记本
              - 常用药物/保健品（维生素、避孕药盒等）
              - 电子产品（耳机、平板、游戏机）
              - 个人兴趣相关物品（书籍、手办、饰品）
              - 更私密的物件（根据角色设定，以客观设计图视角呈现）
            
            #### 核心道具 (Signature Items)
            - 角色的标志性物品（武器、法器、工作道具、宠物等）
            - 展示多个角度或使用状态
            
            ## 视觉风格与注释 (Style & Annotations)
            
            ### 美术风格
            整体美术风格为：**{style}**
            - anime：日系动漫画风，线条清晰，色彩鲜明，赛璐璐风格
            - realistic：写实风格，注重真实感、光影和材质细节
            - concept art：概念艺术风格，艺术性强，设计感突出，手绘质感
            
            ### 背景
            - 使用**米黄色、羊皮纸或浅灰色纹理背景**
            - 营造专业设计手稿/概念设定图的氛围
            - 背景不抢主体，保持整洁
            
            ### 文字注释
            - 在每个拆解元素旁添加**模拟手写注释**（中英文结合）
            - 注释内容：
              - 材质说明（"Soft Lace 柔软蕾丝"、"Suede Leather 磨砂皮革"）
              - 品牌/型号暗示（"Favorite Shade #520"、"Daily Use 日常款"）
              - 设计特点（"Transparent Design 透明设计"、"Handmade 手工制作"）
            - 注释文字应简洁专业，字体模拟手写感
            
            ## 输出要求
            
            - **分辨率**：4K 高清输出 (3840×2160 或更高)
            - **透视准确**：所有拆解元素保持统一的透视关系
            - **光影统一**：所有元素处于同一光照环境，阴影方向一致
            - **注释清晰**：所有文字标注清晰可读，不遮挡主体
            - **细节丰富**：放大后每个元素都应有足够的细节表现
            - **整体协调**：画面布局平衡，视觉引导流畅
            
            ## 执行要点
            
            1. **深度分析**：首先分析参考图中角色的核心特征、穿着风格、潜在性格
            2. **元素提取**：提取一级元素（服装、表情、大道具）
            3. **深度脑补**：设计二级深度元素（她的内衣风格？包里装什么？独处时用什么？）
            4. **统一创作**：生成包含所有元素的组合图，确保风格统一
            5. **质量保证**：透视准确、光影统一、注释清晰、细节丰富
            
            请基于参考图和上述专业要求，创作一张高质量的**角色深度概念设定图 (Character Sheet)**。
        """,
        ).strip()


