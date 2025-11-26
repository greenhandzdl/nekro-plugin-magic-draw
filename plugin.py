from pydantic import Field

from nekro_agent.api.plugin import ConfigBase, ExtraField, NekroPlugin

# 插件定义
plugin = NekroPlugin(
    name="🪄高级绘图魔法✨",
    module_name="magic_draw",
    description="提供复杂图像生成与处理能力封装，如 GIF 动画生成、透明背景 PNG 图片生成等。",
    version="1.0.0",
    author="KroMiose",
    url="https://github.com/KroMiose/nekro-plugin-magic-draw",
)


@plugin.mount_config()
class MagicDrawConfig(ConfigBase):
    """高级绘图插件配置"""

    BASIC_MODEL_GROUP: str = Field(
        default="default-chat",
        title="基础绘图模型组",
        description="用于辅助绘图任务的模型组，要求遵循指令能力较强，速度快。",
        json_schema_extra=ExtraField(
            ref_model_groups=True,
            model_type="draw",
        ).model_dump(),
    )

    ADVANCED_MODEL_GROUP: str = Field(
        default="default-chat",
        title="高级绘图模型组",
        description="用于复杂绘图任务的模型组，要求理解能力尽可能强，能处理精细需求。",
        json_schema_extra=ExtraField(
            ref_model_groups=True,
            model_type="draw",
        ).model_dump(),
    )

    STREAM_MODE: bool = Field(
        default=True,
        title="使用流式 API",
        description="启用流式模式可以避免长时间处理导致的超时错误，推荐开启。",
    )

    TIMEOUT: int = Field(
        default=300,
        title="请求超时时间",
        description="单位：秒。图像生成可能需要较长时间，建议设置较大的值。",
    )

    GIF_EDGE_FILTER_PIXELS: int = Field(
        default=4,
        title="GIF 帧边缘过滤像素数",
        description="处理 GIF 帧时过滤掉边缘的像素数，用于移除可能存在的分割栅格。默认 4 像素。",
    )

    DEBUG: bool = Field(
        default=False,
        title="调试模式",
        description="开启后，中间产物（如生成的过程 GIF 序列图）会被发送到聊天中。",
    )


# 挂载配置
config: MagicDrawConfig = plugin.get_config(MagicDrawConfig)
