"""高级图像绘制插件

提供复杂图像生成与处理能力，通过策略模式支持多种绘图工作流。
支持的功能包括：GIF 动画生成、透明背景 PNG 图片生成等。
"""

from typing import Dict, Type

from nekro_agent.api.plugin import SandboxMethodType
from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core import logger

from .plugin import config, plugin
from .strategies import DrawingStrategy, GifGenerationStrategy, TransparentPngStrategy

# 策略注册表
STRATEGIES: Dict[str, Type[DrawingStrategy]] = {
    "gif_generation": GifGenerationStrategy,
    "transparent_png": TransparentPngStrategy,
}


@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="高级绘图魔法",
    description="执行高级绘图任务，如生成 GIF 动画、透明背景 PNG 等",
)
async def magic_draw(_ctx: AgentCtx, strategy_name: str, send_to_chat: bool = True, **kwargs) -> str:
    """执行高级绘图任务

    Args:
        strategy_name (str): 策略名称，例如 "gif_generation"（GIF动画）或 "transparent_png"（透明PNG）
        send_to_chat (bool): 是否自动发送生成的图片到聊天，默认 True（避免忘记发送）
        **kwargs: 策略所需的具体参数，请参考 AI 提示中注入的策略说明

    Returns:
        str: 生成的文件路径（沙盒路径）

    Examples:
        # 自动发送图片（推荐）
        magic_draw(strategy_name="gif_generation", content="...", style="pixel art")

        # 不自动发送（需要手动处理）
        magic_draw(strategy_name="transparent_png", content="...", send_to_chat=False)
    """
    if strategy_name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"未知的策略名称: {strategy_name}。可用策略: {available}")

    strategy_cls = STRATEGIES[strategy_name]
    strategy = strategy_cls(config)

    logger.info(f"开始执行高级绘图策略: {strategy_name}")
    result = await strategy.execute(_ctx, **kwargs)

    # 自动发送到聊天
    if send_to_chat:
        await _ctx.ms.send_image(_ctx.chat_key, result, ctx=_ctx)
        logger.info(f"图片已自动发送到聊天: {_ctx.chat_key}")

    return result


@plugin.mount_prompt_inject_method("inject_magic_draw_strategies")
async def inject_strategies(_ctx: AgentCtx) -> str:
    """注入可用策略说明到 AI 提示词中"""
    prompt_parts = ["## 高级绘图插件可用功能\n"]

    for name, strategy_cls in STRATEGIES.items():
        strategy = strategy_cls(config)
        description = strategy.get_description()
        prompt_parts.append(f"### 策略: `{name}`")
        prompt_parts.append(description)
        prompt_parts.append("")  # 空行分隔

    prompt_parts.append("\n**重要提示**：")
    prompt_parts.append("- 默认情况下，生成的图片会**自动发送到聊天**，无需手动发送")
    prompt_parts.append("- 如果需要对图片进行后续处理再发送，可设置 `send_to_chat=False`")
    prompt_parts.append("- 使用方式：`magic_draw(strategy_name='策略名', ...)`")

    return "\n".join(prompt_parts)


@plugin.mount_cleanup_method()
async def clean_up():
    """清理插件资源"""
    logger.info("高级绘图插件资源已清理")
