from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from nekro_agent.api.schemas import AgentCtx

if TYPE_CHECKING:
    from ..plugin import MagicDrawConfig


class DrawingStrategy(ABC):
    """绘图策略基类"""

    def __init__(self, config: "MagicDrawConfig"):
        """初始化策略

        Args:
            config: 插件配置对象
        """
        self.config = config

    @abstractmethod
    async def execute(self, ctx: AgentCtx, **kwargs) -> str:
        """执行绘图策略

        Args:
            ctx: 代理上下文
            **kwargs: 策略所需的参数

        Returns:
            str: 执行结果描述或生成的文件路径
        """

    @abstractmethod
    def get_description(self) -> str:
        """获取策略描述

        Returns:
            str: 策略的描述信息，用于告知 AI 如何使用该策略
        """
