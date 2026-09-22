from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool


@tool
def get_time()->datetime:
    """获取本地时间"""
    return datetime.now()