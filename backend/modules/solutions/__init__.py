# 解决方案模块 - Solutions Module
"""
独立模块：提供 Ultralytics Solutions 功能
包括对象计数、热图、速度估算、距离计算、对象模糊、对象裁剪、队列管理等
"""

from .solutions_service import solutions_service

__all__ = ['solutions_service']
