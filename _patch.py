"""一次性补丁脚本：在服务器上运行，修复 solutions_service.py 视频编解码器"""
from pathlib import Path

BASE = Path("/root/wuyu/Vision_Platform")

# ===== solutions_service.py: 视频编解码器 H.264 优先 =====
p = BASE / "backend/modules/solutions/solutions_service.py"
c = p.read_text()

OLD = "out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))"
NEW = (
    "# 优先尝试 H.264（浏览器兼容性好），回退到 mp4v\n"
    "                for _fc in ('avc1', 'mp4v'):\n"
    "                    _w = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*_fc), fps, (width, height))\n"
    "                    if _w.isOpened():\n"
    "                        out = _w\n"
    "                        break\n"
    "                    _w.release()\n"
    "                else:\n"
    "                    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))"
)

if OLD in c:
    p.write_text(c.replace(OLD, NEW, 1))
    print("solutions_service.py: codec fixed OK")
else:
    print("solutions_service.py: already patched or pattern not found")

print("Patch done.")
