import os
import sys

sys_path = sys.path.copy()

# 本文件所在位置，就是extensions文件夹所
# The location of this file is the extensions folder
extensions_dir = os.path.dirname(os.path.abspath(__file__))
