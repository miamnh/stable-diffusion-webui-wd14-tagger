""" Extensions to include as modules """  # 包含的扩展作为模块
# 由 WSH032 慷慨提供: graciously provided by WSH032
import os
from typing import List
import sys
from collections import OrderedDict


from addons.extension_tools import extensions_dir
from addons.extensions_ui import (
    ui_sd_webui_infinite_image_browsing,
)

# 注册的扩展名字列表，键名请保证与文件夹同名
# List of registered extensions, please ensure key are as folder names
registered_extensions = OrderedDict(
    sd_webui_infinite_image_browsing=ui_sd_webui_infinite_image_browsing,
)


# TODO: 最好是调用某个扩展的时候再修改相应的sys.path，而不是一次性修改全部
# TODO: The best way is to modify the corresponding sys.path when calling an
# extension, rather than modifying all at once
def sys_path_for_extensions() -> List[str]:
    # 将每个扩展的所在的文件夹添加到sys.path中，以便各扩展可以正常import
    # Returns: List[str]: 改变之前的sys.path.copy()
    """Add the folder where each extension is located to sys.path so that each
    extension can be imported normally

    Returns:
        List[str]: sys.path.copy() before the change
    """
    sys_path = sys.path.copy()

    for extension in registered_extensions:
        # 让扩展在前面: Place the extension up front
        sys.path = [os.path.join(extensions_dir, extension)] + sys.path

    return sys_path
