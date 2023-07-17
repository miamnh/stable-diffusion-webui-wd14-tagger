import argparse
# 由 WSH032 慷慨提供: graciously provided by WSH032

# 插件不应依赖于modules.cmd_args中的参数
# 插件所需的全部参数应该在这里注册
# 将会在modules.shared中被调用
# Plugins should not depend on arguments in modules.cmd_args
# All parameters required by the plugin should be registered here
# will be called in modules.shared


def preload_sd_webui_infinite_image_browsing(
    parser: argparse.ArgumentParser
) -> None:
    parser.add_argument("--sd_webui_config", type=str, default=None,
                        help="The path to the config file")
    parser.add_argument("--update_image_index", action="store_true",
                        help="Update the image index")
    parser.add_argument("--extra_paths", nargs="+", default=[],
                        help="Extra paths to use, added to Quick Move.")
    parser.add_argument("--disable_image_browsing", default=False,
                        action="store_true",
                        help="Disable sd_webui_infinite_image_browsing")

# 注册的扩展名字列表: List of registered extension names
registered_extensions_preload = {
    "sd_webui_infinite_image_browsing": preload_sd_webui_infinite_image_browsing,
}
