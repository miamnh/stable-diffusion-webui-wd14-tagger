import argparse
import logging
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


def preload_image_deduplicate_cluster_webui(
    parser: argparse.ArgumentParser
) -> None:
    try:
        import extensions.image_deduplicate_cluster_webui.preload  # noqa: F401
        # 这个只需要被导入就行了，导入时会自行执行某些操作L: import suffices.
    except Exception as e:
        logging.warning(f"Failed to import extensions.image_deduplicate_cluster_webui.preload: {e}")
    parser.add_argument("--disable_deduplicate_cluster", default=False,
                        action="store_true",
                        help="Disable image_deduplicate_cluster_webui")


def preload_dataset_tag_editor_standalone(
    parser: argparse.ArgumentParser
) -> None:

    parser.add_argument(
        "--device-id", type=int, help="CUDA Device ID to use interrogators",
        default=None
    )
    parser.add_argument("--disable_tag_editor", default=False,
                        action="store_true",
                        help="Disable dataset_tag_editor_standalone")


def preload_Gelbooru_API_Downloader(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--disable_Gelbooru_Downloader", default=False,
                        action="store_true",
                        help="Disable Gelbooru_API_Downloader")


# 注册的扩展名字列表: List of registered extension names
registered_extensions_preload = {
    "dataset_tag_editor_standalone": preload_dataset_tag_editor_standalone,
    "image_deduplicate_cluster_webui": preload_image_deduplicate_cluster_webui,
    "sd_webui_infinite_image_browsing": preload_sd_webui_infinite_image_browsing,
    "Gelbooru_API_Downloader": preload_Gelbooru_API_Downloader,
}
