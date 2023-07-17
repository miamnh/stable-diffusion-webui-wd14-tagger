"""Module for the UI of extensions"""  # 扩展的UI模块
# 由 WSH032 慷慨提供: graciously provided by WSH032
import os
from typing import Tuple, Union, Callable
import sys
from pathlib import Path

import gradio as gr
from fastapi import FastAPI

from extensions.extensions_tools import (
    extensions_dir,
    javascript_html,
    css_html,
    dir_path2html,
)
from modules import shared

# 把与各扩展有关的 import 都放在各自的函数里面
# 因为有可能会有某个扩展缺失的清空，所以不需要导入该扩展的模块
# 否则会应某个扩展失效而照成整个程序无法启动
# Put the imports related to each extension in their respective functions
# Because there may be an extension that is missing and cleared, there is no
# need to import the module of the extension otherwise, the entire program will
# fail to start due to the failure of an extension


def ui_image_deduplicate_cluster_webui(
    extension_name: str
) -> Tuple[Callable, Union[None, Callable], str, str]:

    print("Build recurring clustering")  # 构建查重聚类

    # 扩展名字，即extensions文件夹中的文件夹名字:
    """ extension_name: the folder name, which is the extensions folder """

    from extensions.image_deduplicate_cluster_webui \
        import cluster_images, deduplicate_images

    title = "Deduplicate-Cluster-Image"  # 显示在SD-WebUI中的名字: Displayed in
    elem_id = "Deduplicate-Cluster-Image"  # htlm id

    def create_demo():
        with gr.Blocks() as demo:
            with gr.TabItem("Deduplicate Images"):
                deduplicate_images.demo.render()
            with gr.TabItem("Cluster Images"):
                cluster_images.demo.render()
        return demo

    js_str = ""
    css_path = os.path.join(extensions_dir, extension_name, "style.css")
    css_str = css_html(css_path)

    def on_ui_tabs():
        return (create_demo(), title, elem_id)

    return on_ui_tabs, None, js_str, css_str


def ui_sd_webui_infinite_image_browsing(
    extension_name: str
) -> Tuple[Callable, Union[None, Callable], str, str]:

    print("Build gallery")  # 构建图库

    # 扩展名字，即extensions文件夹中的文件夹名字:
    """ extension_name: the folder name, which is the extensions folder """

    from extensions.sd_webui_infinite_image_browsing.scripts.iib.api \
        import send_img_path
    from extensions.sd_webui_infinite_image_browsing.scripts.iib.tool \
        import read_info_from_image
    from extensions.sd_webui_infinite_image_browsing.scripts.iib.logger \
        import logger
    from extensions.sd_webui_infinite_image_browsing.app import AppUtils
    from PIL import Image

    title = "Infinite image browsing"  # 显示在SD-WebUI中的名字
    elem_id = "infinite-image-browsing"  # htlm id

    def on_img_change():
        send_img_path["value"] = ""  # 真正收到图片改变才允许放行: Only allow
        # passage when the picture is really changed

    # 修改文本和图像，等待修改完成后前端触发粘贴按钮
    # 有时在触发后收不到回调，可能是在解析params。txt时除了问题删除掉就行了:
    # Modify the text and image, wait for the modification to be completed,
    # and the front end triggers the paste button
    def img_update_func():
        try:
            path = send_img_path.get("value")
            logger.info("img_update_func %s", path)
            img = Image.open(path)  # type: ignore
            info = read_info_from_image(img)
            return img, info
        except Exception as e:
            logger.error("img_update_func %s", e)

    def not_implemented_error():
        # 独立于SD-WebUI运行时不支持
        logger.info("Not_Implemented_Error, unsupported by SD-WebUI runtime")

    def create_demo():
        # ！！！注意，所有的elem_id都不要改，js依靠这些id来操作！！！
        """ elem_id should not be changed, js relies on these ids """
        with gr.Blocks(analytics_enabled=False) as demo:
            gr.HTML("error", elem_id="infinite_image_browsing_container_wrapper")
            # 以下是使用2个组件模拟粘贴过程: use 2 components to simulate paste
            img = gr.Image(
                type="pil",
                elem_id="iib_hidden_img",
            )
            img_update_trigger = gr.Button(
                "button",
                elem_id="iib_hidden_img_update_trigger",
            )
            img_file_info = gr.Textbox(elem_id="iib_hidden_img_file_info")

            for tab in ["txt2img", "img2img", "inpaint", "extras"]:
                btn = gr.Button(f"Send to {tab}", elem_id=f"iib_hidden_tab_{tab}")
                # 独立运行时后不起作用，logger.info一个未实现错误:
                # It does not work after running independently, logger.info an
                btn.click(fn=not_implemented_error)

            img.change(on_img_change)
            img_update_trigger.click(img_update_func,
                                     outputs=[img, img_file_info])
        return demo

    # js应该所在的文件夹: The folder where js should be
    js_dir = os.path.join(extensions_dir, extension_name, "javascript")
    # 该文件夹内所有js文件的绝对路径: The absolute path of all js files in the folder
    js_str = dir_path2html(
        dir=js_dir,
        ext=".js",
        html_func=javascript_html
    )

    css_path = os.path.join(extensions_dir, extension_name, "style.css")
    css_str = css_html(css_path)

    def on_ui_tabs():
        return create_demo(), title, elem_id

    # 一定要注意原作者是否修改这个接口！！！:
    # Be sure to adapt to interface changes by the original author!!!
    def on_app_start(_: gr.Blocks, app: FastAPI):
        app_utils = AppUtils(
            sd_webui_config=shared.cmd_opts.sd_webui_config,
            update_image_index=shared.cmd_opts.update_image_index,
            extra_paths=shared.cmd_opts.extra_paths,
        )
        app_utils.wrap_app(app)

    return on_ui_tabs, on_app_start, js_str, css_str


def ui_dataset_tag_editor_standalone(
    extension_name: str
) -> Tuple[Callable, Union[None, Callable], str, str]:

    print("Build tag editor")  # 构建tag editor

    # 扩展名字，即extensions文件夹中的文件夹名字:
    """ extension_name: the folder name, which is the extensions folder """

    # scripts中的脚本以相对引用的方式import，所以这里要添加它们所在的路径:
    # The scripts in scripts are imported in a relative reference manner,
    # so the path where they are located must be added here:
    scripts_dir = os.path.join(extensions_dir, extension_name, "scripts")
    sys.path = [scripts_dir] + sys.path

    from extensions.dataset_tag_editor_standalone.scripts.interface import (
        tab_main,
        tab_settings,
        versions_html,
        state,
        settings,
        paths,
        utilities,
        cleanup_tmpdr,
    )

    title = "Dataset Tag Editor"
    elem_id = "dataset_tag_editor_interface"

    state.begin()  # 设置成初始值，用于等候中断重启: Initial value for wait
    settings.load()  # 载入设置: Load settings
    paths.initialize()  # 创建路径: Create path

    state.temp_dir = (utilities.base_dir_path() / "temp").absolute()
    if settings.current.use_temp_files and settings.current.temp_directory != "":
        state.temp_dir = Path(settings.current.temp_directory)

    if settings.current.cleanup_tmpdir:
        cleanup_tmpdr()  # 清理上一次的临时文件: Clean temp files of last time

    def create_demo():
        with gr.Blocks(title="Dataset Tag Editor") as demo:
            with gr.Tab("Main"):
                tab_main.on_ui_tabs()
            with gr.Tab("Settings"):
                tab_settings.on_ui_tabs()

            gr.Textbox(elem_id="ui_created", value="", visible=False)

            # 其实 versions_html() 中有一定的异常处理，但是这里再加一层:
            # There is actually a certain amount of exception handling in
            # versions_html(), but here is another layer:
            try:
                footer = f'<div class="versions">{versions_html()}</div>'
            except Exception as e:
                footer = f'<div class="versions">Error: {e}</div>'

            gr.HTML(footer, elem_id="footer")
        return demo

    # js应该所在的文件夹: The folder where js should be
    js_dir = os.path.join(extensions_dir, extension_name, "javascript")
    # 该文件夹内所有js文件的绝对路径: The absolute path of all js files
    js_str = dir_path2html(dir=js_dir, ext=".js", html_func=javascript_html)

    # css应该所在的文件夹
    css_dir = os.path.join(extensions_dir, extension_name, "css")
    # 该文件夹内所有 css 文件的绝对路径: The absolute path of all css files
    css_str = dir_path2html(dir=css_dir, ext=".css", html_func=css_html)

    def on_ui_tabs():
        return create_demo(), title, elem_id

    return on_ui_tabs, None, js_str, css_str


def ui_Gelbooru_API_Downloader(extension_name: str):

    def not_implemented_error():
        extension_name = "Gelbooru_API_Downloader"
        # 的WebUI界面尚未实现:
        print(f"Not_Implemented_Error: WebUI extension {extension_name}")
        download_ps1_path = os.path.join(extensions_dir, extension_name,
                                         "run_download_images_coroutine.ps1")
        if os.path.exists(download_ps1_path):
            # 如果要使用下载功能，请使用:
            print(f"To download, please use:\n{download_ps1_path}")
    not_implemented_error()

    return None, None, "", ""
