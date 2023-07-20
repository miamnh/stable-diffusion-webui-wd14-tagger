"""Module for the UI of extensions"""  # 扩展的UI模块
# 由 WSH032 慷慨提供: graciously provided by WSH032

import sys
import os
from typing import Tuple, Callable

import gradio as gr
from fastapi import FastAPI


DIR = os.path.dirname(os.path.abspath(__file__))


# 把与各扩展有关的 import 都放在各自的函数里面
# 因为有可能会有某个扩展缺失的清空，所以不需要导入该扩展的模块
# 否则会应某个扩展失效而照成整个程序无法启动
# Put the imports related to each extension in their respective functions
# Because there may be an extension that is missing and cleared, there is no
# need to import the module of the extension otherwise, the entire program will
# fail to start due to the failure of an extension
def ui_sd_webui_infinite_image_browsing() -> Tuple[Callable, Callable]:

    # for sd_webui_infinite_image_browsing working
    sys_path_copy = sys.path.copy()
    sys.path = [os.path.join(DIR, "sd_webui_infinite_image_browsing")] + sys_path_copy

    print("Build gallery")  # 构建图库

    # 扩展名字，即extensions文件夹中的文件夹名字:
    """ extension_name: the folder name, which is the extensions folder """

    from addons.sd_webui_infinite_image_browsing.scripts.iib.api \
        import send_img_path
    from addons.sd_webui_infinite_image_browsing.scripts.iib.tool \
        import read_info_from_image
    from addons.sd_webui_infinite_image_browsing.scripts.iib.logger \
        import logger
    from addons.sd_webui_infinite_image_browsing.app import AppUtils
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
        logger.info("Not_Implemented_Error, unsupported on standalone mode")

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
                # 独立运行时不起作用，logger.info一个未实现错误:
                # It does not work on standalone mode, logger.info an error
                btn.click(fn=not_implemented_error)

            img.change(on_img_change)
            img_update_trigger.click(img_update_func,
                                     outputs=[img, img_file_info])
        return demo

    def on_ui_tabs():
        return create_demo(), title, elem_id

    # 一定要注意原作者是否修改这个接口！！！:
    # Be sure to adapt to interface changes by the original author!!!
    def on_app_startd(_: gr.Blocks, app: FastAPI):
        return AppUtils().wrap_app(app)
    
    sys.path = sys_path_copy  # don't forget to recover sys.path

    return on_ui_tabs, on_app_startd

on_ui_tabs, on_app_startd = ui_sd_webui_infinite_image_browsing()
