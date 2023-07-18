""" This module contains the ui for the tagger tab. """
from typing import Dict, Tuple, List, Callable
import gradio as gr
from gradio import routes
from PIL import Image
from packaging import version
import os
import sys

from modules import ui
from modules import generation_parameters_copypaste as parameters_copypaste
from modules import shared
from modules.paths import script_path

from tensorflow import __version__ as tf_version
from webui import wrap_gradio_gpu_call
from tagger import utils
from tagger.interrogator import Interrogator as It
from tagger.uiset import IOData, QData, ItRetTP
from addons.extensions import registered_extensions
from addons.extension_tools import javascript_html


# issues:
# exclude tags are not excluded from the tags .txt files
# the tags are not written to the images

def unload_interrogators() -> List[str]:
    unloaded_models = 0

    for i in utils.interrogators.values():
        if i.unload():
            unloaded_models = unloaded_models + 1

    return (f'Successfully unload {unloaded_models} model(s)',)


def on_interrogate(name: str, inverse=False) -> ItRetTP:
    if It.input["input_glob"] == '':
        return (None, None, None, 'No input directory selected')

    interrogator: It = next((i for i in utils.interrogators.values() if
                             i.name == name), None)
    if interrogator is None:
        return (None, None, None, f"'{name}': invalid interrogator")

    QData.inverse = inverse
    return interrogator.batch_interrogate()


def on_inverse_interrogate(name: str) -> Tuple[str, Dict[str, float], str]:
    ret = on_interrogate(name, True)
    return (ret[0], ret[2], ret[3])


def on_gallery() -> List:
    return It.get_image_dups()


def on_interrogate_image(image: Image, name: str) -> ItRetTP:

    # hack brcause image interrogaion occurs twice
    # It.odd_increment = It.odd_increment + 1
    # if It.odd_increment & 1 == 1:
    #    return (None, None, None, '')

    if image is None:
        return (None, None, None, 'No image selected')
    interrogator: It = next((i for i in utils.interrogators.values() if
                             i.name == name), None)
    if interrogator is None:
        return (None, None, None, f"'{name}': invalid interrogator")

    return interrogator.interrogate_image(image)


def move_selection_to_input(
    tag_search_filter: str,
    name: str,
    field: str
) -> Tuple[str, str, Dict[str, float], Dict[str, float], str]:
    """ moves the selected to the input field """
    if It.output is None:
        return (None, None, None, None, '')

    filt = {(k, v) for k, v in It.output[2].items() if tag_search_filter in k}
    if len(filt) == 0:
        return (None, None, None, None, '')

    add = set(dict(filt).keys())
    if It.input[field] != '':
        add = add.union({x.strip() for x in It.input[field].split(',')})

    It.input[field] = ', '.join(add)

    ret = on_interrogate(name, QData.inverse)
    return (It.input[field],) + ret


def move_selection_to_keep(
    tag_search_filter: str, name: str
) -> Tuple[str, str, Dict[str, float], str]:
    ret = move_selection_to_input(tag_search_filter, name, "keep")
    # ratings are not displayed on this tab
    return ('',) + ret[:2] + ret[3:]


def move_selection_to_exclude(
    tag_search_filter: str, name: str
) -> Tuple[str, str, Dict[str, float], Dict[str, float], str]:
    return ('',) + move_selection_to_input(tag_search_filter, name, "exclude")


def on_tag_search_filter_change(
    part: str
) -> Tuple[str, Dict[str, float], str]:
    if It.output is None:
        return (None, None, '')
    if len(part) < 2:
        return (It.output[0], It.output[2], '')
    tags = dict(filter(lambda x: part in x[0], It.output[2].items()))
    return (', '.join(tags.keys()), tags, '')


def reload_javascript(
    GradioTemplateResponseOriginal: Callable, js: str, css: str
):
    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    return template_response


def amend_ui() -> Tuple[gr.Blocks, List[Callable]]:

    callbacks_ui_tabs = []
    callbacks_app_started = []
    js_str = ""
    css_str = ""

    # 各扩展正常工作需要将它们所在的文件夹加入sys.path，这里返回的是修改前的sys.path
    # Each extension adds folders to sys.path, but keep a path before
    # sys_path = sys_path_for_extensions()

    # global registered_extensions
    # 拷贝下，避免修改了原字典
    create_ui_extensions_dict_copy = registered_extensions.copy()

    for extension_name, extension_ui in create_ui_extensions_dict_copy.items():

        on_ui_tabs, on_app_started, extension_js_str, extension_css_str = extension_ui(extension_name)
        if on_ui_tabs:
            callbacks_ui_tabs.append(on_ui_tabs)
        if on_app_started:
            callbacks_app_started.append(on_app_started)
        if extension_js_str:
            js_str += extension_js_str
        if extension_css_str:
            css_str += extension_css_str

    # modules_dir = os.path.join(shared.script_path, "modules")
    script_js = os.path.join(script_path, "script.js")  # webui的js
    js_str = javascript_html(script_js) + js_str  # webui的js要放在前面

    # 先修改gradio，再创建demo，像SD-WebUI那样
    routes.templates.TemplateResponse = reload_javascript(
        routes.templates.TemplateResponse,
        js=js_str,
        css=css_str
    )

    with gr.Blocks() as demo:
        print("开始渲染UI")
        for interface, label, ifid in interfaces_tuple_list:
            with gr.Tab(label, id=ifid, elem_id=f"tab_{ifid}"):
                interface.render()

    if recover_sys_path:
        sys.path = sys_path  # 恢复sys.path

    return demo, callbacks_app_started

def on_ui_tabs():
    """ configures the ui on the tagger tab """
    demo, callbacks = amend_ui()
    # If checkboxes misbehave you have to adapt the default.json preset

    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):

                # input components
                with gr.Tabs():
                    tab_single_process = gr.TabItem(label='Single process')
                    tab_batch_from_directory = gr.TabItem(
                        label='Batch from directory'
                    )
                    with tab_single_process:
                        image = gr.Image(
                            label='Source',
                            source='upload',
                            interactive=True,
                            type="pil"
                        )
                        image_submit = gr.Button(
                            value='Interrogate image',
                            variant='primary'
                        )

                    with tab_batch_from_directory:
                        input_glob = utils.preset.component(
                            gr.Textbox,
                            value=It.input["input_glob"],
                            label='Input directory - See also settings tab.',
                            placeholder='/path/to/images or to/images/**/*'
                        )
                        output_dir = utils.preset.component(
                            gr.Textbox,
                            value=It.input["output_dir"],
                            label='Output directory',
                            placeholder='Leave blank to save images '
                                        'to the same path.'
                        )

                        batch_submit = gr.Button(
                            value='Interrogate',
                            variant='primary'
                        )
                        with gr.Row(variant='compact'):
                            with gr.Column(variant='panel'):
                                large_query = utils.preset.component(
                                    gr.Checkbox,
                                    label='huge batch query (TF 2.10, '
                                    'experimental)',
                                    value=False,
                                    interactive=version.parse(tf_version) ==
                                    version.parse('2.10')
                                )
                            with gr.Column(variant='panel'):
                                save_tags = utils.preset.component(
                                    gr.Checkbox,
                                    label='Save to tags files',
                                    value=True
                                )

                info = gr.HTML(
                    label='Info',
                    interactive=False,
                    elem_classes=['info']
                )

                # preset selector
                with gr.Row(variant='compact'):
                    available_presets = utils.preset.list()
                    selected_preset = gr.Dropdown(
                        label='Preset',
                        choices=available_presets,
                        value=available_presets[0]
                    )

                    save_preset_button = gr.Button(
                        value=ui.save_style_symbol
                    )

                    ui.create_refresh_button(
                        selected_preset,
                        lambda: None,
                        lambda: {'choices': utils.preset.list()},
                        'refresh_preset'
                    )

                # interrogator selector
                with gr.Column():
                    with gr.Row(variant='compact'):
                        def refresh():
                            utils.refresh_interrogators()
                            return sorted(x.name for x in utils.interrogators
                                                               .values())
                        interrogator_names = refresh()
                        interrogator = utils.preset.component(
                            gr.Dropdown,
                            label='Interrogator',
                            choices=interrogator_names,
                            value=(
                                None
                                if len(interrogator_names) < 1 else
                                interrogator_names[-1]
                            )
                        )

                        ui.create_refresh_button(
                            interrogator,
                            lambda: None,
                            lambda: {'choices': refresh()},
                            'refresh_interrogator'
                        )

                    unload_all_models = gr.Button(
                        value='Unload all interrogate models'
                    )
                add_tags = utils.preset.component(
                    gr.Textbox,
                    label='Additional tags (comma split)',
                    elem_id='additional-tags'
                )
                with gr.Row(variant='compact'):
                    with gr.Column(variant='compact'):
                        threshold = utils.preset.component(
                            gr.Slider,
                            label='Weight threshold',
                            minimum=0,
                            maximum=1,
                            value=QData.threshold
                        )
                        cumulative = utils.preset.component(
                            gr.Checkbox,
                            label='Combine interrogations',
                            value=False
                        )
                        search_tags = utils.preset.component(
                            gr.Textbox,
                            label='Search tag, .. ->',
                            elem_id='search-tags'
                        )
                        keep_tags = utils.preset.component(
                            gr.Textbox,
                            label='Kept tag, ..',
                            elem_id='keep-tags'
                        )
                    with gr.Column(variant='compact'):
                        tag_frac_threshold = utils.preset.component(
                            gr.Slider,
                            label='Mininmum fraction for tags',
                            minimum=0,
                            maximum=1,
                            value=QData.tag_frac_threshold,
                        )
                        unload_after = utils.preset.component(
                            gr.Checkbox,
                            label='Unload model after running',
                            value=False
                        )
                        replace_tags = utils.preset.component(
                            gr.Textbox,
                            label='-> Replace tag, ..',
                            elem_id='replace-tags'
                        )
                        exclude_tags = utils.preset.component(
                            gr.Textbox,
                            label='Exclude tag, ..',
                            elem_id='exclude-tags'
                        )

            # output components
            with gr.Column(variant='panel'):
                with gr.Row(variant='compact'):
                    with gr.Column(variant='compact'):
                        mv_selection_to_keep = gr.Button(
                            value='Move visible tags to keep tags',
                            variant='secondary'
                        )
                        mv_selection_to_exclude = gr.Button(
                            value='Move visible tags to exclude tags',
                            variant='secondary'
                        )
                    with gr.Column(variant='compact'):
                        tag_search_selection = utils.preset.component(
                            gr.Textbox,
                            label='string search selected tags'
                        )
                with gr.Tabs():
                    tab_include = gr.TabItem(label='Ratings and included tags')
                    tab_discard = gr.TabItem(label='Excluded tags')
                    tab_gallery = gr.TabItem(label='Gallery')
                    with tab_include:
                        # clickable tags to populate excluded tags
                        tags = gr.HTML(
                            label='Tags',
                            elem_id='tags',
                        )

                        with gr.Row():
                            parameters_copypaste.bind_buttons(
                                parameters_copypaste.create_buttons(
                                    ["txt2img", "img2img"],
                                ),
                                None,
                                tags
                            )
                        rating_confidences = gr.Label(
                            label='Rating confidences',
                            elem_id='rating-confidences',
                        )
                        tag_confidences = gr.Label(
                            label='Tag confidences',
                            elem_id='tag-confidences',
                        )
                    with tab_discard:
                        # clickable tags to populate keep tags
                        discarded_tags = gr.HTML(
                            label='Tags',
                            elem_id='tags',
                        )
                        excluded_tag_confidences = gr.Label(
                            label='Excluded Tag confidences',
                            elem_id='discard-tag-confidences',
                        )
                    with tab_gallery:
                        # Note: this elem_id is dapted inmy fork of the
                        # infinite image browsing component to avoid conflicts
                        # with the same extension on a webui tab.
                        gr.HTML("error", elem_id="tab_iib_container")
                        gallery = gr.Gallery(
                            label='Gallery',
                            elem_id='gallery',
                        ).style(
                            columns=[2],
                            rows=[8],
                            object_fit="contain",
                            height="auto"
                        )

        tab_gallery.select(fn=on_gallery,
                           inputs=[],
                           outputs=[gallery])

        tab_include.select(fn=wrap_gradio_gpu_call(on_interrogate),
                           inputs=[interrogator],
                           outputs=[tags, rating_confidences, tag_confidences,
                                    info])

        tab_discard.select(fn=wrap_gradio_gpu_call(on_inverse_interrogate),
                           inputs=[interrogator],
                           outputs=[discarded_tags, excluded_tag_confidences,
                                    info])

        mv_selection_to_keep.click(
            fn=wrap_gradio_gpu_call(move_selection_to_keep),
            inputs=[tag_search_selection, interrogator],
            outputs=[tag_search_selection, keep_tags, discarded_tags,
                     excluded_tag_confidences, info])

        mv_selection_to_exclude.click(
            fn=wrap_gradio_gpu_call(move_selection_to_exclude),
            inputs=[tag_search_selection, interrogator],
            outputs=[tag_search_selection, exclude_tags, tags,
                     rating_confidences, tag_confidences, info])

        cumulative.input(fn=It.flip('cumulative'), inputs=[], outputs=[])
        large_query.input(fn=It.flip('large_query'), inputs=[], outputs=[])
        unload_after.input(fn=It.flip('unload_after'), inputs=[], outputs=[])

        save_tags.input(fn=IOData.flip_save_tags(), inputs=[], outputs=[])

        input_glob.blur(fn=wrap_gradio_gpu_call(It.set("input_glob")),
                        inputs=[input_glob], outputs=[input_glob, info])
        output_dir.blur(fn=wrap_gradio_gpu_call(It.set("output_dir")),
                        inputs=[output_dir], outputs=[output_dir, info])

        threshold.input(fn=QData.set("threshold"), inputs=[threshold],
                        outputs=[])
        threshold.release(fn=QData.set("threshold"), inputs=[threshold],
                          outputs=[])

        tag_frac_threshold.input(fn=QData.set("tag_frac_threshold"),
                                 inputs=[tag_frac_threshold], outputs=[])
        tag_frac_threshold.release(fn=QData.set("tag_frac_threshold"),
                                   inputs=[tag_frac_threshold], outputs=[])

        add_tags.blur(fn=wrap_gradio_gpu_call(It.set('add')),
                      inputs=[add_tags], outputs=[add_tags, info])

        keep_tags.blur(fn=wrap_gradio_gpu_call(It.set('keep')),
                       inputs=[keep_tags], outputs=[keep_tags, info])
        exclude_tags.blur(fn=wrap_gradio_gpu_call(It.set('exclude')),
                          inputs=[exclude_tags], outputs=[exclude_tags, info])

        search_tags.blur(fn=wrap_gradio_gpu_call(It.set('search')),
                         inputs=[search_tags], outputs=[search_tags, info])
        replace_tags.blur(fn=wrap_gradio_gpu_call(It.set('replace')),
                          inputs=[replace_tags], outputs=[replace_tags, info])

        # register events
        tag_search_selection.change(
            fn=wrap_gradio_gpu_call(on_tag_search_filter_change),
            inputs=[tag_search_selection],
            outputs=[
                discarded_tags if QData.inverse else tags,
                excluded_tag_confidences if QData.inverse else tag_confidences,
                info])

        # register events
        tag_search_selection.blur(
            fn=wrap_gradio_gpu_call(on_tag_search_filter_change),
            inputs=[tag_search_selection],
            outputs=[
                discarded_tags if QData.inverse else tags,
                excluded_tag_confidences if QData.inverse else tag_confidences,
                info])

        # register events
        selected_preset.change(
            fn=utils.preset.apply,
            inputs=[selected_preset],
            outputs=[*utils.preset.components, info])

        save_preset_button.click(
            fn=utils.preset.save,
            inputs=[selected_preset, *utils.preset.components],  # values only
            outputs=[info])

        unload_all_models.click(fn=unload_interrogators, outputs=[info])

        image.change(
            fn=wrap_gradio_gpu_call(on_interrogate_image),
            inputs=[image, interrogator],
            outputs=[tags, rating_confidences, tag_confidences, info])

        image_submit.click(
            fn=wrap_gradio_gpu_call(on_interrogate_image),
            inputs=[image, interrogator],
            outputs=[tags, rating_confidences, tag_confidences, info])

        batch_submit.click(
            fn=wrap_gradio_gpu_call(on_interrogate),
            inputs=[interrogator],
            outputs=[tags, rating_confidences, tag_confidences, info])

    return [(tagger_interface, "Tagger", "tagger")]
