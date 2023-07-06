import gradio as gr
from PIL import Image
from typing import Dict, Tuple, List

from modules import ui
from modules import generation_parameters_copypaste as parameters_copypaste

from tagger import utils
from tagger.interrogator import Interrogator as It
from webui import wrap_gradio_gpu_call
from tagger.uiset import IOData, QData, it_ret_tp
from tensorflow import __version__ as tf_version
from packaging import version

BATCH_REWRITE = 'Update tag lists'


def unload_interrogators() -> List[str]:
    unloaded_models = 0

    for i in utils.interrogators.values():
        if i.unload():
            unloaded_models = unloaded_models + 1

    return (f'Successfully unload {unloaded_models} model(s)',)


def on_interrogate(button: str, name: str, inverse=False) -> it_ret_tp:
    if It.err != '':
        return (None, None, None, It.err)

    if name in utils.interrogators:
        it: It = utils.interrogators[name]
        QData.inverse = inverse
        return it.batch_interrogate(button == BATCH_REWRITE)

    return (None, None, None, f"'{name}': invalid interrogator")


def on_inverse_interrogate(button: str, name: str) -> it_ret_tp:
    ret = on_interrogate(button, name, True)
    return (ret[0], ret[2], ret[3])


def on_interrogate_image(image: Image, interrogator: str) -> it_ret_tp:
    if It.err != '':
        return (None, None, None, It.err)

    if image is None:
        return (None, None, None, 'No image selected')

    if interrogator in utils.interrogators:
        interrogator: It = utils.interrogators[interrogator]
        return interrogator.interrogate_image(image)

    return (None, None, None, f"'{interrogator}': invalid interrogator")


def move_filter_to_input_fn(
    tag_search_filter: str,
    name: str,
    input: str
) -> Tuple[str, str, Dict[str, float], Dict[str, float], str]:
    if It.output is None:
        return (None, None, None, None, '')

    filt = {(k, v) for k, v in It.output[2].items() if tag_search_filter in k}
    if len(filt) == 0:
        return (None, None, None, None, '')

    add = set(dict(filt).keys())
    if It.input[input] != '':
        add = add.union({x.strip() for x in It.input[input].split(',')})

    It.input[input] = ', '.join(add)

    ret = on_interrogate(BATCH_REWRITE, name, QData.inverse)
    return (It.input[input],) + ret


def move_filter_to_keep_fn(
    tag_search_filter: str, name: str
) -> Tuple[str, str, Dict[str, float], str]:
    ret = move_filter_to_input_fn(tag_search_filter, name, "keep")
    # ratings are not displayed on this tab
    return ('',) + ret[:2] + ret[3:]


def move_filter_to_exclude_fn(
    tag_search_filter: str, name: str
) -> Tuple[str, str, Dict[str, float], Dict[str, float], str]:
    return ('',) + move_filter_to_input_fn(tag_search_filter, name, "exclude")


def on_tag_search_filter_change(
    part: str
) -> Tuple[str, Dict[str, float], str]:
    if It.output is None:
        return (None, None, '')
    if len(part) < 2:
        return (It.output[0], It.output[2], '')
    tags = dict(filter(lambda x: part in x[0], It.output[2].items()))
    return (', '.join(tags.keys()), tags, '')


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as tagger_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):

                # input components
                with gr.Tabs():
                    with gr.TabItem(label='Single process'):
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

                    with gr.TabItem(label='Batch from directory'):
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

                        batch_rewrite = gr.Button(value=BATCH_REWRITE)

                        batch_submit = gr.Button(
                            value='Interrogate',
                            variant='primary'
                        )

                info = gr.HTML()

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
                        interrogator_names = utils.refresh_interrogators()
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
                            lambda: {'choices': utils.refresh_interrogators()},
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
                            label='Threshold',
                            minimum=0,
                            maximum=1,
                            value=QData.threshold
                        )
                        cumulative = utils.preset.component(
                            gr.Checkbox,
                            label='combine interrogations',
                            value=It.input["cumulative"]
                        )
                        threshold_on_average = utils.preset.component(
                            gr.Checkbox,
                            label='threshold applies on average of all images '
                            'and interrogations',
                            value=It.input["threshold_on_average"]
                        )
                        save_tags = utils.preset.component(
                            gr.Checkbox,
                            label='Save to tags files',
                            value=IOData.save_tags
                        )
                    with gr.Column(variant='compact'):
                        count_threshold = utils.preset.component(
                            gr.Slider,
                            label='Tag count threshold',
                            minimum=1,
                            maximum=500,
                            value=QData.count_threshold,
                            step=1.0
                        )

                        large_query = utils.preset.component(
                            gr.Checkbox,
                            label='huge batch query (tensorflow 2.10, '
                            'experimental)',
                            value=It.input["large_query"],
                            interactive=version.parse(tf_version) ==
                            version.parse('2.10')
                        )
                        unload_after = utils.preset.component(
                            gr.Checkbox,
                            label='Unload model after running',
                            value=It.input["unload_after"]
                        )
                with gr.Row(variant='compact'):
                    with gr.Column(variant='compact'):
                        search_tags = utils.preset.component(
                            gr.Textbox,
                            label='Search tags (comma split)',
                            elem_id='search-tags'
                        )
                        keep_tags = utils.preset.component(
                            gr.Textbox,
                            label='Preserved tags; regardless of thresholds, '
                            'comma delimited',
                            elem_id='keep-tags'
                        )
                    with gr.Column(variant='compact'):
                        replace_tags = utils.preset.component(
                            gr.Textbox,
                            label='Replacement tags (comma split)',
                            elem_id='replace-tags'
                        )
                        exclude_tags = utils.preset.component(
                            gr.Textbox,
                            label='Excluded tags; regardless, comma delimited',
                            elem_id='exclude-tags'
                        )

            # output components
            with gr.Column(variant='panel'):

                with gr.Tabs():
                    tab_ratings_and_included_tags = gr.TabItem(label='Ratings and included tags')
                    tab_discarded_tags = gr.TabItem(label='Excluded tags')
                    with tab_ratings_and_included_tags:
                        with gr.Row(variant='compact'):
                            with gr.Column(variant='compact'):
                                move_filter_to_exclude = gr.Button(
                                    value='Move visible tags to exclude tags',
                                    variant='secondary'
                                )
                            with gr.Column(variant='compact'):
                                tag_search_selected = utils.preset.component(
                                    gr.Textbox,
                                    label='string search selected tags'
                                )
                            label='Tags',
                            placeholder='Found tags',
                            interactive=False,
                            elem_classes=':link',
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
                    with tab_discarded_tags:
                        with gr.Row(variant='compact'):
                            with gr.Column(variant='compact'):
                                move_filter_to_keep = gr.Button(
                                    value='Move visible tags to keep tags',
                                    variant='secondary'
                                )
                            with gr.Column(variant='compact'):
                                tag_search_excluded = utils.preset.component(
                                    gr.Textbox,
                                    label='string search excluded tags'
                                )

                        discarded_tags = gr.Textbox(
                            label='Tags',
                            placeholder='Discarding tags',
                            interactive=False,
                            elem_classes=':link',
                        )
                        excluded_tag_confidences = gr.Label(
                            label='Excluded Tag confidences',
                            elem_id='tag-confidences',
                        )

        tab_ratings_and_included_tags.select(fn=wrap_gradio_gpu_call(on_interrogate),
                                 inputs=[batch_rewrite, interrogator],
                                 outputs=[tags, rating_confidences,
                                          tag_confidences, info])

        tab_discarded_tags.select(fn=wrap_gradio_gpu_call(on_inverse_interrogate),
                                 inputs=[batch_rewrite, interrogator],
                                 outputs=[discarded_tags,
                                          excluded_tag_confidences, info])

        move_filter_to_keep.click(
            fn=wrap_gradio_gpu_call(move_filter_to_keep_fn),
            inputs=[tag_search_excluded, interrogator],
            outputs=[
                tag_search_excluded,
                keep_tags,
                discarded_tags,
                excluded_tag_confidences,
                info
            ]
        )
        move_filter_to_exclude.click(
            fn=wrap_gradio_gpu_call(move_filter_to_exclude_fn),
            inputs=[tag_search_selected, interrogator],
            outputs=[
                tag_search_selected,
                exclude_tags,
                tags,
                rating_confidences,
                tag_confidences,
                info
            ]
        )
        cumulative.input(fn=wrap_gradio_gpu_call(It.flip('cumulative')), inputs=[], outputs=[info])
        large_query.input(fn=wrap_gradio_gpu_call(It.flip('large_query')), inputs=[], outputs=[info])
        unload_after.input(fn=wrap_gradio_gpu_call(It.flip('unload_after')), inputs=[], outputs=[info])
        threshold_on_average.input(fn=wrap_gradio_gpu_call(It.flip('threshold_on_average')), inputs=[], outputs=[info])

        save_tags.input(fn=wrap_gradio_gpu_call(IOData.flip_save_tags()), inputs=[], outputs=[info])
        input_glob.blur(fn=wrap_gradio_gpu_call(It.set("input_glob")), inputs=[input_glob],
                        outputs=[info])
        output_dir.blur(fn=wrap_gradio_gpu_call(It.set("output_dir")), inputs=[output_dir],
                        outputs=[info])

        threshold.input(fn=wrap_gradio_gpu_call(QData.set("threshold")), inputs=[threshold],
                        outputs=[info])
        threshold.release(fn=wrap_gradio_gpu_call(QData.set("threshold")), inputs=[threshold],
                          outputs=[info])

        count_threshold.input(fn=wrap_gradio_gpu_call(QData.set("count_threshold")),
                              inputs=[count_threshold], outputs=[info])
        count_threshold.release(fn=wrap_gradio_gpu_call(QData.set("count_threshold")),
                                inputs=[count_threshold], outputs=[info])

        add_tags.blur(fn=wrap_gradio_gpu_call(It.set('add')), inputs=[add_tags], outputs=[info])
        keep_tags.blur(fn=wrap_gradio_gpu_call(It.set('keep')), inputs=[keep_tags], outputs=[info])
        exclude_tags.blur(fn=wrap_gradio_gpu_call(It.set('exclude')), inputs=[exclude_tags],
                          outputs=[info])
        search_tags.blur(fn=wrap_gradio_gpu_call(It.set('search')), inputs=[search_tags],
                         outputs=[info])
        replace_tags.blur(fn=wrap_gradio_gpu_call(It.set('replace')), inputs=[replace_tags],
                          outputs=[info])

        # register events
        tag_search_selected.change(
            fn=wrap_gradio_gpu_call(on_tag_search_filter_change),
            inputs=[tag_search_selected],
            outputs=[tags, tag_confidences, info]
        )

        # register events
        tag_search_selected.blur(
            fn=wrap_gradio_gpu_call(on_tag_search_filter_change),
            inputs=[tag_search_selected],
            outputs=[tags, tag_confidences, info]
        )

        tag_search_excluded.change(
            fn=wrap_gradio_gpu_call(on_tag_search_filter_change),
            inputs=[tag_search_excluded],
            outputs=[discarded_tags, excluded_tag_confidences, info]
        )

        # register events
        tag_search_excluded.blur(
            fn=wrap_gradio_gpu_call(on_tag_search_filter_change),
            inputs=[tag_search_excluded],
            outputs=[discarded_tags, excluded_tag_confidences, info]
        )
        # register events
        selected_preset.change(
            fn=utils.preset.apply,
            inputs=[selected_preset],
            outputs=[*utils.preset.components, info]
        )

        save_preset_button.click(
            fn=utils.preset.save,
            inputs=[selected_preset, *utils.preset.components],  # values only
            outputs=[info]
        )

        unload_all_models.click(fn=unload_interrogators, outputs=[info])

        image.change(
            fn=wrap_gradio_gpu_call(on_interrogate_image),
            inputs=[image, interrogator],
            outputs=[tags, rating_confidences, tag_confidences, info]
        )
        image_submit.click(
            fn=wrap_gradio_gpu_call(on_interrogate_image),
            inputs=[image, interrogator],
            outputs=[tags, rating_confidences, tag_confidences, info]
        )

        for button in [batch_rewrite, batch_submit]:
            button.click(
                fn=wrap_gradio_gpu_call(on_interrogate),
                inputs=[button, interrogator],
                outputs=[tags, rating_confidences, tag_confidences, info]
            )

    return [(tagger_interface, "Tagger", "tagger")]
