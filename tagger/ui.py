import gradio as gr
from PIL import Image

from modules import ui
from modules import generation_parameters_copypaste as parameters_copypaste

from tagger import utils
from tagger.interrogator import Interrogator as It
from webui import wrap_gradio_gpu_call

BATCH_REWRITE = 'Update tag lists'


def unload_interrogators():
    unloaded_models = 0

    for i in utils.interrogators.values():
        if i.unload():
            unloaded_models = unloaded_models + 1

    return [f'Successfully unload {unloaded_models} model(s)']


def on_interrogate(
    button: str, name: str, unload_after: bool
):
    if name not in utils.interrogators:
        return [None, None, None, f"'{name}': invalid interrogator"]

    it: It = utils.interrogators[name]
    return it.batch_interrogate(unload_after, button == BATCH_REWRITE)


def on_interrogate_image_change(*args):
    # FIXME for some reason an image change is triggered twice.
    # this is a dirty hack to prevent summation/flushing the output.
    print("interrogator: "+args[1])
    It.image_counter += 1
    if It.image_counter & 1 == 0:
        # in db.json ratings can be 2x too high
        return It.results()
    return on_interrogate_image(*args)


def on_interrogate_image(image: Image, interrogator: str, unload_after: bool):
    if image is None:
        return [None, None, None, 'No image']

    if interrogator not in utils.interrogators:
        return [None, None, None, f"'{interrogator}': invalid interrogator"]

    interrogator: It = utils.interrogators[interrogator]
    return interrogator.interrogate_image(image, unload_after)


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
                        batch_input_glob = utils.preset.component(
                            gr.Textbox,
                            label='Input directory - See also settings tab.',
                            placeholder='/path/to/images or to/images/**/*'
                        )
                        batch_output_dir = utils.preset.component(
                            gr.Textbox,
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
                with gr.Row(variant='compact'):
                    with gr.Column(variant='compact'):
                        threshold = utils.preset.component(
                            gr.Slider,
                            label='Threshold',
                            minimum=0,
                            maximum=1,
                            value=0.35
                        )
                        add_tags = utils.preset.component(
                            gr.Textbox,
                            label='Additional tags (split by comma)',
                            elem_id='additional-tags'
                        )

                        search_tags = utils.preset.component(
                            gr.Textbox,
                            label='Search tags (split by comma)',
                            elem_id='search-tags'
                        )
                        cumulative_mean = utils.preset.component(
                            gr.Checkbox,
                            label='combine interrogations',
                            value=True
                        )
                        save_tags = utils.preset.component(
                            gr.Checkbox,
                            label='Auto save to tags files',
                            value=True
                        )
                    with gr.Column(variant='compact'):
                        count_threshold = utils.preset.component(
                            gr.Slider,
                            label='Tag count threshold',
                            minimum=1,
                            maximum=500,
                            value=50,
                            step=1.0
                        )
                        exclude_tags = utils.preset.component(
                            gr.Textbox,
                            label='Exclude tags (split by comma)',
                            elem_id='exclude-tags'
                        )
                        replace_tags = utils.preset.component(
                            gr.Textbox,
                            label='Replacement tags (split by comma)',
                            elem_id='replace-tags'
                        )
                        unload_model_after_run = utils.preset.component(
                            gr.Checkbox,
                            label='Unload model after running',
                        )

            # output components
            with gr.Column(variant='panel'):
                tags = gr.Textbox(
                    label='Tags',
                    placeholder='Found tags',
                    interactive=False,
                    elem_classes=':link'
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
                    elem_id='rating-confidences'
                )
                alphabetical = utils.preset.component(
                    gr.Checkbox,
                    label='Sort by alphabetical order',
                )
                tag_confidences = gr.Label(
                    label='Tag confidences',
                    elem_id='tag-confidences'
                )

        batch_input_glob.blur(fn=It.set_input_glob, inputs=[batch_input_glob], outputs=[])
        batch_output_dir.blur(fn=It.set_output_dir, inputs=[batch_output_dir], outputs=[])

        cumulative_mean.input(fn=It.set_cumulative, inputs=[cumulative_mean], outputs=[])

        threshold.input(fn=It.set_threshold, inputs=[threshold], outputs=[])
        threshold.release(fn=It.set_threshold, inputs=[threshold], outputs=[])
        count_threshold.input(fn=It.set_count_threshold, inputs=[count_threshold], outputs=[])
        count_threshold.release(fn=It.set_count_threshold, inputs=[count_threshold], outputs=[])

        add_tags.blur(fn=It.set_add, inputs=[add_tags], outputs=[])
        exclude_tags.blur(fn=It.set_exclude, inputs=[exclude_tags], outputs=[])
        search_tags.blur(fn=It.set_search, inputs=[search_tags], outputs=[info])
        replace_tags.blur(fn=It.set_replace, inputs=[replace_tags], outputs=[info])

        alphabetical.input(fn=It.set_alphabetical, inputs=[alphabetical], outputs=[])
        save_tags.input(fn=It.set_save_tags, inputs=[save_tags], outputs=[])

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
            fn=wrap_gradio_gpu_call(on_interrogate_image_change),
            inputs=[image, interrogator, unload_model_after_run],
            outputs=[tags, rating_confidences, tag_confidences, info]
        )
        image_submit.click(
            fn=wrap_gradio_gpu_call(on_interrogate_image),
            inputs=[image, interrogator, unload_model_after_run],
            outputs=[tags, rating_confidences, tag_confidences, info]
        )

        for button in [batch_rewrite, batch_submit]:
            button.click(
                fn=wrap_gradio_gpu_call(on_interrogate),
                inputs=[button, interrogator, unload_model_after_run],
                outputs=[tags, rating_confidences, tag_confidences, info]
            )

    return [(tagger_interface, "Tagger", "tagger")]
