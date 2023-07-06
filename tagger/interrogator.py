import os
from pathlib import Path
from io import BytesIO
from hashlib import sha256
import json
from pandas import read_csv, read_json
from PIL import Image
from typing import Tuple, List, Dict, Callable
from numpy import asarray, float32, expand_dims
from tqdm import tqdm

from huggingface_hub import hf_hub_download
from modules import shared

from . import dbimutils
from tagger import settings
from tagger.uiset import QData, IOData, it_ret_tp

Its = settings.InterrogatorSettings

# select a device to process
use_cpu = ('all' in shared.cmd_opts.use_cpu) or (
    'interrogate' in shared.cmd_opts.use_cpu)

if use_cpu:
    TF_DEVICE_NAME = '/cpu:0'
else:
    TF_DEVICE_NAME = '/gpu:0'

    if shared.cmd_opts.device_id is not None:
        try:
            TF_DEVICE_NAME = f'/gpu:{int(shared.cmd_opts.device_id)}'
        except ValueError:
            print('--device-id is not a integer')


def get_file_interrogator_id(bytes, interrogator_name):
    hasher = sha256()
    hasher.update(bytes)
    return str(hasher.hexdigest()) + interrogator_name


def split_str(string: str, separator=',') -> List[str]:
    return [x.strip() for x in string.split(separator) if x]


class Interrogator:
    # the raw input and output.
    input = {
        "cumulative": True,
        "large_query": False,
        "unload_after": False,
        "threshold_on_average": False,
        "add": '',
        "keep": '',
        "exclude": '',
        "search": '',
        "replace": '',
        "paths": '',
        "input_glob": '',
        "output_dir": '',
    }
    output = None

    @classmethod
    def flip(cls, key):
        def toggle():
            cls.input[key] = not cls.input[key]
            return ('',)
        return toggle

    @classmethod
    def set(cls, key: str) -> Callable[[str], Tuple[str, str]]:
        def setter(val) -> Tuple[str, str]:
            err = ''
            if val != cls.input[key]:
                if key == 'input_glob' or key == 'output_dir':
                    err = getattr(IOData, "update_" + key)(val)
                else:
                    err = getattr(QData, "update_" + key)(val)
                if err == '':
                    cls.input[key] = val
            return cls.input[key], err

        return setter

    def __init__(self, name: str) -> None:
        self.name = name
        # run_mode 0 is dry run, 1 means run (alternating), 2 means disabled
        self.run_mode = 0 if hasattr(self, "large_batch_interrogate") else 2

    def load(self):
        raise NotImplementedError()

    def unload(self) -> bool:
        unloaded = False

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            unloaded = True
            print(f'Unloaded {self.name}')

        if hasattr(self, 'tags'):
            del self.tags

        return unloaded

    def interrogate_image(self, image: Image) -> it_ret_tp:
        fi_key = get_file_interrogator_id(image.tobytes(), self.name)
        if not Interrogator.input["cumulative"]:
            QData.init_query()

        if fi_key in QData.query:
            # this file was already queried for this interrogator.
            data = QData.get_single_data(fi_key)
            # clear: the data does not require storage
            fi_key = ""
        else:
            # single process
            data = self.interrogate(image)
            if Interrogator.input["unload_after"]:
                self.unload()

        on_avg = Interrogator.input["threshold_on_average"]
        QData.apply_filters(('', '') + data, fi_key, on_avg)
        Interrogator.output = QData.finalize(1, on_avg)
        return Interrogator.output

    def batch_interrogate(
        self,
        batch_rewrite: bool,
    ) -> it_ret_tp:
        QData.init_query()
        in_db = {}
        ct = len(QData.query)

        if Interrogator.input["large_query"] is True and self.run_mode < 2:
            # TODO: write specified tags files instead of simple .txt
            image_list = [str(x[0].resolve()) for x in IOData.paths]
            err = self.large_batch_interrogate(image_list, self.run_mode == 0)
            if err:
                return (None, None, None, err)

            self.run_mode = (self.run_mode + 1) % 2
            Interrogator.output = QData.finalize(len(image_list))
            return Interrogator.output

        vb = getattr(shared.opts, 'tagger_verbose', True)
        on_avg = Interrogator.input["threshold_on_average"]

        for i in tqdm(range(len(IOData.paths)), disable=vb, desc='Tags'):
            # if outputpath is '', no tags file will be written
            (path, out_path, output_dir) = IOData.paths[i]
            if output_dir:
                output_dir.mkdir(0o755, True, True)
                # next iteration we don't need to create the directory
                IOData.paths[i][2] = ''

            try:
                image = Image.open(path)
            except Exception as e:
                # just in case, user has mysterious file...
                print(f'${path} is not supported image type: {e}')
                continue

            abspath = str(path.absolute())
            fi_key = get_file_interrogator_id(image.tobytes(), self.name)

            if fi_key in QData.query:
                # this file was already queried for this interrogator.
                index = QData.get_index(fi_key, abspath)
                in_db[index] = [abspath, out_path, {}, {}]

            elif batch_rewrite:
                # with batch rewrite we only read from database
                print(f'new file {abspath}: requires interrogation (skipped)')
            else:
                data = (abspath, out_path) + self.interrogate(image)
                QData.apply_filters(data, fi_key, on_avg)

        if Interrogator.input["unload_after"]:
            self.unload()

        ct = len(QData.query) - ct
        Interrogator.output = QData.finalize_batch(in_db, ct, on_avg)
        return Interrogator.output

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidences
        Dict[str, float]  # tag confidences
    ]:
        raise NotImplementedError()


class DeepDanbooruInterrogator(Interrogator):
    def __init__(self, name: str, project_path: os.PathLike) -> None:
        super().__init__(name)
        self.project_path = project_path

    def load(self) -> None:
        print(f'Loading {self.name} from {str(self.project_path)}')

        # deepdanbooru package is not include in web-sd anymore
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/c81d440d876dfd2ab3560410f37442ef56fc663
        from launch import is_installed, run_pip
        if not is_installed('deepdanbooru'):
            package = os.environ.get(
                'DEEPDANBOORU_PACKAGE',
                'git+https://github.com/KichangKim/DeepDanbooru.'
                'git@d91a2963bf87c6a770d74894667e9ffa9f6de7ff'
            )

            run_pip(
                f'install {package} tensorflow tensorflow-io', 'deepdanbooru')

        import tensorflow as tf

        # tensorflow maps nearly all vram by default, so we limit this
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        # TODO: only run on the first run
        for device in tf.config.experimental.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError as e:
                print(e)

        with tf.device(TF_DEVICE_NAME):
            import deepdanbooru.project as ddp

            self.model = ddp.load_model_from_project(
                project_path=self.project_path,
                compile_model=False
            )

            print(f'Loaded {self.name} model from {str(self.project_path)}')

            self.tags = ddp.load_tags_from_project(
                project_path=self.project_path
            )

    def unload(self) -> bool:
        # unloaded = super().unload()

        # if unloaded:
        #     # tensorflow suck
        #     # https://github.com/keras-team/keras/issues/2102
        #     import tensorflow as tf
        #     tf.keras.backend.clear_session()
        #     gc.collect()

        # return unloaded

        # There is a bug in Keras where it is not possible to release a model
        # that has been loaded into memory. Downgrading to keras==2.1.6 may
        # solve the issue, but it may cause compatibility issues with other
        # packages. Using subprocess to create a new process may also solve the
        # problem, but it can be too complex (like Automatic1111 did). It seems
        # that for now, the best option is to keep the model in memory, as most
        # users use the Waifu Diffusion model with onnx.
        return False

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidences
        Dict[str, float]  # tag confidences
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        import deepdanbooru.data as ddd

        # convert an image to fit the model
        image_bufs = BytesIO()
        image.save(image_bufs, format='PNG')
        image = ddd.load_image_for_evaluate(
            image_bufs,
            self.model.input_shape[2],
            self.model.input_shape[1]
        )

        image = image.reshape((1, *image.shape[0:3]))

        # evaluate model
        result = self.model.predict(image)

        confidences = result[0].tolist()
        ratings = {}
        tags = {}

        for i, tag in enumerate(self.tags):
            if tag[:7] != "rating:":
                tags[tag] = confidences[i]
            else:
                ratings[tag[7:]] = confidences[i]

        return ratings, tags


class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
        self,
        name: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
        **kwargs
    ) -> None:
        super().__init__(name)
        self.model_path = model_path
        self.tags_path = tags_path
        self.kwargs = kwargs

    def download(self) -> Tuple[os.PathLike, os.PathLike]:
        print(f"Loading {self.name} model file from {self.kwargs['repo_id']}")

        mdir = Path(shared.models_path, 'interrogators')
        model_path = Path(hf_hub_download(**self.kwargs, filename=self.model_path, cache_dir=mdir))
        tags_path = Path(hf_hub_download(**self.kwargs, filename=self.tags_path, cache_dir=mdir))

        download_model = {
            'name': self.name,
            'model_path': str(model_path),
            'tags_path': str(tags_path),
        }
        mpath = Path(mdir, 'model.json')

        if not os.path.exists(mdir):
            os.makedir(mdir)

        elif os.path.exists(mpath):
            with open(mpath, 'r') as f:
                try:
                    data = json.load(f)
                    data.append(download_model)
                except Exception as e:
                    print(f'Adding download_model {mpath} raised {repr(e)}')
                    data = [download_model]

        with open(mpath, 'w') as f:
            json.dump(data, f)

        return model_path, tags_path

    def get_model_path(self) -> Tuple[os.PathLike, os.PathLike]:
        model_path = ''
        tags_path = ''
        mpath = Path(shared.models_path, 'interrogators', 'model.json')
        try:
            models = read_json(mpath).to_dict(orient='records')
            i = next(i for i in models if i['name'] == self.name)
            model_path = i['model_path']
            tags_path = i['tags_path']
        except Exception as e:
            print(f'{mpath}: requires a name, model_ and tags_path: {repr(e)}')
            model_path, tags_path = self.download()
        return model_path, tags_path

    def load(self) -> None:
        if hasattr(self, 'model_path') and hasattr(self, 'tags_path'):
            model_path = self.model_path
            tags_path = self.tags_path
        else:
            model_path, tags_path = self.download()

        # only one of these packages should be installed a time in any one env
        # https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime
        # TODO: remove old package when the environment changes?
        from launch import is_installed, run_pip
        if not is_installed('onnxruntime'):
            package = os.environ.get(
                'ONNXRUNTIME_PACKAGE',
                'onnxruntime-gpu'
            )

            run_pip(f'install {package}', 'onnxruntime')

        from onnxruntime import InferenceSession

        # https://onnxruntime.ai/docs/execution-providers/
        # https://github.com/toriato/stable-diffusion-webui-wd14-tagger/commit/e4ec460122cf674bbf984df30cdb10b4370c1224#r92654958
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if use_cpu:
            providers.pop(0)

        self.model = InferenceSession(str(model_path), providers=providers)

        print(f'Loaded {self.name} model from {model_path}, {tags_path}')

        self.tags = read_csv(tags_path)

    def interrogate(
        self,
        image: Image
    ) -> Tuple[
        Dict[str, float],  # rating confidences
        Dict[str, float]  # tag confidences
    ]:
        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        # code for converting the image and running the model is taken from the
        # link below. thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

        # convert an image to fit the model
        _, height, _, _ = self.model.get_inputs()[0].shape

        # alpha to white
        image = image.convert('RGBA')
        new_image = Image.new('RGBA', image.size, 'WHITE')
        new_image.paste(image, mask=image)
        image = new_image.convert('RGB')
        image = asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(float32)
        image = expand_dims(image, 0)

        # evaluate model
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        confidences = self.model.run([label_name], {input_name: image})[0]

        tags = self.tags[:][['name']]
        tags['confidences'] = confidences[0]

        # first 4 items are for rating (general, sensitive, questionable,
        # explicit)
        ratings = dict(tags[:4].values)

        # rest are regular tags
        tags = dict(tags[4:].values)

        return ratings, tags

    def large_batch_interrogate(self, images_list, dry_run=True) -> str:

        # init model
        if not hasattr(self, 'model') or self.model is None:
            self.load()

        os.environ["TF_XLA_FLAGS"] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        # Reduce logging
        # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

        import tensorflow as tf

        from tagger.Generator.TFDataReader import DataGenerator

        # tensorflow maps nearly all vram by default, so we limit this
        # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
        # TODO: only run on the first run
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for device in gpus:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except RuntimeError as e:
                    print(e)

        if dry_run:  # dry run
            height, width = 224, 224

            def process_images(filepaths, images):
                lines = []
                for image_path in filepaths:
                    image_path = image_path.numpy().decode("utf-8")
                    lines.append(f"{image_path}\n")
                with open("dry_run_read.txt", "a") as f:
                    f.writelines(lines)

            scheduled = [f"{image_path}\n" for image_path in images_list]

            # Truncate the file from previous runs
            print("updating dry_run_read.txt")
            open("dry_run_read.txt", "w").close()
            with open("dry_run_scheduled.txt", "w") as f:
                f.writelines(scheduled)
        else:
            _, height, width, _ = self.model.inputs[0].shape

            threshold = QData.threshold
            self.tags["sanitized_name"] = self.tags["name"].map(
                lambda x: x if x in Its.kaomojis else x.replace("_", " ")
            )

            @tf.function
            def pred_model(x):
                return self.model(x, training=False)

            def process_images(filepaths, images):
                preds = pred_model(images).numpy()

                for image_path, pred in zip(filepaths, preds):
                    image_path = image_path.numpy().decode("utf-8")

                    self.tags["preds"] = pred
                    general_tags = self.tags[self.tags["category"] == 0]
                    chosen_tags = general_tags[general_tags["preds"] > threshold]
                    chosen_tags = chosen_tags.sort_values(by="preds", ascending=False)
                    tags_names = chosen_tags["sanitized_name"]

                    fi_key = image_path.split("/")[-1].split(".")[0] + "_" + self.name
                    QData.add_tags = tags_names
                    QData.apply_filters((image_path, '', {}, {}), fi_key, False)

                    tags_string = ", ".join(tags_names)
                    with open(Path(image_path).with_suffix(".txt"), "w") as f:
                        f.write(tags_string)

        batch_size = getattr(shared.opts, 'tagger_batch_size', 1024)
        generator = DataGenerator(
            file_list=images_list, target_height=height, target_width=width, batch_size=batch_size
        ).genDS()

        orig_add_tags = QData.add_tags
        for filepaths, images in tqdm(generator):
            process_images(filepaths, images)
        QData.add_tag = orig_add_tags
        del os.environ["TF_XLA_FLAGS"]
        return ''
