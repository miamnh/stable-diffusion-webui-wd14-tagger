import os
from re import compile as re_comp, sub as re_sub, match as re_match, IGNORECASE
from math import ceil
from typing import Tuple, List, Dict
from io import BytesIO
from pathlib import Path
from glob import glob
from hashlib import sha256
from json import dumps as json_dumps, loads as json_loads

from pandas import read_csv
from numpy import asarray, float32, expand_dims
from tqdm import tqdm

from PIL import Image, UnidentifiedImageError

from huggingface_hub import hf_hub_download

from modules import shared
from modules.deepbooru import re_special as tag_escape_pattern

# i'm not sure if it's okay to add this file to the repository
from tagger import utils, format as tagger_format
from . import dbimutils

BATCH_REWRITE_VALUE = 'Update tag lists'

# select a device to process
use_cpu = ('all' in shared.cmd_opts.use_cpu) or (
    'interrogate' in shared.cmd_opts.use_cpu)

# PIL.Image.registered_extensions() returns only PNG if you call early
supported_extensions = {
    e
    for e, f in Image.registered_extensions().items()
    if f in Image.OPEN
}

if use_cpu:
    TF_DEVICE_NAME = '/cpu:0'
else:
    TF_DEVICE_NAME = '/gpu:0'

    if shared.cmd_opts.device_id is not None:
        try:
            TF_DEVICE_NAME = f'/gpu:{int(shared.cmd_opts.device_id)}'
        except ValueError:
            print('--device-id is not a integer')


def get_i_wt(stored: float):
    i = ceil(stored) - 1
    return (i, stored - i)


def get_file_interrogator_id(bytes, interrogator_name):
    hasher = sha256()
    hasher.update(bytes)
    return str(hasher.hexdigest()) + interrogator_name


def on_interrogate(
    button_value: str, interrogator_name: str, unload_model_after_run: bool,
    auto_save_tags: bool, *args
):
    batch_rewrite = button_value == BATCH_REWRITE_VALUE

    if interrogator_name not in utils.interrogators:
        return [None, None, None, f"'{interrogator_name}': invalid interrogator"]

    interrogator: Interrogator = utils.interrogators[interrogator_name]

    error = Interrogator.test_reinit(*args)
    if error:
        return [None, None, None, error]

    auto_json = getattr(shared.opts, 'tagger_auto_serde_json', True)

    json_db = Interrogator.output_dir_root.joinpath('db.json') if auto_json else None
    if json_db:
        try:
            data = json_loads(json_db.read_text())
            if "tag" not in data or "rating" not in data or len(data) < 3:
                raise TypeError
            Interrogator.data = data
        except Exception:
            if batch_rewrite:
                return [None, None, None, 'No database']

    verbose = getattr(shared.opts, 'tagger_verbose', True)
    in_db = {}
    processed_ct = 0

    for (path, out_path) in tqdm(Interrogator.iter(), disable=verbose, desc='Tagging'):
        try:
            image = Image.open(path)

        except UnidentifiedImageError:
            # just in case, user has mysterious file...
            print(f'${path} is not supported image type')
            continue

        abspath = str(path.absolute())
        key = get_file_interrogator_id(image.tobytes(), interrogator_name)

        if key in Interrogator.data:
            if abspath != Interrogator.data[key][0]:
                if Interrogator.data[key][0] != '':
                    print(f'Dup or rename: Identical checksums for {abspath}\n'
                          'and: {Interrogator.data[key][0]} (path updated)')
                Interrogator.data[key][0] = abspath

            # this file was already queried for this interrogator.
            index = Interrogator.data[key][1]
            in_db[index] = {
                "tag": {},
                "rating": {},
                "path": path,
                "out_path": out_path
            }
            continue

        if batch_rewrite:
            # with batch rewrite we only read from database
            print(f'new file {abspath}: requires interrogation (skipped)')
            continue

        ratings, tags = interrogator.interrogate(image)
        index = Interrogator.ct
        Interrogator.data[key] = (abspath, index)

        Interrogator.postprocess(ratings, "rating", True)
        (count, txt) = Interrogator.postprocess(tags, "tag", True)
        Interrogator.ct += 1
        if verbose:
            print(f'{path}: {count}/{len(tags)} tags kept')

        if auto_save_tags:
            out_path.write_text(txt, encoding='utf-8')

        processed_ct += 1

    if unload_model_after_run:
        interrogator.unload()

    if json_db and processed_ct > 0:
        json_db.write_text(json_dumps(Interrogator.data))

    # collect the weights per file/interrogation of the prior in db stored.
    for entry in ["tag", "rating"]:
        for ent, lst in Interrogator.data[entry].items():
            for index, val in filter(lambda x: x[0] in in_db, map(get_i_wt, lst)):
                in_db[index][entry][ent] = val
    # process the retrieved from db and add them to the stats
    for index in in_db:
        processed_ct += 1
        Interrogator.postprocess(in_db[index]["rating"], "rating", False)
        (count, txt) = Interrogator.postprocess(in_db[index]["tag"], "tag", False)
        if verbose:
            print(f'{in_db[index]["path"]} (redo): {count}/{len(tags)} tags kept')

        if auto_save_tags:
            in_db[index]["out_path"].write_text(txt, encoding='utf-8')

    print('all done :)')
    return interrogator.results(count=processed_ct)


def on_interrogate_image_change(*args):
    # FIXME for some reason an image change is triggered twice.
    # this is a dirty hack to prevent summation/flushing the output.
    print("interrogator: "+args[1])
    Interrogator.image_counter += 1
    if Interrogator.image_counter & 1 == 0:
        # in db.json ratings can be 2x too high
        res = Interrogator.results()
        return res
    return on_interrogate_image(*args)


def on_interrogate_image(
    image: Image,
    interrogator: str,
    unload_model_after_run: bool,
    *args
):
    if image is None:
        return [None, None, None, 'No image']

    if interrogator not in utils.interrogators:
        return [None, None, None, f"'{interrogator}': invalid interrogator"]

    interrogator: Interrogator = utils.interrogators[interrogator]

    key = get_file_interrogator_id(image.tobytes(), interrogator.name)
    file_was_already_interrogated = key in Interrogator.data

    Interrogator.init_filters(*args)

    if file_was_already_interrogated:
        print("already interrogated")
        # this file was already queried for this interrogator.
        idx = Interrogator.data[key][1]
        tags = {}
        ratings = {}

        for ent, lst in Interrogator.data["tag"].items():
            x = next((x for i, x in map(get_i_wt, lst) if i == idx), None)
            if x is not None:
                tags[ent] = x
        for ent, lst in Interrogator.data["rating"].items():
            x = next((x for i, x in map(get_i_wt, lst) if i == idx), None)
            if x is not None:
                ratings[ent] = x
    else:
        # single process
        ratings, tags = interrogator.interrogate(image)

        if unload_model_after_run:
            interrogator.unload()

    Interrogator.postprocess(tags, "tag", not file_was_already_interrogated)
    Interrogator.postprocess(ratings, "rating", not file_was_already_interrogated)
    if not file_was_already_interrogated:
        Interrogator.data[key] = ('', Interrogator.ct)

    return interrogator.results()


def split_str(string: str, separator=',') -> List[str]:
    return [x.strip() for x in string.split(separator) if x]


class Interrogator:
    threshold = 0.35
    count_threshold = 100
    input_glob = ''
    output_filename_format = ''
    re_flags = IGNORECASE
    additional_tags = []
    exclude_tags = []
    search_tags = []
    replace_tags = []
    replace_underscore_excludes = set()
    rord = True
    replace_underscore = True
    tagger_escape = False
    image_counter = 0

    ct = 0
    data = {"tag": {}, "rating": {}}
    filt = {"tag": {}, "rating": {}}
    paths = []
    all_interrogator_names = {}
    error = ''

    @classmethod
    def get_func(cls, tab):
        if tab == "image":
            return on_interrogate_image
        else:
            return on_interrogate

    @classmethod
    def interrogator_id(cls, name):
        if name in cls.all_interrogator_names:
            return cls.all_interrogator_names[name]

        count = len(cls.all_interrogator_names)
        cls.all_interrogator_names[name] = count
        return count

    @classmethod
    def get_paths(cls, filename):
        path = Path(filename)
        # guess the output path
        base_dir_last_idx = path.parts.index(cls.base_dir_last)
        output_dir = cls.output_dir_root.joinpath(
            *path.parts[base_dir_last_idx + 1:]).parent

        output_dir.mkdir(0o755, True, True)
        # format output filename
        format_info = tagger_format.Info(path, 'txt')
        out_filename_fmt = getattr(shared.opts, 'tagger_out_filename_fmt', '')
        try:
            formatted_output_filename = tagger_format.pattern.sub(
                lambda m: tagger_format.format(m, format_info),
                out_filename_fmt
            )
        except (TypeError, ValueError) as error:
            print(str(error))
            return (None, None)

        return (path, output_dir.joinpath(formatted_output_filename))

    @classmethod
    def iter(cls):
        return filter(lambda x: x[0], map(cls.get_paths, cls.paths))

    @classmethod
    def init_filters(
        cls,
        additional_tags: str,
        exclude_tags: str,
        search_tags: str,
        replace_tags: str,

        threshold: float,
        count_threshold: int,
        sort_by_alphabetical_order=False,
    ):
        ignore_case = getattr(shared.opts, 'tagger_re_ignore_case', True)
        cls.re_flags = IGNORECASE if ignore_case else 0
        cls.sort_by_alphabetical_order = sort_by_alphabetical_order
        cls.replace_underscore = getattr(shared.opts, 'tagger_repl_us', True)
        cls.tagger_escape = getattr(shared.opts, 'tagger_escape', False)

        cls.additional_tags = split_str(additional_tags)
        exclude_tags = split_str(exclude_tags)
        search_tags = split_str(search_tags)
        replace_tags = split_str(replace_tags)

        ruxs = getattr(shared.opts, 'tagger_repl_us_excl', '')
        cls.replace_underscore_excludes = set(split_str(ruxs))

        cls.threshold = threshold
        cls.count_threshold = count_threshold
        cls.filt = {"tag": {}, "rating": {}}

        # the first entry can be a regexp, if a string like /.*/
        if len(exclude_tags) == 1:
            regexp_str = '^'+exclude_tags[0]+'$'
            cls.exclude_tags = re_comp(regexp_str, flags=cls.re_flags)
        else:
            cls.exclude_tags = set(exclude_tags)

        rlen = len(replace_tags)
        if rlen == 1:
            # if we replace only with one, we assume a regexp
            cls.replace_tags = replace_tags[0]
            alts = '|'.join(search_tags)
            cls.search_tags = re_comp('^('+alts+')$', flags=cls.re_flags)
        elif len(search_tags) == rlen:
            cls.search_tags = dict((search_tags[i], i) for i in range(rlen))
        else:
            print("search and replace strings have different counts, ignored")
            cls.search_tags = {}
            cls.replace_tags = []

    @classmethod
    def test_reinit(
        cls, input_glob: str, batch_output_dir: str, cumulative_mean: bool,
        *args
    ):
        # batch process, make sure directory is the same
        input_glob = input_glob.strip()
        if input_glob == '':
            return 'Input directory is empty'

        filename_fmt = getattr(shared.opts, 'tagger_out_filename_fmt', '')
        cls.output_filename_format = filename_fmt.strip()
        # if there is no glob pattern, insert it automatically
        if not input_glob.endswith('*'):
            if not input_glob.endswith(os.sep):
                input_glob += os.sep
            input_glob += '*'

        # get root directory of input glob pattern
        base_dir = input_glob.replace('?', '*')
        base_dir = base_dir.split(os.sep + '*').pop(0)
        if not os.path.isdir(base_dir):
            return 'Invalid input directory'

        cls.init_filters(*args)

        output_dir = batch_output_dir.strip()
        if not output_dir:
            output_dir = base_dir

        cls.output_dir_root = Path(output_dir)
        cls.base_dir_last = Path(base_dir).parts[-1]

        # Any change to input directory images should cause a reinit
        # maybe could make checksums per file but then adding or changing
        # file(s) would require a lot of administration
        if not cumulative_mean or input_glob != cls.input_glob:
            cls.ct = 0
            cls.data = {"tag": {}, "rating": {}}
            cls.input_glob = input_glob
            cls.paths = []
            recursive = getattr(shared.opts, 'tagger_batch_recursive', '')
            for path in glob(input_glob, recursive=recursive):
                if '.' + path.split('.').pop().lower() in supported_extensions:
                    cls.paths.append(path)

            print(f'found {len(cls.paths)} image(s)')

    @classmethod
    def correct_tag(cls, tag):
        if tag not in cls.replace_underscore_excludes:
            tag = tag.replace('_', ' ')

        if cls.tagger_escape:
            tag = tag_escape_pattern.sub(r'\\\1', tag)

        if len(cls.replace_tags) != 1:
            if tag in cls.search_tags:
                tag = cls.replace_tags[cls.search_tags[tag]]

        elif re_match(cls.search_tags, tag):
            tag = re_sub(cls.search_tags, cls.replace_tags, tag, 1)

        return tag

    @classmethod
    def is_skipped(cls, tag: str, val: float):
        if val < cls.threshold:
            return True

        if isinstance(cls.exclude_tags, set):
            return tag in cls.exclude_tags

        return re_match(cls.exclude_tags, tag)

    @classmethod
    def postprocess(
        cls, data: Dict[str, float], entry: str, do_store: bool
    ) -> Dict[str, float]:

        if entry == "tag":
            max_ct = cls.count_threshold - len(cls.additional_tags)
        else:
            max_ct = 1000
        torv = 0 if cls.sort_by_alphabetical_order else 1
        rord = not cls.sort_by_alphabetical_order
        lst = sorted(data.items(), key=lambda x: x[torv], reverse=rord)
        processed_ct = 0
        for_json = ""
        # loop with db update
        for ent, val in lst:
            if do_store:
                if val <= 0.005:
                    continue

                if ent not in cls.data[entry]:
                    cls.data[entry][ent] = []

                cls.data[entry][ent].append(val + cls.ct)
            if processed_ct < max_ct:
                if entry == "tag":
                    ent = cls.correct_tag(ent)
                    if cls.is_skipped(ent, val):
                        continue
                    for_json += ", " + ent
                    processed_ct += 1
                if ent not in cls.filt[entry]:
                    cls.filt[entry][ent] = 0.0

                cls.filt[entry][ent] += val
            elif not do_store:
                break
        if entry == "tag":
            for tag in cls.additional_tags:
                if tag not in cls.filt[entry]:
                    cls.filt[entry][tag] = 0.0
                cls.filt[entry][tag] += 1.0
        return (processed_ct, for_json[2:])

    @classmethod
    def results(cls, count=1):
        if count > 1:
            # average
            for entry in cls.filt:
                for ent in cls.filt[entry]:
                    cls.filt[entry][ent] /= count

        s = ', '.join(cls.filt["tag"].keys())
        return [s, cls.filt["rating"], cls.filt["tag"], '']

    def __init__(self, name: str) -> None:
        self._threshold = 0.35
        self._tag_count_threshold = 100
        self.name = name

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
            tf.config.experimental.set_memory_growth(device, True)

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
            tags[tag] = confidences[i]

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

        model_path = Path(hf_hub_download(
            **self.kwargs, filename=self.model_path))
        tags_path = Path(hf_hub_download(
            **self.kwargs, filename=self.tags_path))
        return model_path, tags_path

    def load(self) -> None:
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

        print(f'Loaded {self.name} model from {model_path}')

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
