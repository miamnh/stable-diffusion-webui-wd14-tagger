import os
from pathlib import Path
from io import BytesIO
from glob import glob
from hashlib import sha256
import json
from pandas import read_csv, read_json
from PIL import Image
from typing import Tuple, List, Dict
from numpy import asarray, float32, expand_dims
from tqdm import tqdm
from re import compile as re_comp, sub as re_sub, match as re_match, IGNORECASE

from huggingface_hub import hf_hub_download
from modules.deepbooru import re_special as tag_escape_pattern
from modules import shared

# i'm not sure if it's okay to add this file to the repository
from tagger import utils, format as tagger_format
from . import dbimutils
from tagger import interrogation_data as idb

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

    json_db = getattr(shared.opts, 'tagger_auto_serde_json', True)
    json_db = Interrogator.output_root.joinpath('db.json') if json_db else None
    try:
        Interrogator.data = idb.InterrogationDB(json_db)
    except Exception as e:
        print(f'error reading {json_db}: {repr(e)}')
        if batch_rewrite:
            return [None, None, None, 'No database']
        if Interrogator.data is None:
            Interrogator.data = idb.InterrogationDB()

    verbose = getattr(shared.opts, 'tagger_verbose', True)
    in_db = {}
    processed_ct = Interrogator.data.query_count
    if Interrogator.warn:
        print(f'Got warnings:\n{Interrogator.warn}\ntags files not written.')

    rpl = Interrogator.get_search_replace()

    iter = filter(None, map(Interrogator.get_paths, Interrogator.paths))
    for (path, out_path) in tqdm(iter, disable=verbose, desc='Tagging'):
        try:
            image = Image.open(path)

        except Exception as e:
            # just in case, user has mysterious file...
            print(f'${path} is not supported image type: {e}')
            continue

        abspath = str(path.absolute())
        fi_key = get_file_interrogator_id(image.tobytes(), interrogator_name)

        if fi_key in Interrogator.data:
            # this file was already queried for this interrogator.
            index = Interrogator.data.get_index(fi_key, abspath)
            in_db[index] = [abspath, out_path, {}, {}]

        elif batch_rewrite:
            # with batch rewrite we only read from database
            print(f'new file {abspath}: requires interrogation (skipped)')
        else:
            data = interrogator.interrogate(image)
            txt = Interrogator.postprocess(data, fi_key, rpl, abspath, verbose)

            if auto_save_tags and Interrogator.warn == '':
                out_path.write_text(txt, encoding='utf-8')

    processed_ct = Interrogator.data.query_count - processed_ct

    if unload_model_after_run:
        interrogator.unload()

    if json_db and processed_ct > 0:
        Interrogator.data.write_json(json_db)

    # collect the weights per file/interrogation of the prior in db stored.
    Interrogator.data.collect(in_db)

    # process the retrieved from db and add them to the stats
    for got in in_db.values():
        txt = Interrogator.postprocess(got[2:], '', rpl, got[0], verbose)

        if auto_save_tags and Interrogator.warn == '':
            got[1].write_text(txt, encoding='utf-8')

    print('all done :)')
    return interrogator.results(count=processed_ct + len(in_db))


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

    # if we just interrogated a batch query, it might be in the db
    if Interrogator.data is None:
        Interrogator.data = idb.InterrogationDB()

    fi_key = get_file_interrogator_id(image.tobytes(), interrogator.name)
    Interrogator.init_filters(*args)

    if fi_key in Interrogator.data:
        # this file was already queried for this interrogator.
        data = Interrogator.data.get(fi_key)
        fi_key = ""
    else:
        # single process
        data = interrogator.interrogate(image)

        if unload_model_after_run:
            interrogator.unload()

    rpl = Interrogator.get_search_replace()
    Interrogator.postprocess(data, fi_key, rpl)
    return interrogator.results()


def split_str(string: str, separator=',') -> List[str]:
    return [x.strip() for x in string.split(separator) if x]


class Interrogator:
    threshold = 0.35
    count_threshold = 100
    input_glob = ''
    output_filename_format = '[name].[output_extension]'
    re_flags = IGNORECASE
    tags = {"add": [], "exclude": set(), "search": [], "replace": []}
    replace_underscore_excludes = set()
    rord = True
    replace_underscore = True
    tagger_escape = False
    image_counter = 0

    ct = 0
    data = None
    filt = {"tag": {}, "rating": {}}
    paths = []
    warn = ''

    @classmethod
    def set_add(cls, add_tags):
        cls.tags["add"] = split_str(add_tags)

    @classmethod
    def set_exclude(cls, exclude_tags):
        excl = split_str(exclude_tags)
        # one entry, it is treated as a regexp
        if len(excl) == 1:
            cls.tags["exclude"] = re_comp('^'+excl[0]+'$', flags=cls.re_flags)
        else:
            cls.tags["exclude"] = set(excl)

    @classmethod
    def set_search(cls, search_tags):
        cls.tags["search"] = split_str(search_tags)
        return cls.check_search_replace_lengths()

    @classmethod
    def set_replace(cls, replace_tags):
        cls.tags["replace"] = split_str(replace_tags)
        return cls.check_search_replace_lengths()

    @classmethod
    def check_search_replace_lengths(cls):
        rlen = len(cls.tags["replace"])
        if rlen != 1 and rlen != len(cls.tags["search"]):
            cls.warn = 'search, replace: unequal len, replacements > 1.'
        return [cls.warn]

    @classmethod
    def get_search_replace(cls):
        rlen = len(cls.tags["replace"])
        if rlen == 1:
            # if we replace only with one, we assume a regexp
            replace_tags = cls.tags["replace"][0]
            alts = '|'.join(cls.tags["search"])
            return (re_comp('^('+alts+')$', flags=cls.re_flags), replace_tags)
        else:
            lookup = dict((cls.tags["search"][i], i) for i in range(rlen))
            return (lookup, cls.tags["replace"])

    @classmethod
    def get_paths(cls, filename):
        path = Path(filename)
        # guess the output path
        base_dir_last_idx = path.parts.index(cls.base_dir_last)
        output_dir = cls.output_root.joinpath(
            *path.parts[base_dir_last_idx + 1:]).parent

        output_dir.mkdir(0o755, True, True)
        # format output filename
        format_info = tagger_format.Info(path, 'txt')
        try:
            formatted_output_filename = tagger_format.pattern.sub(
                lambda m: tagger_format.format(m, format_info),
                cls.output_filename_format
            )
        except (TypeError, ValueError) as error:
            print(str(error))
            return None

        return (path, output_dir.joinpath(formatted_output_filename))

    @classmethod
    def init_filters(
        cls,

        threshold: float,
        count_threshold: int,
        sort_by_alphabetical_order=False,
    ):
        ignore_case = getattr(shared.opts, 'tagger_re_ignore_case', True)
        cls.re_flags = IGNORECASE if ignore_case else 0
        cls.sort_by_alphabetical_order = sort_by_alphabetical_order
        cls.replace_underscore = getattr(shared.opts, 'tagger_repl_us', True)
        cls.tagger_escape = getattr(shared.opts, 'tagger_escape', False)

        ruxs = getattr(shared.opts, 'tagger_repl_us_excl', '')
        cls.replace_underscore_excludes = set(split_str(ruxs))

        cls.threshold = threshold
        cls.count_threshold = count_threshold
        cls.filt = {"tag": {}, "rating": {}}


    @classmethod
    def test_reinit(
        cls, input_glob: str, batch_output_dir: str, cumulative_mean: bool,
        *args
    ):
        # batch process, make sure directory is the same
        input_glob = input_glob.strip()
        if input_glob == '':
            return 'Input directory is empty'

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

        filename_fmt = getattr(shared.opts, 'tagger_out_filename_fmt', '')
        if filename_fmt[-12:] == '.[extension]':
            cls.warn += 'Settings: output format ends with image extension.'

        cls.output_filename_format = filename_fmt.strip()

        output_dir = batch_output_dir.strip()
        if not output_dir:
            output_dir = base_dir

        cls.output_root = Path(output_dir)
        cls.base_dir_last = Path(base_dir).parts[-1]

        # Any change to input directory images should cause a reinit
        # maybe could make checksums per file but then adding or changing
        # file(s) would require a lot of administration
        if not cumulative_mean or input_glob != cls.input_glob:
            cls.input_glob = input_glob
            cls.paths = []
            recursive = getattr(shared.opts, 'tagger_batch_recursive', '')
            for path in glob(input_glob, recursive=recursive):
                if '.' + path.split('.').pop().lower() in supported_extensions:
                    cls.paths.append(path)

            print(f'found {len(cls.paths)} image(s)')

    @classmethod
    def correct_tag(cls, tag, search, replace):
        if tag not in cls.replace_underscore_excludes:
            tag = tag.replace('_', ' ')

        if cls.tagger_escape:
            tag = tag_escape_pattern.sub(r'\\\1', tag)

        if len(replace) != 1:
            if tag in search:
                tag = replace[search[tag]]

        elif re_match(search, tag):
            tag = re_sub(search, replace, tag, 1)

        return tag

    @classmethod
    def is_skipped(cls, tag: str, val: float):
        if val < cls.threshold:
            return True

        if isinstance(cls.tags["exclude"], set):
            return tag in cls.tags["exclude"]

        return re_match(cls.tags["exclude"], tag)

    @classmethod
    def postprocess(
        cls, data: (Dict[str, float], Dict[str, float]), fi_key: str, rpl,
        path='', verbose=False
    ) -> Dict[str, float]:
        do_store = fi_key != ''

        torv = 0 if cls.sort_by_alphabetical_order else 1
        rord = not cls.sort_by_alphabetical_order
        for_json = ""
        count = 0
        max_ct = cls.count_threshold - len(cls.tags["add"])

        for i in range(2):
            lst = sorted(data[i].items(), key=lambda x: x[torv], reverse=rord)
            key = "tag" if i else "rating"
            # loop with db update
            for ent, val in lst:
                if do_store:
                    if val <= 0.005:
                        continue
                    cls.data.add(i, ent, val)

                if key == "tag":
                    if count < max_ct:
                        ent = cls.correct_tag(ent, *rpl)
                        if cls.is_skipped(ent, val):
                            continue
                        for_json += ", " + ent
                        count += 1
                    elif not do_store:
                        break
                if ent not in cls.filt[key]:
                    cls.filt[key][ent] = 0.0

                cls.filt[key][ent] += val

        for tag in cls.tags["add"]:
            cls.filt["tag"][tag] = 1.0

        if verbose:
            print(f'{path}: {count}/{len(lst)} tags kept')

        if do_store:
            cls.data.story_query(fi_key, path)

        return for_json[2:]

    @classmethod
    def results(cls, count=1):
        if count > 1:
            # average
            for key in cls.filt:
                for ent in cls.filt[key]:
                    cls.filt[key][ent] /= count

        s = ', '.join(cls.filt["tag"].keys())
        return [s, cls.filt["rating"], cls.filt["tag"], cls.warn]

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
