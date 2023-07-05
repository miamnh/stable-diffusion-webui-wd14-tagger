""" for handling ui settings """

from typing import List, Dict, Tuple, Callable
import os
from pathlib import Path
from glob import glob
from math import ceil
from re import compile as re_comp, sub as re_sub, match as re_match, IGNORECASE
from json import dumps, loads
from PIL import Image
from modules import shared
from modules.deepbooru import re_special as tag_escape_pattern
from functools import partial

from tagger import format as tags_format

from tagger import settings
Its = settings.InterrogatorSettings

# PIL.Image.registered_extensions() returns only PNG if you call early
supported_extensions = {
    e
    for e, f in Image.registered_extensions().items()
    if f in Image.OPEN
}

# interrogator return type
it_ret_tp = Tuple[
    str,               # tags as string
    Dict[str, float],  # rating confidences
    Dict[str, float],  # tag confidences
    str,               # error message
]


class IOData:
    """ data class for input and output paths """
    last_input_glob = None
    base_dir = None
    output_root = None
    paths = []
    save_tags = True

    @classmethod
    def flip_save_tags(cls) -> callable:
        def toggle():
            cls.save_tags = not cls.save_tags
        return toggle

    @classmethod
    def toggle_save_tags(cls) -> None:
        cls.save_tags = not cls.save_tags

    @classmethod
    def update_output_dir(cls, output_dir: str) -> str:
        """ update output directory, and set input and output paths """
        pout = Path(output_dir)
        if pout != cls.output_root:
            paths = [x[0] for x in cls.paths]
            cls.paths = []
            cls.output_root = pout
            err = cls.set_batch_io(paths)
            return err
        return ''

    @classmethod
    def update_input_glob(cls, input_glob: str) -> str:
        """ update input glob pattern, and set input and output paths """
        input_glob = input_glob.strip()
        if input_glob == cls.last_input_glob:
            print('input glob did not change')
            return ''
        last_input_glob = input_glob

        cls.paths = []

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

        if cls.output_root is None:
            output_dir = base_dir

            cls.output_root = Path(output_dir)
        elif not cls.output_root or cls.output_root == Path(cls.base_dir):
            cls.output_root = Path(base_dir)

        cls.base_dir_last = Path(base_dir).parts[-1]
        cls.base_dir = base_dir
        err = QData.read_json(cls.output_root)
        if err != '':
            return err

        recursive = getattr(shared.opts, 'tagger_batch_recursive', '')
        paths = glob(input_glob, recursive=recursive)
        print(f'found {len(paths)} image(s)')
        err = cls.set_batch_io(paths)
        if err == '':
            cls.last_input_glob = last_input_glob

        return err

    @classmethod
    def set_batch_io(cls, paths: List[Path]) -> str:
        """ set input and output paths for batch mode """
        checked_dirs = set()
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext in supported_extensions:
                path = Path(path)
                if not cls.save_tags:
                    cls.paths.append([path, '', ''])
                    continue

                # guess the output path
                base_dir_last_idx = path.parts.index(cls.base_dir_last)
                # format output filename

                info = tags_format.Info(path, 'txt')
                fm = partial(lambda info, m: tags_format.parse(m, info), info)

                try:
                    formatted_output_filename = tags_format.pattern.sub(
                        fm,
                        Its.output_filename_format
                    )
                except (TypeError, ValueError) as error:
                    return f"{path}: output format: {str(error)}"

                output_dir = cls.output_root.joinpath(
                    *path.parts[base_dir_last_idx + 1:]).parent

                tags_out = output_dir.joinpath(formatted_output_filename)

                if output_dir in checked_dirs:
                    cls.paths.append([path, tags_out, ''])
                else:
                    checked_dirs.add(output_dir)
                    if os.path.exists(output_dir):
                        if os.path.isdir(output_dir):
                            cls.paths.append([path, tags_out, ''])
                        else:
                            return f"{output_dir}: not a directory."
                    else:
                        cls.paths.append([path, tags_out, output_dir])
            elif ext != '.txt' and 'db.json' not in path:
                print(f'{path}: not an image extension: "{ext}"')
        return ''


def get_i_wt(stored: float) -> Tuple[int, float]:
    """
    in db.json or InterrogationDB.weighed, with weights + increment in the list
    similar for the "query" dict. Same increment per filestamp-interrogation.
    """
    i = ceil(stored) - 1
    return (i, stored - i)


class QData:
    """ Query data: contains parameters for the query """
    add_tags = []
    excl_tags = set()
    rexcl = None
    srch_tags = {}
    repl_tags = []
    re_srch = None
    threshold = 0.35
    count_threshold = 100

    json_db = None
    weighed = ({}, {})
    query = {}
    data = None
    ratings = {}
    tags = {}
    inverse = False

    @classmethod
    def set(cls, key: str) -> Callable[[str], Tuple[str]]:
        def setter(val) -> Tuple[str]:
            setattr(cls, key, val)
            return ('')
        return setter

    @classmethod
    def update_add(cls, add: str) -> str:
        cls.add_tags = [x.strip() for x in add.split(',')]
        return ''

    @classmethod
    def update_exclude(cls, exclude: str) -> str:
        cls.excl_tags = set(map(lambda x: x.strip(), exclude.split(',')))
        if len(cls.excl_tags) == 1:
            cls.rexcl = re_comp('^'+cls.excl_tags.pop()+'$', flags=IGNORECASE)
        else:
            cls.rexcl = None
        return ''

    @classmethod
    def update_search(cls, search: str) -> str:
        srch_map = [x.strip() for x in search.split(',')]
        cls.srch_tags = dict(enumerate(srch_map))
        slen = len(cls.srch_tags)
        if slen == 1:
            regex = '^'+cls.srch_tags.pop()+'$'
            cls.re_srch = re_comp(regex, flags=IGNORECASE)
        elif slen != len(cls.repl_tags):
            return 'search, replace: unequal len, replacements > 1.'
        return ''

    @classmethod
    def update_replace(cls, replace: str) -> str:
        repl_tag_map = [x.strip() for x in replace.split(',')]
        cls.repl_tags = list(repl_tag_map)
        if cls.re_srch is None and len(cls.srch_tags) != len(cls.repl_tags):
            return 'search, replace: unequal len, replacements > 1.'

    @classmethod
    def read_json(cls, outdir) -> str:
        """ read db.json if it exists """
        cls.json_db = None
        if getattr(shared.opts, 'tagger_auto_serde_json', True):
            cls.json_db = outdir.joinpath('db.json')
            if cls.json_db.is_file():
                try:
                    data = loads(cls.json_db.read_text())
                    if "tag" not in data or "rating" not in data or len(data) < 3:
                        raise TypeError
                except Exception as err:
                    return f'error reading {cls.json_db}: {repr(err)}'
                cls.weighed = (data["tag"], data["rating"])
                cls.query = data["query"]
        return ''

    @classmethod
    def move_filter_to_exclude(cls) -> None:
        """ move filter tags to exclude tags """

        cls.excl_tags.update()

    @classmethod
    def get_index(cls, fi_key: str, path='') -> int:
        """ get index for filestamp-interrogator """
        if path and path != cls.query[fi_key][0]:
            if cls.query[fi_key][0] != '':
                print(f'Dup or rename: Identical checksums for {path}\n'
                      'and: {cls.query[fi_key][0]} (path updated)')
            cls.query[fi_key][0] = path

        # this file was already queried for this interrogator.
        return cls.query[fi_key][1]

    @classmethod
    def init_query(cls) -> None:
        cls.tags.clear()
        cls.ratings.clear()

    @classmethod
    def apply_filters(
        cls,
        data,
        fi_key: str,
        on_avg: bool,
    ):
        """ apply filters to query data, store in db.json if required """

        for_tags_file = ""
        replace_underscore = getattr(shared.opts, 'tagger_repl_us', True)

        tags = sorted(data[3].items(), key=lambda x: x[1], reverse=True)
        if cls.inverse:
            # loop with db update
            for ent, val in tags:
                if replace_underscore and ent not in Its.kamojis:
                    ent = ent.replace('_', ' ')

                if getattr(shared.opts, 'tagger_escape', False):
                    ent = tag_escape_pattern.sub(r'\\\1', ent)

                if cls.re_srch:
                    ent = re_sub(cls.re_srch, cls.repl_tags[0], ent, 1)
                elif ent in cls.srch_tags:
                    ent = cls.repl_tags[cls.srch_tags[ent]]
                if on_avg or val < cls.threshold:
                    for_tags_file += ", " + ent
                elif cls.rexcl and re_match(cls.rexcl, ent):
                    for_tags_file += ", " + ent
                elif ent in cls.excl_tags:
                    for_tags_file += ", " + ent
                else:
                    continue
                cls.tags[ent] = cls.tags[ent] + val if ent in cls.tags else val
            return

        do_store = fi_key != ''
        count = 0
        max_ct = QData.count_threshold - len(cls.add_tags)
        ratings = sorted(data[2].items(), key=lambda x: x[1], reverse=True)
        # loop with db update
        for ent, val in ratings:
            if do_store:
                if ent not in cls.weighed[0]:
                    cls.weighed[0][ent] = []

                cls.weighed[0][ent].append(val + len(cls.query))
            if ent not in cls.ratings:
                cls.ratings[ent] = 0.0

            cls.ratings[ent] += val

        # loop with db update
        for ent, val in tags:
            if do_store:
                if val > 0.005:
                    if ent not in cls.weighed[1]:
                        cls.weighed[1][ent] = []

                    cls.weighed[1][ent].append(val + len(cls.query))

            if count < max_ct:
                if replace_underscore and ent not in Its.kamojis:
                    ent = ent.replace('_', ' ')

                if getattr(shared.opts, 'tagger_escape', False):
                    ent = tag_escape_pattern.sub(r'\\\1', ent)

                if cls.re_srch:
                    ent = re_sub(cls.re_srch, cls.repl_tags[0], ent, 1)
                elif ent in cls.srch_tags:
                    ent = cls.repl_tags[cls.srch_tags[ent]]
                if on_avg or val >= cls.threshold:
                    if cls.rexcl and re_match(cls.rexcl, ent):
                        continue
                    elif ent in cls.excl_tags:
                        continue
                    for_tags_file += ", " + ent
                    count += 1
            elif not do_store:
                break
            cls.tags[ent] = cls.tags[ent] + val if ent in cls.tags else val

        for tag in cls.add_tags:
            cls.tags[tag] = 1.0

        if getattr(shared.opts, 'tagger_verbose', True):
            print(f'{data[0]}: {count}/{len(tags)} tags kept')

        if do_store:
            cls.query[fi_key] = [data[0], len(cls.query)]

        if data[1]:
            data[1].write_text(for_tags_file[2:], encoding='utf-8')

    @classmethod
    def finalize_batch(
        cls,
        in_db,
        ct: int,
        on_avg: bool
    ) -> it_ret_tp:
        if cls.json_db and ct > 0:
            cls.json_db.write_text(dumps({
                "tag": cls.weighed[0],
                "rating": cls.weighed[1],
                "query": cls.query
            }))

        # collect the weights per file/interrogation of the prior in db stored.
        for index in range(2):
            for ent, lst in cls.weighed[index].items():
                for i, val in map(get_i_wt, lst):
                    if i in in_db:
                        in_db[i][2+index][ent] = val

        # process the retrieved from db and add them to the stats
        for got in in_db.values():
            cls.apply_filters(got, '', on_avg)

        # average
        return cls.finalize(ct + len(in_db), on_avg)

    @classmethod
    def finalize(cls, count: int, on_avg: bool) -> it_ret_tp:
        output = ['', {}, {}, '']
        for k, val in cls.tags.items():
            weight = val / count
            if not on_avg or weight >= cls.threshold:
                output[2][k] = weight
                output[0] = output[0] + ', ' + k if output[0] else k

        for ent, val in cls.ratings.items():
            output[1][ent] = val / count
        print('all done :)')
        return output
