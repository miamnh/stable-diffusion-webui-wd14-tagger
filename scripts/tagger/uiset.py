""" for handling ui settings """

from typing import List, Dict, Tuple, Callable, Set
import os
from pathlib import Path
from glob import glob
from math import ceil
from hashlib import sha256
from re import compile as re_comp, sub as re_sub, match as re_match, IGNORECASE
from json import dumps, loads, JSONDecodeError
from functools import partial
from html import escape as html_escape
from collections import defaultdict
from PIL import Image
from modules import shared
from modules.deepbooru import re_special as tag_escape_pattern

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
ItRetTP = Tuple[
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
    def get_bytes_hash(cls, data) -> str:
        """ get sha256 checksum of file """
        # Note: the checksum from an image is not the same as from file
        return sha256(data).hexdigest()

    @classmethod
    def get_hashes(cls):
        """ get hashes of all files """
        ret = set()
        for entries in cls.paths:
            if len(entries) == 4:
                ret.add(entries[3])
            else:
                # if there is no checksum, calculate it
                image = Image.open(entries[0])
                checksum = cls.get_bytes_hash(image.tobytes())
                entries.append(checksum)
                ret.add(checksum)
        return ret

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
                fmt = partial(lambda info, m: tags_format.parse(m, info), info)

                try:
                    formatted_output_filename = tags_format.pattern.sub(
                        fmt,
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
    return i, stored - i


class QData:
    """ Query data: contains parameters for the query """
    add_tags = []
    keep_tags = set()
    exclude_tags = set()
    rexcl = None
    search_tags = {}
    replace_tags = []
    re_search = None
    threshold = 0.35
    tag_frac_threshold = 0.05

    # read from db.json, update with what should be written to db.json:
    json_db = None
    weighed = (defaultdict(lambda: []), defaultdict(lambda: []))
    query = {}

    # representing the (cumulative) current interrogations
    ratings = defaultdict(lambda: 0.0)
    tags = defaultdict(lambda: [])
    in_db = {}
    for_tags_file = defaultdict(set)

    inverse = False
    had_new = False

    @classmethod
    def set(cls, key: str) -> Callable[[str], Tuple[str]]:
        def setter(val) -> Tuple[str]:
            setattr(cls, key, val)
        return setter

    @classmethod
    def update_keep(cls, keep: str) -> str:
        cls.keep_tags = {x for x in map(str.strip, keep.split(',')) if x != ''}

    @classmethod
    def update_add(cls, add: str) -> str:
        cls.add_tags = [x for x in map(str.strip, add.split(',')) if x != '']
        count_threshold = getattr(shared.opts, 'tagger_count_threshold', 100)
        if len(cls.add_tags) > count_threshold:
            # secretly raise count threshold to avoid issue in apply_filters
            shared.opts.tagger_count_threshold = len(cls.add_tags)

    @classmethod
    def update_exclude(cls, exclude: str) -> str:
        excl = exclude.strip()
        # first filter empty strings
        if ',' in excl:
            filtered = [x for x in map(str.strip, excl.split(',')) if x != '']
            cls.exclude_tags = set(filtered)
            cls.rexcl = None
        elif excl != '':
            cls.rexcl = re_comp('^'+excl+'$', flags=IGNORECASE)

    @classmethod
    def update_search(cls, search: str) -> str:
        search = [x for x in map(str.strip, search.split(',')) if x != '']
        cls.search_tags = dict(enumerate(search))
        slen = len(cls.search_tags)
        if len(cls.search_tags) == 1:
            cls.re_search = re_comp('^'+search[0]+'$', flags=IGNORECASE)
        elif slen != len(cls.replace_tags):
            return 'search, replace: unequal len, replacements > 1.'
        return ''

    @classmethod
    def update_replace(cls, replace: str) -> str:
        repl = [x for x in map(str.strip, replace.split(',')) if x != '']
        cls.replace_tags = list(repl)
        if not cls.re_search and len(cls.search_tags) != len(cls.replace_tags):
            return 'search, replace: unequal len, replacements > 1.'
        return ''

    @classmethod
    def read_json(cls, outdir) -> str:
        """ read db.json if it exists """
        cls.json_db = None
        if getattr(shared.opts, 'tagger_auto_serde_json', True):
            cls.json_db = outdir.joinpath('db.json')
            if cls.json_db.is_file():
                cls.had_new = False
                try:
                    data = loads(cls.json_db.read_text())
                except JSONDecodeError as err:
                    return f'Error reading {cls.json_db}: {repr(err)}'
                for key in ["tag", "rating", "query"]:
                    if key not in data:
                        return f'{cls.json_db}: missing {key} key.'
                for key in ["add", "keep", "exclude", "search", "replace"]:
                    if key in data:
                        err = getattr(cls, f"update_{key}")(data[key])
                        if err:
                            return err
                cls.weighed = (
                    defaultdict(lambda: [], data["rating"]),
                    defaultdict(lambda: [], data["tag"])
                )
                cls.query = data["query"]
                print(f'Read {cls.json_db}: {len(cls.query)} interrogations, '
                      f'{len(cls.tags)} tags.')
        return ''

    @classmethod
    def write_json(cls) -> None:
        """ write db.json """
        if cls.json_db is not None:
            search = sorted(cls.search_tags.items(), key=lambda x: x[0])
            data = {
                "rating": cls.weighed[0],
                "tag": cls.weighed[1],
                "query": cls.query,
                "add": ','.join(cls.add_tags),
                "keep": ','.join(cls.keep_tags),
                "exclude": ','.join(cls.exclude_tags),
                "search": ','.join([x[1] for x in search]),
                "repl": ','.join(cls.replace_tags)
            }
            cls.json_db.write_text(dumps(data, indent=2))
            print(f'Wrote {cls.json_db}: {len(cls.query)} interrogations, '
                  f'{len(cls.tags)} tags.')

    @classmethod
    def get_index(cls, fi_key: str, path='') -> int:
        """ get index for filestamp-interrogator """
        if path and path != cls.query[fi_key][0]:
            if cls.query[fi_key][0] != '':
                print(f'Dup or rename: Identical checksums for {path}\n'
                      f'and: {cls.query[fi_key][0]} (path updated)')
                cls.had_new = True
            cls.query[fi_key] = (path, cls.query[fi_key][1])

        return cls.query[fi_key][1]

    @classmethod
    def single_data(
        cls, fi_key: str
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """ get tags and ratings for filestamp-interrogator """
        index = cls.query.get(fi_key)[1]
        data = ({}, {})
        for j in range(2):
            for ent, lst in cls.weighed[j].items():
                for i, val in map(get_i_wt, lst):
                    if i == index:
                        data[j][ent] = val
        QData.in_db[index] = ('', '', '') + data

    @classmethod
    def is_excluded(cls, ent: str) -> bool:
        """ check if tag is excluded """
        if cls.rexcl:
            return re_match(cls.rexcl, ent)

        return ent in cls.exclude_tags

    @classmethod
    def correct_tag(cls, tag: str) -> str:
        """ correct tag for display """
        replace_underscore = getattr(shared.opts, 'tagger_repl_us', True)
        if replace_underscore and tag not in Its.kamojis:
            tag = tag.replace('_', ' ')

        if getattr(shared.opts, 'tagger_escape', False):
            tag = tag_escape_pattern.sub(r'\\\1', tag)

        if cls.re_search:
            tag = re_sub(cls.re_search, cls.replace_tags[0], tag, 1)
        elif tag in cls.search_tags:
            tag = cls.replace_tags[cls.search_tags[tag]]

        return tag

    @classmethod
    def inverse_apply_filters(cls, tags: List[Tuple[str, float]]) -> None:
        """ inverse: List all tags marked for exclusion """
        for tag, val in tags:
            tag = cls.correct_tag(tag)

            if tag in cls.keep_tags or tag in cls.add_tags:
                continue

            if cls.is_excluded(tag) or val < cls.threshold:
                cls.tags[tag].append(val)

    @classmethod
    def apply_filters(cls, data) -> Set[str]:
        """ apply filters to query data, store in db.json if required """
        # fi_key == '' means this is a new file or interrogation for that file

        tags = sorted(data[4].items(), key=lambda x: x[1], reverse=True)
        if cls.inverse:
            cls.inverse_apply_filters(tags)
            return

        # not inverse: display all tags marked for inclusion

        fi_key = data[2]
        index = len(cls.query)

        ratings = sorted(data[3].items(), key=lambda x: x[1], reverse=True)
        # loop over ratings
        for rating, val in ratings:
            if fi_key != '':
                cls.weighed[0][rating].append(val + index)
            cls.ratings[rating] += val

        count_threshold = getattr(shared.opts, 'tagger_count_threshold', 100)
        max_ct = count_threshold - len(cls.add_tags)
        count = 0
        # loop over tags with db update
        for tag, val in tags:
            if isinstance(tag, float):
                print(f'bad return from interrogator, float: {tag} {val}')
                # FIXME: why does this happen? what does it mean?
                continue

            if fi_key != '' and val >= 0.005:
                cls.weighed[1][tag].append(val + index)

            if count < max_ct:
                tag = cls.correct_tag(tag)
                if tag not in cls.keep_tags:
                    if cls.is_excluded(tag) or val < cls.threshold:
                        continue
                if data[1] != '':
                    cls.for_tags_file[data[1]].add(tag)
                count += 1
            elif fi_key == '':
                break
            if tag not in cls.add_tags:
                # those are already added
                cls.tags[tag].append(val)

        for tag in cls.add_tags:
            if data[1] != '':
                cls.for_tags_file[data[1]].add(tag)
            cls.tags[tag] = [1.0 for _ in range(len(cls.query))]

        if getattr(shared.opts, 'tagger_verbose', True):
            print(f'{data[0]}: {count}/{len(tags)} tags kept')

        if fi_key != '':
            cls.query[fi_key] = (data[0], index)

    @classmethod
    def finalize_batch(cls, count: int) -> ItRetTP:
        """ finalize the batch query """
        if cls.json_db and cls.had_new:
            cls.write_json()
            cls.had_new = False

        # collect the weights per file/interrogation of the prior in db stored.
        for index in range(2):
            for ent, lst in cls.weighed[index].items():
                for i, val in map(get_i_wt, lst):
                    if i not in cls.in_db:
                        continue
                    cls.in_db[i][3+index][ent] = val

        # process the retrieved from db and add them to the stats
        for got in cls.in_db.values():
            cls.apply_filters(got)

        # average
        return cls.finalize(count)

    @classmethod
    def finalize(cls, count: int) -> ItRetTP:
        """ finalize the query, return the results """
        count += len(cls.in_db)
        if count == 0:
            return [None, None, None, 'no results for query']

        tags_str, ratings, tags = '', {}, {}

        js_bool = 'true' if cls.inverse else 'false'

        for k, lst in cls.tags.items():
            # len(!) fraction of the all interrogations was above the threshold
            fraction_of_queries = len(lst) / count

            if fraction_of_queries >= cls.tag_frac_threshold:
                # store the average of those interrogations sum(!) / count
                tags[k] = sum(lst) / count
                # trigger an event to place the tag in the active tags list
                # replace if k interferes with html code
                escaped = html_escape(k)
                tags_str += f""", <a href='javascript:tag_clicked("{k}", """\
                            f"""{js_bool})'>{escaped}</a>"""
            else:
                for remaining_tags in cls.for_tags_file.values():
                    remaining_tags.discard(k)

        for ent, val in cls.ratings.items():
            ratings[ent] = val / count

        for file, remaining_tags in cls.for_tags_file.items():
            file.write_text(', '.join(remaining_tags), encoding='utf-8')

        print('all done :)')
        return (tags_str[2:], ratings, tags, '')
