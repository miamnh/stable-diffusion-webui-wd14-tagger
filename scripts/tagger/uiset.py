""" for handling ui settings """

from typing import List, Dict, Tuple, Callable, Set, Optional
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

from scripts.tagger import format as tags_format

from scripts.tagger import settings
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
    err = set()

    @classmethod
    def flip_save_tags(cls) -> callable:
        def toggle():
            cls.save_tags = not cls.save_tags
        return toggle

    @classmethod
    def toggle_save_tags(cls) -> None:
        cls.save_tags = not cls.save_tags

    @classmethod
    def update_output_dir(cls, output_dir: str) -> None:
        """ update output directory, and set input and output paths """
        pout = Path(output_dir)
        if pout != cls.output_root:
            paths = [x[0] for x in cls.paths]
            cls.paths = []
            cls.output_root = pout
            cls.set_batch_io(paths)

    @classmethod
    def get_bytes_hash(cls, data) -> str:
        """ get sha256 checksum of file """
        # Note: the checksum from an image is not the same as from file
        return sha256(data).hexdigest()

    @classmethod
    def get_hashes(cls) -> Set[str]:
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
    def update_input_glob(cls, input_glob: str) -> None:
        """ update input glob pattern, and set input and output paths """
        input_glob = input_glob.strip()
        if input_glob == cls.last_input_glob:
            print('input glob did not change')
            return
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
        msg = 'Invalid input directory'
        if not os.path.isdir(base_dir):
            cls.err.add(msg)
            return
        cls.err.discard(msg)

        if cls.output_root is None:
            output_dir = base_dir

            cls.output_root = Path(output_dir)
        elif not cls.output_root or cls.output_root == Path(cls.base_dir):
            cls.output_root = Path(base_dir)

        cls.base_dir_last = Path(base_dir).parts[-1]
        cls.base_dir = base_dir
        QData.read_json(cls.output_root)

        recursive = getattr(shared.opts, 'tagger_batch_recursive', '')
        paths = glob(input_glob, recursive=recursive)
        print(f'found {len(paths)} image(s)')
        cls.set_batch_io(paths)
        if len(cls.err) == 0:
            cls.last_input_glob = last_input_glob
            QData.tags.clear()
            QData.ratings.clear()
            QData.in_db.clear()

        return

    @classmethod
    def set_batch_io(cls, paths: List[Path]) -> None:
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

                msg = 'Invalid output format'
                cls.err.discard(msg)
                try:
                    formatted_output_filename = tags_format.pattern.sub(
                        fmt,
                        Its.output_filename_format
                    )
                except (TypeError, ValueError):
                    cls.err.add(msg)

                output_dir = cls.output_root.joinpath(
                    *path.parts[base_dir_last_idx + 1:]).parent

                tags_out = output_dir.joinpath(formatted_output_filename)

                if output_dir in checked_dirs:
                    cls.paths.append([path, tags_out, ''])
                else:
                    checked_dirs.add(output_dir)
                    if os.path.exists(output_dir):
                        msg = 'output_dir: not a directory.'
                        if os.path.isdir(output_dir):
                            cls.paths.append([path, tags_out, ''])
                            cls.err.discard(msg)
                        else:
                            cls.err.add(msg)
                    else:
                        cls.paths.append([path, tags_out, output_dir])
            elif ext != '.txt' and 'db.json' not in path:
                print(f'{path}: not an image extension: "{ext}"')


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
    exclude_tags = []
    search_tags = {}
    replace_tags = []
    threshold = 0.35
    tag_frac_threshold = 0.05

    # read from db.json, update with what should be written to db.json:
    json_db = None
    weighed = (defaultdict(list), defaultdict(list))
    query = {}

    # representing the (cumulative) current interrogations
    ratings = defaultdict(float)
    tags = defaultdict(list)
    in_db = {}
    for_tags_file = defaultdict(lambda: defaultdict(float))

    inverse = False
    had_new = False
    err = set()

    @classmethod
    def set(cls, key: str) -> Callable[[str], Tuple[str]]:
        def setter(val) -> Tuple[str]:
            setattr(cls, key, val)
        return setter

    @classmethod
    def set_attr(cls, current: str, tag: str) -> None:
        """ set attribute for current tag """
        attr = getattr(cls, current + '_tags')
        if current in ['add', 'replace']:
            attr.append(tag)
        elif current == 'keep':
            attr.add(tag)
        else:
            rex = cls.compile_rex(tag)
            if rex:
                if current == 'exclude':
                    attr.append(rex)
                elif current == 'search':
                    attr[len(attr)] = rex
            else:
                cls.err.add(f'empty regex in {current} tags')

    @classmethod
    def test_add(cls, tag: str, current: str, incompatble: list) -> None:
        """ check if there are incompatible collections """
        msg = f'Empty tag in {current} tags'
        if tag == '':
            cls.err.add(msg)
            return
        cls.err.discard(msg)
        for bad in incompatble:
            if current < bad:
                msg = f'"{tag}" is both in {bad} and {current} tags'
            else:
                msg = f'"{tag}" is both in {current} and {bad} tags'
            attr = getattr(cls, bad + '_tags')
            if bad == 'search':
                for rex in attr.values():
                    if rex.match(tag):
                        cls.err.add(msg)
                        return
            elif bad in 'exclude':
                if any(rex.match(tag) for rex in attr):
                    cls.err.add(msg)
                    return
            else:
                if tag in attr:
                    cls.err.add(msg)
                    return
        cls.set_attr(current, tag)

    @classmethod
    def update_keep(cls, keep: str) -> None:
        cls.keep_tags = set()
        msg = 'Empty tag in keep tags'
        if keep == '':
            cls.err.add(msg)
            return
        cls.err.discard(msg)
        un_re = re_comp(r' keep(?: and \w+)? tags')
        cls.err = {err for err in cls.err if not un_re.search(err)}
        for tag in map(str.strip, keep.split(',')):
            cls.test_add(tag, 'keep', ['exclude', 'search'])

    @classmethod
    def update_add(cls, add: str) -> None:
        cls.add_tags = []
        if add == '':
            return
        un_re = re_comp(r' add(?: and \w+)? tags')
        cls.err = {err for err in cls.err if not un_re.search(err)}
        for tag in map(str.strip, add.split(',')):
            cls.test_add(tag, 'add', ['exclude', 'search'])

        # silently raise count threshold to avoid issue in apply_filters
        count_threshold = getattr(shared.opts, 'tagger_count_threshold', 100)
        if len(cls.add_tags) > count_threshold:
            shared.opts.tagger_count_threshold = len(cls.add_tags)

    @classmethod
    def compile_rex(cls, rex: str) -> Optional:
        if rex in {'', '^', '$', '^$'}:
            return None
        if rex[0] == '^':
            rex = rex[1:]
        if rex[-1] == '$':
            rex = rex[:-1]
        return re_comp('^'+rex+'$', flags=IGNORECASE)

    @classmethod
    def update_exclude(cls, exclude: str) -> None:
        cls.exclude_tags = []
        if exclude == '':
            return
        un_re = re_comp(r' exclude(?: and \w+)? tags')
        cls.err = {err for err in cls.err if not un_re.search(err)}
        for excl in map(str.strip, exclude.split(',')):
            incompatble = ['add', 'keep', 'search', 'replace']
            cls.test_add(excl, 'exclude', incompatble)

    @classmethod
    def update_search(cls, search_str: str) -> None:
        cls.search_tags = {}
        if search_str == '':
            return
        un_re = re_comp(r' search(?: and \w+)? tags')
        cls.err = {err for err in cls.err if not un_re.search(err)}
        for rex in map(str.strip, search_str.split(',')):
            incompatble = ['add', 'keep', 'exclude', 'replace']
            cls.test_add(rex, 'search', incompatble)

        msg = 'Unequal number of search and replace tags'
        if len(cls.search_tags) != len(cls.replace_tags):
            cls.err.add(msg)
        else:
            cls.err.discard(msg)

    @classmethod
    def update_replace(cls, replace: str) -> None:
        cls.replace_tags = []
        if replace == '':
            return
        un_re = re_comp(r' replace(?: and \w+)? tags')
        cls.err = {err for err in cls.err if not un_re.search(err)}
        for repl in map(str.strip, replace.split(',')):
            cls.test_add(repl, 'replace', ['exclude', 'search'])
        msg = 'Unequal number of search and replace tags'
        if len(cls.search_tags) != len(cls.replace_tags):
            cls.err.add(msg)
        else:
            cls.err.discard(msg)

    @classmethod
    def read_json(cls, outdir) -> None:
        """ read db.json if it exists """
        cls.json_db = None
        if getattr(shared.opts, 'tagger_auto_serde_json', True):
            cls.json_db = outdir.joinpath('db.json')
            if cls.json_db.is_file():
                cls.had_new = False
                msg = f'Error reading {cls.json_db}'
                cls.err.discard(msg)
                try:
                    data = loads(cls.json_db.read_text())
                except JSONDecodeError as err:
                    print(f'{msg}: {repr(err)}')
                    cls.err.add(msg)
                    return
                for key in ["tag", "rating", "query"]:
                    msg = f'{cls.json_db}: missing {key} key.'
                    if key not in data:
                        cls.err.add(msg)
                    else:
                        cls.err.discard(msg)
                for key in ["add", "keep", "exclude", "search", "replace"]:
                    if key in data and data[key] != '':
                        getattr(cls, f"update_{key}")(data[key])
                cls.weighed = (
                    defaultdict(list, data["rating"]),
                    defaultdict(list, data["tag"])
                )
                cls.query = data["query"]
                print(f'Read {cls.json_db}: {len(cls.query)} interrogations, '
                      f'{len(cls.tags)} tags.')

    @classmethod
    def write_json(cls) -> None:
        """ write db.json """
        if cls.json_db is not None:
            search = sorted(cls.search_tags.items(), key=lambda x: x[0])
            search = [x[1].pattern for x in search]
            exclude = [x.pattern for x in cls.exclude_tags]
            data = {
                "rating": cls.weighed[0],
                "tag": cls.weighed[1],
                "query": cls.query,
                "add": ','.join(cls.add_tags),
                "keep": ','.join(cls.keep_tags),
                "exclude": ','.join(exclude),
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
        return any(re_match(x, ent) for x in cls.exclude_tags)

    @classmethod
    def correct_tag(cls, tag: str) -> str:
        """ correct tag for display """
        replace_underscore = getattr(shared.opts, 'tagger_repl_us', True)
        if replace_underscore and tag not in Its.kamojis:
            tag = tag.replace('_', ' ')

        if getattr(shared.opts, 'tagger_escape', False):
            tag = tag_escape_pattern.sub(r'\\\1', tag)

        for i, regex in cls.search_tags.items():
            if re_match(regex, tag):
                tag = re_sub(regex, cls.replace_tags[i], tag)
                break

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
        # data = (path, fi_key, tags, ratings, new)
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
                    current = cls.for_tags_file[data[1]].get(tag, 0.0)
                    cls.for_tags_file[data[1]][tag] = min(val + current, 1.0)
                count += 1
            elif fi_key == '':
                break
            if tag not in cls.add_tags:
                # those are already added
                cls.tags[tag].append(val)

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
        if cls.inverse:
            return cls.finalize_inverse(count)
        return cls.finalize(count)

    @classmethod
    def sort_tags(cls, tags: Dict[str, float]) -> List[Tuple[str, float]]:
        """ sort tags by value, return list of tuples """
        return sorted(tags.items(), key=lambda x: x[1], reverse=True)

    @classmethod
    def finalize(cls, count: int) -> ItRetTP:
        """ finalize the query, return the results """
        count += len(cls.in_db)
        if count == 0:
            return [None, None, None, 'no results for query']

        tags_str, ratings, tags = '', {}, {}

        for val in cls.for_tags_file.values():
            for k in cls.add_tags:
                val[k] = 1.0 * count

        for k in cls.add_tags:
            escaped = html_escape(k)
            tags_str += f""", <a href='javascript:tag_clicked("{k}", """\
                        f"""true)'>{escaped}</a>"""
            tags[k] = 1.0

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
                            f"""true)'>{escaped}</a>"""
            else:
                for remaining_tags in cls.for_tags_file.values():
                    if k in remaining_tags:
                        if k not in cls.add_tags and k not in cls.keep_tags:
                            del remaining_tags[k]

        for ent, val in cls.ratings.items():
            ratings[ent] = val / count

        weighted_tags_files = getattr(shared.opts,
                                      'tagger_weighted_tags_files', False)
        for file, remaining_tags in cls.for_tags_file.items():
            sorted_tags = cls.sort_tags(remaining_tags)
            if weighted_tags_files:
                sorted_tags = [f'({k}:{v})' for k, v in sorted_tags]
            else:
                sorted_tags = [k for k, v in sorted_tags]
            file.write_text(', '.join(sorted_tags), encoding='utf-8')

        print('all done :)')
        return (tags_str[2:], ratings, tags, '')

    @classmethod
    def finalize_inverse(cls, count: int) -> ItRetTP:
        """ finalize the query, return the results """
        count += len(cls.in_db)
        if count == 0:
            return [None, None, None, 'no results for query']

        tags_str, ratings, tags = '', {}, {}

        for k, lst in cls.tags.items():
            # len(!) fraction of the all interrogations was above the threshold
            fraction_of_queries = len(lst) / count

            if fraction_of_queries < cls.tag_frac_threshold:
                # store the average of those interrogations sum(!) / count
                tags[k] = sum(lst) / count
                # trigger an event to place the tag in the active tags list
                # replace if k interferes with html code
                escaped = html_escape(k)
                tags_str += f""", <a href='javascript:tag_clicked("{k}", """\
                            f"""false)'>{escaped}</a>"""

        print('all done :)')
        return (tags_str[2:], ratings, tags, '')
