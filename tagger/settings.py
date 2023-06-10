from modules import shared


def on_ui_settings():
    section = 'tagger', 'Tagger'
    shared.opts.add_option(
        key='tagger_out_filename_fmt',
        info=shared.OptionInfo(
            '[name].[output_extension]',
            label='Tag file output format. Leave blank to use same filename or'
            ' e.g. "[name].[hash:sha1].[output_extension]". Also allowed are '
            '[extension] or any other [hash:<algorithm>] supported by hashlib',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_batch_recursive',
        info=shared.OptionInfo(
            True,
            label='Glob recursively with input directory pattern',
            section=section,
        ),
    )

    shared.opts.add_option(
        key='tagger_auto_serde_json',
        info=shared.OptionInfo(
            True,
            label='Auto load and save JSON database',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_verbose',
        info=shared.OptionInfo(
            False,
            label='Console log tag counts per file, no progress bar',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_repl_us',
        info=shared.OptionInfo(
            True,
            label='Use spaces instead of underscore',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_repl_us_excl',
        info=shared.OptionInfo(
            # kaomoji from WD 1.4 tagger csv. thanks, Meow-San#5400!
            '0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, '
            '6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||',
            label='Excudes (split by comma)',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_escape',
        info=shared.OptionInfo(
            False,
            label='Escape brackets',
            section=section,
        ),
    )
    shared.opts.add_option(
        key='tagger_re_ignore_case',
        info=shared.OptionInfo(
            True,
            label='Ignore case in RegExp search/replace',
            section=section,
        ),
    )
