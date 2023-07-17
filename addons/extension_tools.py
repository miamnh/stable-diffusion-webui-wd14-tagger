"""Tools for generating html"""  # 一些生成html的工具
# 由 WSH032 慷慨提供: graciously provided by WSH032
import os
from typing import Callable

extensions_dir = os.path.dirname(os.path.abspath(__file__))


def webpath(path: str) -> str:
    # 将path转为webpath，会带上修改时间戳:
    """Converting path to webpath will bring modification timestamp"""
    html_path = path.replace('\\', '/')
    return f'file={html_path}?{os.path.getmtime(path)}'


def javascript_html(js_path: str) -> str:
    """Convert js file to html string"""  # 将js文件转为html字符串
    head = ""
    head += f'<script type="text/javascript" src="{webpath(js_path)}"></script>\n'
    return head


def css_html(css_path: str) -> str:
    """Convert css file to html string"""  # 将css文件转为html字符串
    head = ""
    head += f'<link rel="stylesheet" property="stylesheet" href="{webpath(css_path)}">'
    return head


def dir_path2html(dir: str, ext: str, html_func: Callable[[str], str]) -> str:
    # 将文件夹内的所有文件转为html字符串
    #
    # dir: str, 文件夹路径
    # ext: str, 文件扩展名，如".js"
    # html_func: Callable, 接受一个文件路径，返回一个html字符串
    #    拼接时候，如果需要换行，请在html_func内部加上换行符
    """
    Convert all files in the folder to html strings

    dir: str, folder path
    ext: str, file extension, such as ".js"
    html_func: Callable, accepts a file path and returns an html string
        If you need to wrap, please add a line break in the html_func
    """

    # 该文件夹内所有js文件: All js files in the folder
    js_files_list = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if f.endswith(ext) and os.path.isfile(os.path.join(dir, f))
    ]
    # 转为绝对路径: Convert to absolute path
    js_files_list = [os.path.abspath(f) for f in js_files_list]
    # 生成html: Generate html
    js_html_list = [html_func(js_file) for js_file in js_files_list]
    # 拼接，注意每个元素内的js字符串末尾自带了换行符，所以不需要在这里加:
    # Splicing; note that the js string entries are already newline delimited
    js_str = "".join(js_html_list)

    return js_str
