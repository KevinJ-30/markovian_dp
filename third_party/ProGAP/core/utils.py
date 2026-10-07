import math
from typing import Optional
from rich.table import Table
from rich.highlighter import ReprHighlighter
from rich import box
from tabulate import tabulate


def dict2table(input_dict: dict, num_cols: int = 4, title: Optional[str] = None) -> Table:
    num_items = len(input_dict)
    num_rows = math.ceil(num_items / num_cols)
    col = 0
    data = {}
    keys = []
    vals = []

    for i, (key, val) in enumerate(input_dict.items()):
        keys.append(f'{key}:')
        
        vals.append(val)
        if (i + 1) % num_rows == 0:
            data[col] = keys
            data[col+1] = vals
            keys = []
            vals = []
            col += 2

    data[col] = keys
    data[col+1] = vals

    highlighter = ReprHighlighter()
    message = tabulate(data, tablefmt='plain')
    table = Table(title=title, show_header=False, box=box.HORIZONTALS)
    table.add_row(highlighter(message))
    return table
