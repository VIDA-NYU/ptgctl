'''This contains some utilities/patches to make Fire's output look nice.

'''

import fnmatch
import functools

# adds

import sys
import types
import inspect
import fire
from fire.core import FireError, value_types, _OneLineResult, _DictAsString, helptext, Display
def _PrintResult(component_trace, verbose=False):
  """Prints the result of the Fire call to stdout in a human readable way."""
  # TODO(dbieber): Design human readable deserializable serialization method
  # and move serialization to its own module.
  result = component_trace.GetResult()

  # Allow users to modify the return value of the component and provide 
  # custom formatting.
  if serialize:
    if not callable(serialize):
      raise FireError("serialize argument {} must be empty or callable.".format(serialize))
    result = serialize(result)

  if value_types.HasCustomStr(result):
    # If the object has a custom __str__ method, rather than one inherited from
    # object, then we use that to serialize the object.
    print(str(result))
    return

  if isinstance(result, (list, set, frozenset, types.GeneratorType)):
    for i in result:
      print(_OneLineResult(i))
  elif inspect.isgeneratorfunction(result):
    raise NotImplementedError
  elif isinstance(result, dict) and value_types.IsSimpleGroup(result):
    print(_DictAsString(result, verbose))
  elif isinstance(result, tuple):
    print(_OneLineResult(result))
  elif isinstance(result, value_types.VALUE_TYPES):
    if result is not None:
      print(result)
  else:
    help_text = helptext.HelpText(
        result, trace=component_trace, verbose=verbose)
    output = [help_text]
    Display(output, out=sys.stdout)
fire.core._PrintResult = _PrintResult


def serialize(x):
    if value_types.HasCustomStr(x):
        return str(x)
    if isinstance(x, (list, tuple, dict)):
        return yamltable(x)
    return x



# top-level functions

def yamltable(d, *a, indent=0, width=2, depth=-1, _keys=(), **kw):
    '''Format data as yaml. Any list of dicts will be rendered as a table.
    Arguments:
        *a: positional arguments for ``astable``.
        indent (int): the indent index (how many tabs?).
        width (int): tab width.
        depth (int): How many depths to render? If -1, traverse all.
        **kw: keyword arguments for ``astable``.
    Returns:
        output (str): The formatted data.
    '''
    if depth:
        if isinstance(d, dict):
            d = '\n'.join('{}: {}'.format(k, yamltable(
                    d[k], *a, indent=indent+1,
                    width=width, depth=depth-1,
                    _keys=_keys + (k,), **kw))
                for k in d)

        if isinstance(d, list):
            if all(di is None or isinstance(di, dict) for di in d):
                d = astable(d, *a, **kw)
            else:
                d = '\n'.join([' - {}'.format(di) for di in d])

    d = str(d)
    if indent and len(d.splitlines()) > 1:
        d = '\n' + indent_(d, indent=indent > 0, width=width)
    return d



# boolean display

BOOLS = {
    'moon': ['🌖', '🌒'],
    'full-moon': ['🌕', '🌑'],
    'rose': ['🌹', '🥀'],
    'rainbow': ['🌈', '☔️'],
    'octopus': ['🐙', '🐍'],
    'virus': ['🔬', '🦠'],
    'party-horn': ['🎉', '💥'],
    'party-ball': ['🎊', '🧨'],

    'relieved': ['😅', '🥺'],
    'laughing': ['😂', '😰'],
    'elated': ['🥰', '🤬'],
    'fleek': ['💅', '👺'],
    'thumb': ['👍', '👎'],
    'green-heart': ['💚', '💔'],
    'circle': ['🟢', '🔴'],
    'green-check': ['✅', '❗️'],
    'TF': ['T', 'F'],
    'tf': ['t', 'f'],
    'YN': ['Y', 'N'],
    'yn': ['y', 'n'],
    'check': ['✓', ''],
    'checkx': ['✓', 'x'],
}

CURRENT_BOOL = 'rose'

def get_bool(name=None):
    '''Get the bool icon.'''
    return BOOLS[name or CURRENT_BOOL]



def astable(data, cols=None, drop=None, no_data_text='no data', **kw):
    '''Format a list of dictionaries as '''
    # short-circuit for non-lists
    if not isinstance(data, (list, tuple)):
        return data
    elif not data:
        return f'-- {no_data_text} --'

    import tabulate

    # get all columns across the data
    all_cols = {c for d in data for c in d} - set(drop or ())
    # default auto columns
    cols = cols or sorted(all_cols)
    # break out columns into a uniform list
    cols = list(_splitnested(cols, ',/|', all_cols))

    # handle leftover columns
    given_cols = {c for ci in cols for cj in ci for c in cj}
    cols = [
        ci for ci_ in cols for ci in (
            [[[c]] for c in sorted(all_cols - given_cols)]
            if ci_ == [['...']] else (ci_,))]

    # convert back to column names
    colnames = ['/'.join('|'.join(cj) for cj in ci) for ci in cols]

    # get data
    rows = [[[[nested_key(d, c, None) for c in cj] for cj in ci] for ci in cols] for d in data]

    # convert to table
    return tabulate.tabulate([[
            '\n'.join([
                '|'.join(str(_cellformat(c, **kw)) for c in subcell)
                for subcell in subrow
            ]) for subrow in row
        ] for row in rows], headers=colnames)


# basic helpers


def indent(d, indent=0, width=2):
    '''Indent a multi-line string.'''
    return '\n'.join('{}{}'.format(' '*indent*width, l) for l in d.splitlines())
indent_ = indent

def nested_key(d, k, default=...):
    '''Get a nested key (a.b.c) from a nested dictionary.'''
    for ki in k.split('.'):
        try:
            d = d[ki]
        except (TypeError, KeyError):
            if default is ...:
                raise
            return default
    return d


def _maybesplit(x, ch, strip=True, filter=True):
    '''Coerce a string to a list by splitting by a certain character,
    or skip if already a list.'''
    return [
        x.strip() if strip and isinstance(x, str) else x
        for x in (x.split(ch) if isinstance(x, str) else x)
        if not filter or x]


def _splitnested(cols, seps=',/|', avail=None):
    '''Splits a shorthand column layout into a nested column list.
    e.g.
        'time,max_laeq|avg_laeq/l90|min_laeq,emb_*,...'
        [
            [['time]],
            [
                ['max_laeq', 'avg_laeq'],
                ['l90', 'min_laeq']
            ],
            [['emb_min']], [['emb_max'], ...],
            [['time']], ...  # leftover columns
        ]
    '''
    if not seps:
        yield cols
        return
    sep, nextsep = seps[0], seps[1] if len(seps) > 1 else None
    for x in _maybesplit(cols, sep):
        xs = [x]
        if isinstance(x, str) and not any(s in x for s in seps) and avail and '*' in x:
            xs = sorted(c for c in avail if fnmatch.fnmatch(c, x))

        for xi in xs:  # inner loop handles unpacked glob
            yield list(_splitnested(xi, seps[1:], avail)) if nextsep else xi


def _cellformat(x, bool_icon=None):
    '''Format a cell's value based on its data type.'''
    if isinstance(x, bool):
        BOOL = get_bool(bool_icon)
        return BOOL[0] if x else BOOL[1]
    if isinstance(x, float):
        return '{:,.3f}'.format(x)
    if isinstance(x, list):
        return yamltable(x)
        # return 'list[{}, {}]'.format(len(x), type(x[0]).__name__ if x else None)
    if isinstance(x, dict):
        return yamltable(x)
        # return 'dict{{{}}}'.format(len(x))
    if x is None:
        return '--'
    return str(x)
