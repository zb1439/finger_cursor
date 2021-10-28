#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import logging
import re
from ast import literal_eval
from colorama import Back, Fore, Style
from easydict import EasyDict


def highlight(keyword, target, color=Fore.BLACK + Back.YELLOW):
    """
    use given color to highlight keyword in target string
    Args:
        keyword(str): highlight string
        target(str): target string
        color(str): string represent the color, use black foreground
        and yellow background as default
    Returns:
        (str) target string with keyword highlighted
    """
    return re.sub(keyword, color + r"\g<0>" + Style.RESET_ALL, target)


def find_key(param_dict: dict, key: str) -> dict:
    """
    find key in dict
    Args:
        param_dict(dict):
        key(str):
    Returns:
        (dict)
    Examples::
        >>> d = dict(abc=2, ab=4, c=4)
        >>> find_key(d, "ab")
        {'abc': 2, 'ab':4}
    """
    find_result = {}
    for k, v in param_dict.items():
        if re.search(key, k):
            find_result[k] = v
        if isinstance(v, dict):
            res = find_key(v, key)
            if res:
                find_result[k] = res
    return find_result


def diff_dict(src, dst):
    """
    find difference between src dict and dst dict
    Args:
        src(dict): src dict
        dst(dict): dst dict
    Returns:
        (dict) dict contains all the difference key
    """
    diff_result = {}
    for k, v in src.items():
        if k not in dst:
            diff_result[k] = v
        elif dst[k] != v:
            if isinstance(v, dict):
                diff_result[k] = diff_dict(v, dst[k])
            else:
                diff_result[k] = v
    return diff_result


def _assert_with_logging(cond, msg):
    logger = logging.getLogger(__name__)

    if not cond:
        logger.error(msg)
    assert cond, msg


def _decode_cfg_value(value):
    """
    Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    If the value is a dict, it will be interpreted as a new config dict.
    If the value is a str, it will be evaluated as literals.
    Otherwise it is returned as-is.
    Args:
        value (dict or str): value to be decoded
    """
    if isinstance(value, str):
        # Try to interpret `value` as a string, number, tuple, list, dict, boolean, or None
        try:
            value = literal_eval(value)
        # The following two excepts allow v to pass through when it represents a
        # string.
        #
        # Longer explanation:
        # The type of v is always a string (before calling literal_eval), but
        # sometimes it *represents* a string and other times a data structure, like
        # a list. In the case that v represents a string, what we got back from the
        # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
        # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
        # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
        # will raise a SyntaxError.
        except (ValueError, SyntaxError):
            pass

    if isinstance(value, dict):
        return EasyDict(value)
    else:
        return value


def _cast_cfg_value_type(replacement, original, full_key):
    """
    Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    logger = logging.getLogger(__name__)
    ori_type = type(original)
    new_type = type(replacement)

    if original is None or replacement is None:
        logger.info("None type, {} to {}".format(ori_type, new_type))
        return replacement

    # The types must match (with some exceptions)
    if new_type == ori_type:
        logger.info(
            "change value of '{}' from {} to {}".format(full_key, original, replacement)
        )
        return replacement

    # try to casts replacement to original type
    try:
        replacement = ori_type(replacement)
        return replacement
    except Exception:
        logger.error(
            "Could not cast '{}' from {} to {} with values ({} vs. {})".format(
                full_key, new_type, ori_type, replacement, original)
        )
        raise ValueError
