# The following code in this file was obtained from different places from
# the standoff2conll project,
# obtained [here](https://github.com/spyysalo/standoff2conll) under the license
# that follows:
#
# The MIT License (MIT)
#
# Copyright (c) 2016 Sampo Pyysalo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Contains useful methods.
"""

import re
import sys

DEBUG_SS_POSTPROCESSING = False

__initial = []

# TODO: some cases that heuristics could be improved on
# - no split inside matched quotes
# - "quoted." New sentence
# - 1 mg .\nkg(-1) .

# breaks sometimes missing after "?", "safe" cases
__initial.append((re.compile(r'\b([a-z]+\?) ([A-Z][a-z]+)\b'), r'\1\n\2'))
# breaks sometimes missing after "." separated with extra space, "safe" cases
__initial.append((re.compile(r'\b([a-z]+ \.) ([A-Z][a-z]+)\b'), r'\1\n\2'))

# join breaks creating lines that only contain sentence-ending punctuation
__initial.append((re.compile(r'\n([.!?]+)\n'), r' \1\n'))

# no breaks inside parens/brackets. (To protect against cases where a
# pair of locally mismatched parentheses in different parts of a large
# document happen to match, limit size of intervening context. As this
# is not an issue in cases where there are no interveining brackets,
# allow an unlimited length match in those cases.)

__repeated = []

# unlimited length for no intevening parens/brackets
__repeated.append((re.compile(r'(\([^\[\]\(\)]*)\n([^\[\]\(\)]*\))'), r'\1 \2'))
__repeated.append((re.compile(r'(\[[^\[\]\(\)]*)\n([^\[\]\(\)]*\])'), r'\1 \2'))
# standard mismatched with possible intervening
__repeated.append(
    (re.compile(r'(\([^\(\)]{0,250})\n([^\(\)]{0,250}\))'), r'\1 \2'))
__repeated.append(
    (re.compile(r'(\[[^\[\]]{0,250})\n([^\[\]]{0,250}\])'), r'\1 \2'))
# nesting to depth one
__repeated.append((re.compile(
    r'(\((?:[^\(\)]|\([^\(\)]*\)){0,250})\n((?:[^\(\)]|\([^\(\)]*\)){0,'
    r'250}\))'),
                   r'\1 \2'))
__repeated.append((re.compile(
    r'(\[(?:[^\[\]]|\[[^\[\]]*\]){0,250})\n((?:[^\[\]]|\[[^\[\]]*\]){0,'
    r'250}\])'),
                   r'\1 \2'))

__final = []

# no break after periods followed by a non-uppercase "normal word"
# (i.e. token with only lowercase alpha and dashes, with a minimum
# length of initial lowercase alpha).
__final.append((re.compile(r'\.\n([a-z]{3}[a-z-]{0,}[ \.\:\,\;])'), r'. \1'))

# no break in likely species names with abbreviated genus (e.g.
# "S. cerevisiae"). Differs from above in being more liberal about
# separation from following text.
__final.append((re.compile(r'\b([A-Z]\.)\n([a-z]{3,})\b'), r'\1 \2'))

# no break in likely person names with abbreviated middle name
# (e.g. "Anton P. Chekhov", "A. P. Chekhov"). Note: Won't do
# "A. Chekhov" as it yields too many false positives.
__final.append((re.compile(
    r'\b((?:[A-Z]\.|[A-Z][a-z]{3,}) [A-Z]\.)\n([A-Z][a-z]{3,})\b'), r'\1 \2'))

# no break before CC ..
__final.append((re.compile(r'\n((?:and|or|but|nor|yet) )'), r' \1'))

# or IN. (this is nothing like a "complete" list...)
__final.append((re.compile(
    r'\n((?:of|in|by|as|on|at|to|via|for|with|that|than|from|into|upon|after'
    r'|while|during|within|through|between|whereas|whether) )'),
                r' \1'))

# no sentence breaks in the middle of specific abbreviations
__final.append((re.compile(r'\b(e\.)\n(g\.)'), r'\1 \2'))
__final.append((re.compile(r'\b(i\.)\n(e\.)'), r'\1 \2'))
__final.append((re.compile(r'\b(i\.)\n(v\.)'), r'\1 \2'))

# no sentence break after specific abbreviations
__final.append((re.compile(
    r'\b(e\. ?g\.|i\. ?e\.|i\. ?v\.|vs\.|cf\.|Dr\.|Mr\.|Ms\.|Mrs\.)\n'),
                r'\1 '))

# or others taking a number after the abbrev
__final.append((re.compile(r'\b([Aa]pprox\.|[Nn]o\.)\n(\d+)'), r'\1 \2'))

from re import compile as re_compile
from re import DOTALL, VERBOSE

SENTENCE_END_REGEX = re_compile(r'''
        # Require a leading non-whitespace character for the sentence
        \S
        # Then, anything goes, but don't be greedy
        .*?
        # Anchor the sentence at...
        (:?
            # One (or multiple) terminal character(s)
            #   followed by one (or multiple) whitespace
            (:?(\.|!|\?|。|！|？)+(?=\s+))
        | # Or...
            # Newlines, to respect file formatting
            (:?(?=\n+))
        | # Or...
            # End-of-file, excluding whitespaces before it
            (:?(?=\s*$))
        )
    ''', DOTALL | VERBOSE)


def refine_split(s):
    """
    Given a string with sentence splits as newlines, attempts to
    heuristically improve the splitting. Heuristics tuned for geniass
    sentence splitting errors.
    """

    if DEBUG_SS_POSTPROCESSING:
        orig = s

    for r, t in __initial:
        s = r.sub(t, s)

    for r, t in __repeated:
        while True:
            n = r.sub(t, s)
            if n == s:
                break
            s = n

    for r, t in __final:
        s = r.sub(t, s)

    # Only do final comparison in debug mode.
    if DEBUG_SS_POSTPROCESSING:
        # revised must match original when differences in space<->newline
        # substitutions are ignored
        r1 = orig.replace('\n', ' ')
        r2 = s.replace('\n', ' ')
        if r1 != r2:
            print(
                "refine_split(): error: text mismatch (returning "
                "original):\nORIG: '%s'\nNEW:  '%s'" % (
                    orig, s), file=sys.stderr)
            s = orig

    return s


def _refine_split(offsets, original_text):
    # Postprocessor expects newlines, so add. Also, replace
    # sentence-internal newlines with spaces not to confuse it.
    new_text = '\n'.join((original_text[o[0]:o[1]].replace('\n', ' ')
                          for o in offsets))

    output = refine_split(new_text)

    # Align the texts and see where our offsets don't match
    old_offsets = offsets[::-1]
    # Protect against edge case of single-line docs missing
    #   sentence-terminal newline
    if len(old_offsets) == 0:
        old_offsets.append((0, len(original_text),))
    new_offsets = []
    for refined_sentence in output.split('\n'):
        new_offset = old_offsets.pop()
        # Merge the offsets if we have received a corrected split
        while new_offset[1] - new_offset[0] < len(refined_sentence) - 1:
            _, next_end = old_offsets.pop()
            new_offset = (new_offset[0], next_end)
        new_offsets.append(new_offset)

    # Protect against missing document-final newline causing the last
    #   sentence to fall out of offset scope
    if len(new_offsets) != 0 and new_offsets[-1][1] != len(original_text) - 1:
        start = new_offsets[-1][1] + 1
        while start < len(original_text) and original_text[start].isspace():
            start += 1
        if start < len(original_text) - 1:
            new_offsets.append((start, len(original_text) - 1))

    # Finally, inject new-lines from the original document as to respect the
    #   original formatting where it is made explicit.
    last_newline = -1
    while True:
        try:
            orig_newline = original_text.index('\n', last_newline + 1)
        except ValueError:
            # No more newlines
            break

        for o_start, o_end in new_offsets:
            if o_start <= orig_newline < o_end:
                # We need to split the existing offsets in two
                new_offsets.remove((o_start, o_end))
                new_offsets.extend(((o_start, orig_newline,),
                                    (orig_newline + 1, o_end),))
                break
            elif o_end == orig_newline:
                # We have already respected this newline
                break
        else:
            # Stand-alone "null" sentence, just insert it
            new_offsets.append((orig_newline, orig_newline,))

        last_newline = orig_newline

    new_offsets.sort()
    return new_offsets


def _sentence_boundary_gen(text, regex):
    for match in regex.finditer(text):
        yield match.span()


def en_sentence_boundary_gen(text):
    for o in _refine_split([_o for _o in _sentence_boundary_gen(
            text, SENTENCE_END_REGEX)], text):
        yield o


def _text_by_offsets_gen(text, offsets):
    for start, end in offsets:
        yield text[start:end]


def split_sentences(text):
    offsets = [o for o in en_sentence_boundary_gen(text)]

    # adjust to include any initial space skipped by the
    # boundary generator. (TODO: fix generator instead.)
    if offsets and offsets[0][0] > 0:
        offsets.insert(0, (0, offsets[0][0]))

    # adjust to include any intervening space
    adjusted = []
    for i in range(len(offsets) - 1):
        adjusted.append((offsets[i][0], offsets[i + 1][0]))
    if offsets:
        adjusted.append((offsets[-1][0], len(text)))
    offsets = adjusted

    return [s for s in _text_by_offsets_gen(text, offsets)]
