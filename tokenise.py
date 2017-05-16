#### PATTERN | TEXT | TOKENIZER #######################################################################
# -*- coding: utf-8 -*-
# Copyright (c) 2010 University of Antwerp, Belgium
# Author: Tom De Smedt <tom@organisms.be>
# License: BSD (see LICENSE.txt for details).
# http://www.clips.ua.ac.be/pages/pattern

####################################################################################################

"""Copyright (c) 2011-2013 University of Antwerp, Belgium
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
  * Neither the name of Pattern nor the names of its
    contributors may be used to endorse or promote products
    derived from this software without specific prior written
    permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE."""

import re

TOKEN = re.compile(r"(\S+)\s")

# Common accent letters.
DIACRITICS = \
diacritics = u"àáâãäåąāæçćčςďèéêëēěęģìíîïīłįķļľņñňńйðòóôõöøþřšťùúûüůųýÿўžż"

# Common punctuation marks.
PUNCTUATION = \
punctuation = ".,;:!?()[]{}`''\"@#$^&*+-|=~_"

# Common abbreviations.
ABBREVIATIONS = \
abbreviations = set((
    "a.", "adj.", "adv.", "al.", "a.m.", "art.", "c.", "capt.", "cert.", "cf.", "col.", "Col.",
    "comp.", "conf.", "def.", "Dep.", "Dept.", "Dr.", "dr.", "ed.", "e.g.", "esp.", "etc.", "ex.",
    "f.", "fig.", "gen.", "id.", "i.e.", "int.", "l.", "m.", "Med.", "Mil.", "Mr.", "n.", "n.q.",
    "orig.", "pl.", "pred.", "pres.", "p.m.", "ref.", "v.", "vs.", "w/"
))

RE_ABBR1 = re.compile(r"^[A-Za-z]\.$")     # single letter, "T. De Smedt"
RE_ABBR2 = re.compile(r"^([A-Za-z]\.)+$")  # alternating letters, "U.S."
RE_ABBR3 = re.compile(r"^[A-Z][%s]+.$" % ( # capital followed by consonants, "Mr."
        "|".join("bcdfghjklmnpqrstvwxz")))

# Common contractions.
replacements = {
     "'d": " 'd",
     "'m": " 'm",
     "'s": " 's",
    "'ll": " 'll",
    "'re": " 're",
    "'ve": " 've",
    "n't": " n't"
}

# Common emoticons.
EMOTICONS = \
emoticons = { # (facial expression, sentiment)-keys
    ("love" , +1.00): set(("<3", u"♥", u"❤")),
    ("grin" , +1.00): set((">:D", ":-D", ":D", "=-D", "=D", "X-D", "x-D", "XD", "xD", "8-D")),
    ("taunt", +0.75): set((">:P", ":-P", ":P", ":-p", ":p", ":-b", ":b", ":c)", ":o)", ":^)")),
    ("smile", +0.50): set((">:)", ":-)", ":)", "=)", "=]", ":]", ":}", ":>", ":3", "8)", "8-)")),
    ("wink" , +0.25): set((">;]", ";-)", ";)", ";-]", ";]", ";D", ";^)", "*-)", "*)")),
    ("blank", +0.00): set((":-|", ":|")),
    ("gasp" , -0.05): set((">:o", ":-O", ":O", ":o", ":-o", "o_O", "o.O", u"°O°", u"°o°")),
    ("worry", -0.25): set((">:/",  ":-/", ":/", ":\\", ">:\\", ":-.", ":-s", ":s", ":S", ":-S", ">.>")),
    ("frown", -0.75): set((">:[", ":-(", ":(", "=(", ":-[", ":[", ":{", ":-<", ":c", ":-c", "=/")),
    ("cry"  , -1.00): set((":'(", ":'''(", ";'("))
}

RE_EMOTICONS = [r" ?".join(map(re.escape, e)) for v in EMOTICONS.values() for e in v]
RE_EMOTICONS = re.compile(r"(%s)($|\s)" % "|".join(RE_EMOTICONS))

# Common emoji.
EMOJI = \
emoji = { # (facial expression, sentiment)-keys
    ("love" , +1.00): set((u"❤️", u"💜", u"💚", u"💙", u"💛", u"💕")),
    ("grin" , +1.00): set((u"😀", u"😄", u"😃", u"😆", u"😅", u"😂", u"😁", u"😻", u"😍", u"😈", u"👌")),
    ("taunt", +0.75): set((u"😛", u"😝", u"😜", u"😋", u"😇")),
    ("smile", +0.50): set((u"😊", u"😌", u"😏", u"😎", u"☺", u"👍")),
    ("wink" , +0.25): set((u"😉")),
    ("blank", +0.00): set((u"😐", u"😶")),
    ("gasp" , -0.05): set((u"😳", u"😮", u"😯", u"😧", u"😦", u"🙀")),
    ("worry", -0.25): set((u"😕", u"😬")),
    ("frown", -0.75): set((u"😟", u"😒", u"😔", u"😞", u"😠", u"😩", u"😫", u"😡", u"👿")),
    ("cry"  , -1.00): set((u"😢", u"😥", u"😓", u"😪", u"😭", u"😿")),
}

RE_EMOJI = [e for v in EMOJI.values() for e in v]
RE_EMOJI = re.compile(r"(\s?)(%s)(\s?)" % "|".join(RE_EMOJI))

# Mention marker: "@tomdesmedt".
RE_MENTION = re.compile(r"\@([0-9a-zA-z_]+)(\s|\,|\:|\.|\!|\?|$)")

# Sarcasm marker: "(!)".
RE_SARCASM = re.compile(r"\( ?\! ?\)")

# Paragraph line breaks
# (\n\n marks end of sentence).
EOS = "END-OF-SENTENCE"


def tokenize(string, punctuation=PUNCTUATION, abbreviations=ABBREVIATIONS,
                replace=replacements, linebreak=r"\n{2,}"):
    """ Returns a list of sentences. Each sentence is a space-separated string of tokens (words).
        Handles common cases of abbreviations (e.g., etc., ...).
        Punctuation marks are split from other words. Periods (or ?!) mark the end of a sentence.
        Headings without an ending period are inferred by line breaks.
    """
    # Handle punctuation.
    punctuation = tuple(punctuation)
    # Handle replacements (contractions).
    for a, b in replace.items():
        string = re.sub(a, b, string)
    # Handle Unicode quotes.
    if isinstance(string, str):
        string = string.replace(u"“", u" “ ")
        string = string.replace(u"”", u" ” ")
        string = string.replace(u"‘", u" ‘ ")
        string = string.replace(u"’", u" ’ ")
    # Collapse whitespace.
    string = re.sub("\r\n", "\n", string)
    string = re.sub(linebreak, " %s " % EOS, string)
    string = re.sub(r"\s+", " ", string)
    tokens = []
    # Handle punctuation marks.
    for t in TOKEN.findall(string+" "):
        if len(t) > 0:
            tail = []
            if not RE_MENTION.match(t):
                while t.startswith(punctuation) and \
                  not t in replace:
                    # Split leading punctuation.
                    if t.startswith(punctuation):
                        tokens.append(t[0]); t=t[1:]
            if not False:
                while t.endswith(punctuation) and \
                  not t in replace:
                    # Split trailing punctuation.
                    if t.endswith(punctuation) and not t.endswith("."):
                        tail.append(t[-1]); t=t[:-1]
                    # Split ellipsis (...) before splitting period.
                    if t.endswith("..."):
                        tail.append("..."); t=t[:-3].rstrip(".")
                    # Split period (if not an abbreviation).
                    if t.endswith("."):
                        if t in abbreviations or \
                          RE_ABBR1.match(t) is not None or \
                          RE_ABBR2.match(t) is not None or \
                          RE_ABBR3.match(t) is not None:
                            break
                        else:
                            tail.append(t[-1]); t=t[:-1]
            if t != "":
                tokens.append(t)
            tokens.extend(reversed(tail))
    # Handle citations (periods + quotes).
    if isinstance(string, str):
        quotes = ("'", "\"", u"”", u"’")
    else:
        quotes = ("'", "\"")
    # Handle sentence breaks (periods, quotes, parenthesis).
    sentences, i, j = [[]], 0, 0
    while j < len(tokens):
        if tokens[j] in ("...", ".", "!", "?", EOS):
            while j < len(tokens) \
              and (tokens[j] in ("...", ".", "!", "?", EOS) or tokens[j] in quotes):
                if tokens[j] in quotes and sentences[-1].count(tokens[j]) % 2 == 0:
                    break # Balanced quotes.
                j += 1
            sentences[-1].extend(t for t in tokens[i:j] if t != EOS)
            sentences.append([])
            i = j
        j += 1
    # Handle emoticons.
    sentences[-1].extend(tokens[i:j])
    sentences = (" ".join(s) for s in sentences if len(s) > 0)
    sentences = (RE_SARCASM.sub("(!)", s) for s in sentences)
    sentences = [RE_EMOTICONS.sub(
        lambda m: m.group(1).replace(" ", "") + m.group(2), s) for s in sentences]
    sentences = [RE_EMOJI.sub(
        lambda m: (m.group(1) or " ") + m.group(2) + (m.group(3) or " "), s) for s in sentences]
    sentences = [s.replace("  ", " ").strip() for s in sentences]
    return sentences

