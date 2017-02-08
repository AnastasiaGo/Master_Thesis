#!/usr/bin/python3

import sys
from lxml import etree

levels = {'Very negative': -2,
        'Verynegative': -2,
        'Negative': -1,
        'Neutral': 0,
        'Positive': 1,
        'Very positive': 2,
        'Verypositive': 2,
}


print("file"
      "\tclass"
      "\tsentences"
      "\ttokens"
      "\ttok_vneg"
      "\ttok_neg"
      "\ttok_neut"
      "\ttok_pos"
      "\ttok_vpos"
      "\ttok_vneg_macro"
      "\ttok_neg_macro"
      "\ttok_neut_macro"
      "\ttok_pos_macro"
      "\ttok_vpos_macro"
      "\tsent_vneg"
      "\tsent_neg"
      "\tsent_neut"
      "\tsent_pos"
      "\tsent_vpos"
)
for f in sys.argv[1:]:
    r = etree.parse(f)
    count_tokens = {k:0 for k in set(levels.values())}
    count_sentence = {k:0 for k in set(levels.values())}
    macroavg_tokens = {k:0 for k in set(levels.values())}
    nsent = 0
    ntoks = 0
    for s in r.findall('//sentence'):
        ssent = s.get('sentiment')
        count_sentence[levels[ssent]] += 1
        nsent += 1
        tcounts = {k:0 for k in set(levels.values())}
        for t in s.findall('./tokens/token'):
            tsent = t.find('./sentiment')
            if tsent is None: # skip tokens with no sentiment information
                continue
            tcounts[levels[tsent.text]] += 1
            ntoks += 1

        for toksent in tcounts:
            count_tokens[toksent] += tcounts[toksent]
            if len(tcounts) != 0:
                macroavg_tokens[toksent] += tcounts[toksent] / len(tcounts)
    if f.startswith("non"):
        cls = "nondepression"
    else:
        cls = "depression"
    
    if ntoks != 0: # skip files with no sentiment data
        print("{}\t{}\t{}\t{}".format(f, cls, nsent, ntoks), end="")
        for sent in sorted(count_tokens):
            print("\t{}".format(count_tokens[sent]/ntoks), end="")
        for sent in sorted(macroavg_tokens):
            print("\t{}".format(macroavg_tokens[sent]/nsent), end="")
        for sent in sorted(count_sentence):
            print("\t{}".format(count_sentence[sent]/nsent), end="")
        print()
