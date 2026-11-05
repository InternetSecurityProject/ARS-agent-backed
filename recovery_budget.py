"""Bounded recovery for distinct menus, independent of ASR prefaces/formatting.

This identifies retry scopes only. It never chooses or authorizes a DTMF key.
"""
from dataclasses import dataclass, field
import re
import hashlib


def _norm(label):
    return re.sub(r'[^0-9A-Za-z가-힣]', '', label).lower()


def _similar(a, b):
    a, b = _norm(a), _norm(b)
    if a == b or (min(len(a),len(b)) >= 3 and (a in b or b in a)):
        return True
    aa = {a[i:i+2] for i in range(max(0,len(a)-1))}
    bb = {b[i:i+2] for i in range(max(0,len(b)-1))}
    return bool(aa and bb) and len(aa & bb) / len(aa | bb) >= .6


@dataclass
class MenuBudget:
    keys: frozenset
    labels: dict
    sources: set = field(default_factory=set)
    analyzed: bool = False
    replayed: bool = False

    def matches(self, keys, labels, source):
        if self.keys != keys:
            return False
        if source in self.sources:
            return True  # Correcting our analysis does not change the source menu.
        common = self.labels.keys() & labels.keys()
        # Confidence changes add evidence to a menu, not a fresh retry. Require
        # corroborated, different named services before splitting equal keys.
        changed = sum(not _similar(self.labels[k],labels[k]) for k in common)
        return not (common and changed >= min(2,len(common)))


class MenuRecoveryBudgets:
    def __init__(self):
        self._scopes = {}
        self._exhausted = MenuBudget(frozenset(),{},analyzed=True,replayed=True)

    def clear(self):
        self._scopes.clear()

    def for_prompt(self, scope, prompt, *, create=True):
        keys = frozenset(getattr(prompt,'_phrase_observed_keys',
                                 [o.key for o in prompt.all_options()]))
        if not keys:
            return None  # No actionable menu: never spend a following menu's retry.
        labels = dict(getattr(prompt,'_phrase_trusted_labels',
                              {o.key:o.category for o in prompt.options if o.category}))
        # Special DTMF characters carry binding information, unlike layout
        # punctuation. A star/hash swap must not collide with the old menu.
        source_text = re.sub(r'[^0-9A-Za-z가-힣*#]', '', prompt.text).lower()
        source = hashlib.sha256(source_text.encode()).hexdigest()
        menus = self._scopes.get(scope,[])
        for budget in menus:
            if budget.matches(keys,labels,source):
                if create:
                    if source in budget.sources:
                        # Same-source reanalysis may correct initially wrong
                        # bindings. Use that correction for later ASR variants.
                        budget.labels.update(labels)
                    else:
                        for key,label in labels.items():
                            budget.labels.setdefault(key,label)
                    if len(budget.sources)<16:
                        budget.sources.add(source)
                return budget
        if not create:
            return self._exhausted if len(menus)>=3 else None
        # ASR corruption may invent changed identities. A shared cap prevents
        # unlimited repair/replay without an actual non-replay key transition.
        if len(menus)>=3:
            return self._exhausted
        budget=MenuBudget(keys,labels,{source})
        self._scopes.setdefault(scope,[]).append(budget)
        return budget
