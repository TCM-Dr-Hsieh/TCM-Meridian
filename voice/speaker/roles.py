"""Role layer: which voice group is the doctor. An LLM reads the dialogue labelled by voice group; a voter makes its answers stable."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..config import PROMPTS_DIR
from ..llm import ValidationError
from ..textutil import extract_json
from . import BACKGROUND_GID, DOCTOR, OTHER
from .render import segment_runs

PROMPT_FILE = 'prompt_speaker_roles.txt'
ANSWER_NAMES = {'醫師': DOCTOR, '患者或家屬': OTHER, '患者家屬': OTHER, '不明': None}       # the old spelling is still understood
LOCK_AFTER = 3             # answers in a row that agree before the model is asked no more (until the groups change)
MAX_LINES = 80             # the most recent segments shown to the model
RETRY_NOTE = '\n上一次輸出無效，請只輸出符合格式的 JSON，roles 要列出每一個群。'


def load_prompt() -> str:
    return (PROMPTS_DIR / PROMPT_FILE).read_text(encoding='utf-8')


def group_label(gid: int) -> str:
    return f'S{gid + 1}'


def dialogue_lines(segments, tracker, limit: int = MAX_LINES) -> list[str]:
    """`#12 [S1] 哪裡不舒服 [S2] 我胃痛` for the latest segments that have timed text."""
    lines = []
    for segment in reversed(segments):
        if len(lines) >= limit:
            break
        if segment.kind != 'speech' or not segment.visible:
            continue
        pieces = tracker.voice_groups_for(segment.index)
        if not pieces:
            continue
        labels = tuple((c0, c1, 'bg' if gid == BACKGROUND_GID else 'S?' if gid is None else group_label(gid), '') for c0, c1, gid in pieces)
        runs = [r for r in segment_runs(segment.added, segment.corrected, labels) if r.label != 'bg']       # background voices are not part of the dialogue
        if runs:
            lines.append(f'#{segment.index} ' + ' '.join(f'[{r.label}] {r.text}' for r in runs))
    return lines[::-1]


def request_payload(groups: list[int], lines: list[str]) -> str:
    return json.dumps({'groups': [group_label(g) for g in groups], 'transcript_by_voice_cluster': lines}, ensure_ascii=False)


def parse_answer(text: str, groups: list[int]) -> dict[int, str | None]:
    """{group: DOCTOR | OTHER | None (the model could not tell)}; raises ValidationError when the reply is unusable."""
    try:
        roles = extract_json(text)['roles']
        answer = {}
        for gid in groups:
            value = roles[group_label(gid)]
            if value not in ANSWER_NAMES:
                raise ValueError(value)
            answer[gid] = ANSWER_NAMES[value]
        return answer
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValidationError('角色判定回應不是 {"roles":{"S1":…}} 的 JSON，或缺少某個群。') from exc


@dataclass
class RoleVoter:
    """Makes a stable map from repeated answers: the first is adopted, later ones confirm it, and it is replaced only after two
    different answers in a row. Three agreeing answers lock it until the voice groups change, provided every group has a
    confirmed role. A map set by hand always wins and the model is not asked while it stands.

    `trusted` is what the prompts for the record writer and reviewer wait for: a usable map confirmed by a second agreeing answer
    in a row, or set by the physician. One answer alone could name the wrong group and turn every mark around; once trusted it
    stays so. The confirmation is per group (`confirmed_roles`): a group that appears later, or whose role was just replaced, has
    one answer behind it and counts as unknown (screen, prompts and files) until a second answer in a row agrees. A confirmed role
    stays confirmed through a single dissenting answer; only the streak towards a first confirmation is broken by it."""
    current: dict[int, str] = field(default_factory=dict)
    confirmed: int = 0
    candidate: dict[int, str] | None = None
    manual: bool = False
    trusted: bool = False
    agree: dict[int, int] = field(default_factory=dict)      # per group: answers in a row that agree with its role in `current`
    sure: set[int] = field(default_factory=set)              # the groups whose role in `current` is confirmed
    groups: int = 0                                          # how many voice groups exist (a group nobody named is still waiting)
    lock_after: int = LOCK_AFTER                             # agreeing answers in a row after which the model is asked no more

    @property
    def confirmed_roles(self) -> dict[int, str]:
        """The roles that may be used: all of them when the physician set the map, otherwise the groups with a confirmed role."""
        if self.manual:
            return dict(self.current)
        return {g: r for g, r in self.current.items() if g in self.sure}

    @property
    def pending_groups(self) -> list[int]:
        """Voice groups without a confirmed role: not named yet (the model said 不明, or it was not asked yet) or named by one answer."""
        confirmed = self.confirmed_roles
        return [g for g in range(self.groups) if g not in confirmed]

    @property
    def usable(self) -> bool:
        """Both a doctor group and another group are named: only then can sentences be labelled."""
        roles = set(self.current.values())
        return DOCTOR in roles and OTHER in roles

    @property
    def locked(self) -> bool:
        return self.confirmed >= self.lock_after and self.usable and not self.pending_groups

    @property
    def wants_answer(self) -> bool:
        return not self.manual and not self.locked

    def groups_changed(self, count: int | None = None):
        """A group was added: the old agreement says nothing about the new one."""
        if count is not None:
            self.groups = max(self.groups, count)
        self.confirmed = 0
        self.candidate = None

    def set_manual(self, roles: dict[int, str]):
        self.current = dict(roles)
        self.agree = {g: 2 for g in roles}
        self.sure = set(roles)
        self.manual = True
        self.trusted = True
        self.confirmed = 0
        self.candidate = None

    def _trust(self):
        roles = set(self.confirmed_roles.values())
        if DOCTOR in roles and OTHER in roles:
            self.trusted = True

    def _count(self, known: dict[int, str], previous: dict[int, str]):
        for g, r in known.items():
            if previous.get(g) == r:
                self.agree[g] = self.agree.get(g, 0) + 1
            else:
                self.agree[g] = 1
                self.sure.discard(g)

    def _settle(self):
        self.sure |= {g for g, n in self.agree.items() if n >= 2 and g in self.current}
        self._trust()

    def release(self):
        """Hand control back to the model (the map in force stays until two answers in a row disagree with it)."""
        self.manual = False
        self.confirmed = 0
        self.candidate = None

    def feed(self, answer: dict[int, str | None]) -> bool:
        """Take one model answer. True when the map in force changed or a role became confirmed (both must be saved and shown)."""
        before = (dict(self.current), dict(self.confirmed_roles))
        self._feed(answer)
        return before != (dict(self.current), dict(self.confirmed_roles))

    def _feed(self, answer: dict[int, str | None]):
        self.groups = max(self.groups, len(answer))
        known = {g: r for g, r in answer.items() if r is not None}
        if not known or self.manual:
            return
        previous = dict(self.current)
        if not self.usable:                           # nothing named yet, or no doctor / no other group: take the answer outright
            self.current, self.candidate = {**self.current, **known}, None
            self._count(known, previous)
            self.confirmed = 1 if self.usable else 0
            self._settle()
            return
        clash = [g for g, r in known.items() if g in self.current and self.current[g] != r]
        if not clash:
            self.current.update({g: r for g, r in known.items() if g not in self.current})
            self._count(known, previous)
            self.confirmed += 1
            self.candidate = None
            self._settle()
            return
        for g in clash:                               # a dissenting answer breaks the streak of the groups it disagrees about
            self.agree[g] = 0
        if self.candidate == known:
            self.current = {**self.current, **known}
            self._count(known, previous)
            self.confirmed, self.candidate = 1, None
            self._settle()
            return
        self.candidate = known
