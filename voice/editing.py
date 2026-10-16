"""Track changes and provide overlap-aware seam hints for rolling edits."""
from difflib import SequenceMatcher

from .textutil import content_chars
from .alignment import normalized
from .textutil import join_texts


def changes(original, corrected):
    return [{'start': i, 'end': j, 'original': original[i:j], 'replacement': corrected[a:b],
             'result_start': a, 'result_end': b}
            for tag, i, j, a, b in SequenceMatcher(None, original, corrected, autojunk=False).get_opcodes()
            if tag != 'equal']


def seam_hints(segments):
    """Find repeated later ASR phrases that resemble text spanning a segment join."""
    if len(segments) < 2:
        return []
    added = [row['added'] for row in segments]
    combined = join_texts(added)
    boundaries = [len(join_texts(added[:count])) for count in range(1, len(added))]
    compact = normalized(combined)
    candidates = []
    first, second = len(segments) - 2, len(segments) - 1
    left, right = segments[first]['raw'], segments[second]['raw']
    for start, _, size in SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks():
        block = left[start:start + size]
        phrases = [block[:length] for length in range(3, min(size, 8) + 1)]
        if size > 8:
            phrases += [block[-length:] for length in range(3, 9)]
        for phrase in phrases:
            key = normalized(phrase)
            if not 3 <= len(key) <= 8 or key in compact:
                continue
            best = (0.0, '')
            for boundary in boundaries:
                for begin in range(max(0, boundary - len(phrase) - 3), boundary):
                    for end in range(boundary + 1, min(len(combined), boundary + len(phrase) + 4) + 1):
                        old = combined[begin:end]
                        old_key = normalized(old)
                        if (not old_key or old_key[0] != key[0] or
                                len(content_chars(combined[begin:boundary])) < 2 or
                                abs(len(old_key) - len(key)) > 3):
                            continue
                        score = SequenceMatcher(None, key, old_key, autojunk=False).ratio()
                        if score > best[0]:
                            best = (score, old)
            if best[0] >= 0.55:
                candidates.append((best[0], {'old_asr': best[1], 'later_asr': phrase,
                                              'supporting_segments': [segments[first]['index'],
                                                                      segments[second]['index']]}))
    candidates.sort(key=lambda item: item[0], reverse=True)
    seen, hints = set(), []
    for _, hint in candidates:
        key = normalized(hint['later_asr'])
        if key not in seen:
            seen.add(key)
            hints.append(hint)
        if len(hints) == 3:
            break
    return hints


def complete_supported_seam(segments, proposed, hint):
    """Complete a model's partial seam fix using a phrase repeated by later ASR.

    Only use this conservative path when current and ASR segments have identical
    character boundaries; otherwise keep the model proposal unchanged.
    """
    added = [row['added'] for row in segments]
    current = [row['current'] for row in segments]
    if any(len(a) != len(b) for a, b in zip(added, current)):
        return None
    source = ''.join(added)
    shown = ''.join(current)
    if source != join_texts(added) or shown != join_texts(current):
        return None
    old, new = hint['old_asr'], hint['later_asr']
    start = source.find(old)
    if start < 0 or source.find(old, start + 1) >= 0:
        return None
    end = start + len(old)
    if SequenceMatcher(None, normalized(old), normalized(shown[start:end]), autojunk=False).ratio() < .6:
        return None
    offsets = [0]
    for part in current:
        offsets.append(offsets[-1] + len(part))
    left = next((i for i in range(len(current)) if offsets[i] <= start < offsets[i + 1]), None)
    right = next((i for i in range(len(current)) if offsets[i] < end <= offsets[i + 1]), None)
    if left is None or right is None or right != left + 1:
        return None

    def after_unique_anchor(prefix, proposal):
        if not prefix:
            return 0
        anchor = prefix[-min(3, len(prefix)):]
        start = proposal.find(anchor)
        if start < 0 or proposal.find(anchor, start + 1) >= 0:
            return None
        return start + len(anchor)

    def before_unique_anchor(suffix, proposal):
        if not suffix:
            return len(proposal)
        anchor = suffix[:min(3, len(suffix))]
        start = proposal.find(anchor)
        if start < 0 or proposal.find(anchor, start + 1) >= 0:
            return None
        return start

    prefix = current[left][:start - offsets[left]]
    suffix = current[right][end - offsets[right]:]
    proposed_start = after_unique_anchor(prefix, proposed[left])
    proposed_end = before_unique_anchor(suffix, proposed[right])
    if proposed_start is None or proposed_end is None:
        return None
    bridge = proposed[left][proposed_start:] + proposed[right][:proposed_end]
    key = normalized(new)
    if (len(key) < 3 or key[:3] not in normalized(bridge) or
            len(content_chars(bridge)) > len(content_chars(old)) + 4):
        return None  # The LLM has not supported a small, locatable seam edit.

    result = proposed.copy()
    left_size = offsets[left + 1] - start
    result[left] = proposed[left][:proposed_start] + new[:left_size]
    result[right] = new[left_size:] + proposed[right][proposed_end:]
    return result
