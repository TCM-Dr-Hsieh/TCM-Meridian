"""The tunable numbers of speaker marking (SPEC 4.3): one registry that feeds the settings pop-up, the validation and the visit.

Every knob has a default (the value the pipeline was built and measured with), a range, a plain-words description of what it does and what
raising or lowering it does. Only the values that differ from the default are stored in `config.json` (`speaker.advanced`), so a knob
that was never touched follows the default of the version in use. `unknown_percentile` is the one knob that lives in its own settings field
(it has been there since the first version); the pop-up shows it with the others.

This module imports nothing from the project (the settings module imports it); `tests/test_speaker_tuning.py` checks that every default here
is the default the consuming code really has and that every knob reaches its consumer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

STAGES = ('單位與句子', '聲音群的建立與更新', '句投票與「不明」', '角色判定', '背景人聲', '文字補標', '二階文字補標')


@dataclass(frozen=True)
class Knob:
    key: str
    label: str
    stage: str
    default: float
    low: float
    high: float
    step: float
    what: str                    # what it does
    higher: str                  # what raising it does
    lower: str                   # what lowering it does
    integer: bool = False
    star: bool = False           # a knob that matters most
    top_level: bool = False      # stored in its own settings field, not in `speaker.advanced`


KNOBS: tuple[Knob, ...] = (
    # ---- 1. units and sentences (units.py)
    Knob('target', '單位目標長度（秒）', STAGES[0], 0.8, 0.3, 3.0, 0.1,
         '一個單位（一個聲紋）累積到這個長度就結束。',
         '聲紋較穩，但一個單位更容易跨到兩個人。', '切得更細，但聲紋更雜、更不穩。'),
    Knob('pause', '單位斷點的停頓（秒）', STAGES[0], 0.3, 0.1, 1.5, 0.05,
         '停頓超過這麼久就結束一個單位。',
         '更少在停頓處切開，一個單位可能混進兩個人。', '切得更碎，碎片多半被併走或丟棄。'),
    Knob('min_unit', '最短單位（秒）', STAGES[0], 0.4, 0.1, 1.5, 0.05,
         '比這更短的碎片併進鄰居，併不了就丟棄（沒有聲紋）。',
         '更多字沒有聲紋，「不明」變多。', '很短的音訊也算聲紋，可靠度下降。'),
    Knob('max_merged', '合併後單位上限（秒）', STAGES[0], 1.3, 0.5, 4.0, 0.1,
         '碎片併進鄰居之後，單位長度的上限。',
         '允許更長的合併單位。', '更多碎片因為併不進去而被丟棄。'),
    Knob('sent_pause', '句子斷點的停頓（秒）', STAGES[0], 0.5, 0.2, 3.0, 0.1,
         '停頓超過這麼久就結束一句（一句一票、一個標籤）。',
         '句子變長，一句可能夾著兩個人。', '句子變短，每句的票數少，「不明」變多。'),
    # ---- 2. voice groups (tracker.py)
    Knob('refit_seconds', '聲音群重算週期（秒）', STAGES[1], 60.0, 20.0, 600.0, 10.0,
         '多久重算一次聲音群（第一次建群也要等到這個時間點）。',
         '反應更慢，新說話者更晚成群，開頭「全部不明」的時間更久。', '反應更快，但每次用的資料少、比較不穩。'),
    Knob('quiet_gate_db', '安靜閘門（dB）', STAGES[1], 10.0, 3.0, 60.0, 0.5,
         '比「所有單位音量的第 75 百分位」小這麼多的單位，不參與建立與更新群中心（仍會被分到最近的群）。'
         '在目前的資料上可用範圍很窄（約 10–11 dB），不建議動。',
         '閘門太鬆：背景人聲可能又自成一群，把醫師與聲音相近的患者併成一群（14 dB 以上失效）。',
         '把真正在場但小聲、離麥克風遠的人也排除（9 dB 以下會壞掉）；整場一直很小聲的第二個人會建不起群。', star=True),
    Knob('quiet_gate_percentile', '安靜閘門的音量基準（百分位）', STAGES[1], 75.0, 50.0, 95.0, 1.0,
         '上面那條安靜閘門，音量基準取所有單位的第幾百分位。',
         '基準變高，被排除的單位變多。', '基準變低，排除變少，閘門變鬆。'),
    Knob('first_fit_units', '第一次建群所需單位數', STAGES[1], 8, 4, 60, 1,
         '至少累積這麼多單位才第一次建群（也用來估計醫師的常態音量）。',
         '更晚才有聲音群。', '用太少的資料建群，不穩。', integer=True),
    Knob('min_group_share', '第二群的最小占比', STAGES[1], 0.15, 0.02, 0.5, 0.01,
         '兩群裡較小的那群，至少要占這麼多說話時間，否則視為只有一個聲音。',
         '說話少的人（例如少開口的患者）不被承認為第二個人。', '容易把同一個人切成兩半，誤當成兩個人。'),
    Knob('min_group_units', '第二群的最少單位數', STAGES[1], 8, 2, 60, 1,
         '較小的那群至少要有這麼多單位。',
         '同上，更嚴。', '同上，更鬆。', integer=True),
    Knob('first_split_max_cos', '首次分兩群：中心相似度上限', STAGES[1], 0.75, 0.3, 0.95, 0.01,
         '兩群中心的相似度要低於這個值，才算兩個人。',
         '聲音更像的兩人也能分開，但同一個人被切成兩半的風險升高。',
         '聲音相近的兩人被當成同一人，全部顯示不明（不會標錯）。', star=True),
    Knob('new_group_max_cos', '新增聲音群：與舊群的相似度上限', STAGES[1], 0.60, 0.2, 0.9, 0.01,
         '新群與每個既有群的相似度都要低於這個值。',
         '更容易新增群，也更容易誤增（例如把背景人聲當成新群）。', '更難新增群，聲音像的家屬會併進最近的群。', star=True),
    Knob('new_group_share', '新增聲音群：最近說話占比', STAGES[1], 0.10, 0.02, 0.5, 0.01,
         '新群要占最近一段時間說話量的這個比例才成立。',
         '偶爾插話的人不成群。', '只說一兩句也成群，容易誤增。'),
    Knob('new_group_window', '新增聲音群：「最近」多長（秒）', STAGES[1], 180.0, 30.0, 900.0, 10.0,
         '上面的占比看多長的最近時間（不看整場累計，晚到的家屬才不必等很久）。',
         '更接近整場累計，晚到的家屬要等很久。', '對新來的人更靈敏，也更容易受短暫干擾。'),
    Knob('max_groups', '聲音群數量上限', STAGES[1], 4, 2, 8, 1,
         '一場看診最多分成幾個聲音群。',
         '允許更多人，也更容易長出誤群。', '超過的說話者併進既有的群。', integer=True),
    Knob('far_cos', '「誰都不像」的相似度', STAGES[1], 0.25, 0.0, 0.6, 0.01,
         '與每個群的相似度都低於這個值的單位：不用來更新中心，也不參與投票。',
         '更多單位不投票，「不明」變多，群中心更乾淨。', '怪聲音的單位也會拉歪群中心、參與投票。'),
    # ---- 3. sentence vote (tracker.py, settings)
    Knob('clip', '單位票數上限（對數似然比）', STAGES[2], 6.0, 1.0, 12.0, 0.5,
         '每個單位的票被夾在 ±這個值，限制單一單位的影響。',
         '單一很強的單位容易主導整句。', '更接近平均，整句更保守。'),
    Knob('unknown_percentile', '「不明」百分位', STAGES[2], 15.0, 0.0, 50.0, 1.0,
         '句子的強度低於「過去各句的第 p 百分位」就標「不明」。',
         '「不明」更多、標錯更少。', '覆蓋率高、標錯多；0 幾乎沒有「不明」。', star=True, top_level=True),
    Knob('min_history', '判斷「弱」所需的過去句數', STAGES[2], 8, 3, 60, 1,
         '至少有這麼多句過去的句子，才用歷史算百分位。',
         '更晚才開始決定句子的標記。', '用太少的句子判斷「弱」，不穩。', integer=True),
    # ---- 4. roles (roles.py)
    Knob('lock_after', '角色對照鎖定：連續一致次數', STAGES[3], 3, 2, 10, 1,
         '連續這麼多次 LLM 答案一致後不再詢問（直到聲音群數量改變）。',
         '更久才鎖定，LLM 呼叫較多。', '更早鎖定，對中途新來的人反應較慢。', integer=True),
    Knob('max_lines', '角色判定看的段數', STAGES[3], 80, 10, 300, 10,
         '角色判定的 LLM 一次看最近幾段。',
         '上下文更多，token 更多。', '上下文少，可能判錯。', integer=True),
    # ---- 5. background voices (tracker.py)
    Knob('background_rel_db', '背景：比醫師小多少（dB，負值）', STAGES[4], -8.0, -30.0, 0.0, 0.5,
         '一句話的音量要比「醫師的常態音量」小這麼多（負值），才可能是背景。',
         '（往 0 靠近）更多句子被標成背景，誤刪風險上升。',
         '（更負）只抓非常小聲的，漏抓增加；在目前的相似度門檻下，−6 到 −8 影響很小。'),
    Knob('background_max_cos', '背景：與聲音群的相似度上限', STAGES[4], 0.425, 0.0, 1.0, 0.025,
         '一句話與所有聲音群的相似度（各單位最佳餘弦的平均）要低於這個值，才可能是背景。0 ＝ 關閉背景過濾。',
         '抓到更多背景，但聲音小的患者開始被誤刪（在目前的資料上 0.45 起）。',
         '更保守，漏抓的背景會以「不明」進逐字稿。', star=True),
    # ---- 6. text fill (tracker.py, fill.py)
    Knob('fill_wait', '文字補標：等待後文句數', STAGES[5], 10, 0, 60, 1,
         '不明句要等後面有這麼多句，才拿去問 LLM（後面的句子是它的上下文）。',
         '上下文更完整，但補標更晚。', '上下文少，補標較不準。', integer=True),
    Knob('fill_batch', '文字補標：累積幾句才送', STAGES[5], 6, 1, 60, 1,
         '累積這麼多待問的句子才送一次 LLM。',
         'LLM 呼叫更少，延遲更久。', '呼叫更頻繁。', integer=True),
    Knob('fill_max_age', '文字補標：最久等多久（秒）', STAGES[5], 120.0, 10.0, 900.0, 10.0,
         '最舊的待問句等了這麼久就送，不管累積了幾句。',
         '延遲更久，呼叫較少。', '更快補標，呼叫較多。'),
    Knob('fill_chunk', '文字補標：每次問幾句', STAGES[5], 25, 5, 60, 1,
         '每次呼叫最多問幾句。',
         '每次呼叫問更多句，token 更多。', '呼叫次數增加。', integer=True),
    Knob('fill_before', '文字補標：前文句數', STAGES[5], 40, 0, 120, 5,
         '每次呼叫給 LLM 看被問句之前的幾句當上下文。',
         '上下文與 token 更多。', '上下文更少，較不準。', integer=True),
    Knob('fill_after', '文字補標：後文句數', STAGES[5], 10, 0, 40, 1,
         '每次呼叫給 LLM 看被問句之後的幾句當上下文。',
         '上下文與 token 更多。', '上下文更少，較不準。', integer=True),
    # ---- 7. second text fill (tracker.py)
    Knob('fill2_batch', '二階補標：累積幾句才送', STAGES[6], 12, 1, 60, 1,
         '二階補標累積這麼多句才送一次（看診結束時的最後一輪會送出剩下的）。',
         '呼叫較少、更晚。', '更早補，呼叫較多。', integer=True),
    Knob('fill2_refits', '二階補標：與一階答案相隔幾次重算', STAGES[6], 1, 0, 10, 1,
         '一階答案回來後，至少隔這麼多次重算才問二階（等聲音群與鄰句的標記更新）。',
         '更晚，鄰句的標記更新更多。', '更早，但鄰句的標記可能還沒變。', integer=True),
)

BY_KEY = {knob.key: knob for knob in KNOBS}
assert len(BY_KEY) == len(KNOBS)
DEFAULTS = {knob.key: knob.default for knob in KNOBS}
STORED = tuple(knob for knob in KNOBS if not knob.top_level)           # the knobs kept in `speaker.advanced`


def effective(advanced: dict | None, unknown_percentile: float) -> dict:
    """Every knob's value for a visit: the default unless `advanced` says otherwise."""
    values = dict(DEFAULTS)
    values.update({k: v for k, v in (advanced or {}).items() if k in BY_KEY and k != 'unknown_percentile'})
    values['unknown_percentile'] = unknown_percentile
    return values


def validate_advanced(raw, number: Callable) -> dict:
    """The stored overrides, checked and normalised: unknown keys are ignored (another version wrote them), a value equal to the default is
    dropped, a bad value raises ValueError. `number(value, name, low, high, integer)` is the settings module's number check."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError('說話者標記的進階參數須為物件。')
    out = {}
    for knob in STORED:
        if knob.key not in raw:
            continue
        value = number(raw[knob.key], f'進階參數「{knob.label}」', knob.low, knob.high, integer=knob.integer)
        if value != knob.default:
            out[knob.key] = value
    values = effective(out, DEFAULTS['unknown_percentile'])
    if not values['min_unit'] <= values['target'] <= values['max_merged']:
        raise ValueError('單位長度須符合「最短單位 ≤ 單位目標長度 ≤ 合併後單位上限」。')
    return out


def changed(values: dict) -> dict:
    """The knobs of `values` that differ from their default (for the log and the saved statistics)."""
    return {key: values[key] for key in DEFAULTS if key in values and values[key] != DEFAULTS[key]}
