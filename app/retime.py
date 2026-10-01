"""按 source/target 时间锚点对 cue 做分段线性校时的纯逻辑层。

规则：

* 对每条 cue 的 start、end 分别做一维分段线性映射；
* 落在某个锚点区间内时按该区间线性插值，锚点位置精确映射到对应 target；
* 超出首尾锚点时沿最近（首段/末段）区间的斜率外推；
* 同一条 cue 的两端分别按各自所在区间独立计算（跨越锚点互不影响）；
* 映射后任一时刻为负，或 ``end <= start`` 的 cue 记为非法；
* id、text、speaker 与输入顺序保持不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Anchor, Cue


@dataclass
class RetimeResult:
    """retime_cues 的结果。

    存在非法 cue（映射时间为负或区间不合法）时 retimed 为空，
    negative_times / invalid_intervals 给出全部相关 cue id（输入顺序、去重）。
    """

    retimed: list[Cue] = field(default_factory=list)
    negative_times: list[str] = field(default_factory=list)
    invalid_intervals: list[str] = field(default_factory=list)
    cue_ids: list[str] = field(default_factory=list)


def map_time(t: float, anchors: list[Anchor]) -> float:
    """把源时间 t 按锚点分段线性映射到目标时间。

    精确命中某锚点时直接返回其 target（避免插值引入的浮点误差）；
    t 在首个锚点之前 / 末个锚点之后时，沿首段 / 末段外推。
    """
    n = len(anchors)
    for idx in range(n):
        if t == anchors[idx].source:
            return anchors[idx].target

    if t < anchors[0].source:
        lo, hi = 0, 1
    elif t > anchors[n - 1].source:
        lo, hi = n - 2, n - 1
    else:
        # 锚点 source 严格递增，找到 t 所在区间 [lo, hi]
        hi = 1
        while t > anchors[hi].source:
            hi += 1
        lo = hi - 1

    s0, s1 = anchors[lo].source, anchors[hi].source
    u0, u1 = anchors[lo].target, anchors[hi].target
    return u0 + (t - s0) * (u1 - u0) / (s1 - s0)


def retime_cues(cues: list[Cue], anchors: list[Anchor]) -> RetimeResult:
    """对全部 cue 的 start/end 分别做锚点映射。

    先统一计算、收集全部问题 cue：任一映射时间为负或映射后 end <= start
    即记入对应列表；只要存在问题 cue 就不输出任何结果。
    """
    retimed: list[Cue] = []
    negative_times: list[str] = []
    invalid_intervals: list[str] = []
    cue_ids: list[str] = []

    for cue in cues:
        new_start = map_time(cue.start, anchors)
        new_end = map_time(cue.end, anchors)

        bad = False
        if new_start < 0 or new_end < 0:
            negative_times.append(cue.id)
            bad = True
        if new_end <= new_start:
            invalid_intervals.append(cue.id)
            bad = True
        if bad:
            if cue.id not in cue_ids:
                cue_ids.append(cue.id)
            continue

        retimed.append(
            Cue(
                id=cue.id,
                start=new_start,
                end=new_end,
                text=cue.text,
                speaker=cue.speaker,
            )
        )

    if cue_ids:
        return RetimeResult(
            retimed=[],
            negative_times=negative_times,
            invalid_intervals=invalid_intervals,
            cue_ids=cue_ids,
        )

    return RetimeResult(retimed=retimed)
