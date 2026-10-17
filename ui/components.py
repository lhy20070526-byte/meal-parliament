from __future__ import annotations

import html
import re
from typing import Any


PRESENTATIONS = {
    "营养师": ("林营养师", "营养搭配", "帮你补足蔬菜、蛋白质和主食"),
    "钱包": ("钱顾问", "餐费预算", "守住这一餐的花费上限"),
    "懒惰": ("周顾问", "时间与精力", "照顾今天的体力和动手成本"),
    "冰箱": ("陈管家", "食材管理", "只使用你明确说有的食材"),
    "口味": ("唐顾问", "饮食偏好", "保留你今晚真正想吃的感觉"),
    "议长": ("餐桌主持人", "用餐建议", "把不同意见整理成今晚能做的选择"),
}


def presentation(name: str) -> tuple[str, str, str]:
    return PRESENTATIONS.get(name, (name, "专家", "等待发言"))


def render_role_seats(st: Any, active_name: str | None = None) -> None:
    colors = {"营养师": "#2f8f6b", "钱包": "#b8860b", "懒惰": "#8d6e63", "冰箱": "#3887a8", "口味": "#c85b45", "议长": "#174c3b"}
    agents = [("营养师", "🥗"), ("钱包", "💰"), ("懒惰", "🛋️"), ("冰箱", "🧊"), ("口味", "🌶️"), ("议长", "🎙️")]
    cards = []
    for name, emoji in agents:
        nickname, title, trait = presentation(name)
        border = colors.get(name, "#174c3b") if name == active_name else "#d9e1d8"
        cards.append(f"<div style='border-top:4px solid {border};padding:8px 4px;min-height:84px'><b>{emoji} {nickname}</b><br><small>{title}</small><br><small>{trait}</small></div>")
    st.markdown("<div style='display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px'>" + "".join(cards) + "</div><style>@media(max-width:700px){div[style*='repeat(6']{grid-template-columns:repeat(3,minmax(0,1fr)) !important}}</style>", unsafe_allow_html=True)


def render_weights(st: Any, weights: dict[str, dict[str, Any]]) -> None:
    st.subheader("今天更在意什么")
    for name, item in weights.items():
        nickname, title, _ = presentation(name)
        st.markdown(f"**{item.get('emoji', '')} {nickname} · {title}**")
        st.progress(min(float(item["value"]) / 3.0, 1.0), text="参考程度")
        with st.expander("为什么更在意这一点", expanded=False):
            for reason in item.get("reasons", []):
                st.caption("• " + _plain_text(reason))


def _plain_text(text: str) -> str:
    """Keep internal evidence readable without exposing implementation labels."""
    human_reasons = {
        "基础权重 1.0": "先按同等程度参考",
        "饥饿度≥8或时间≥21:00，偏好低摩擦方案 ×2": "今天比较饿或时间较晚，优先省时省力",
        "饥饿度≥8或时间≥21:00，降低等待/价格敏感度 ×0.7": "时间较紧，适当放宽等待和价格取舍",
        "本周支出严格超过预算的80%，预算约束 ×2": "近期花费接近预算上限，今天更关注花费",
        "冰箱包含已过期描述，安全检查 ×1.5": "冰箱里有过期提示，先核对食材安全",
        "冰箱包含临期/即将过期食材，利用窗口 ×1.5": "冰箱里有临期提示，先核对食材安全",
        "步数>12000，恢复成本权重 ×1.3": "今天活动量较大，优先省力的安排",
        "用户愿意做饭，库存方案偏好 ×1.15": "你愿意做饭，可以多参考现有食材",
        "用户不愿意做饭，降低执行摩擦 ×1.2": "你今天不想做饭，优先省事的安排",
        "用户愿意买菜，补齐缺料方案偏好 ×1.1": "你愿意买菜，可以考虑补齐缺少的食材",
        "用户不愿意买菜，严格核对现有库存 ×1.2": "你不想买菜，只考虑明确已有的食材",
        "特殊需求为减脂，营养约束 ×1.8": "今天有减脂需求，优先照顾营养搭配",
        "特殊需求为减脂，重口偏好需收敛 ×0.8": "今天有减脂需求，口味适当收一收",
        "特殊需求为增肌，蛋白质约束 ×1.5": "今天有增肌需求，优先照顾蛋白质",
        "特殊需求为增肌，成本可让步 ×0.8": "今天有增肌需求，必要时给营养留出空间",
    }
    for source, target in human_reasons.items():
        text = text.replace(source, target)
    replacements = {
        "各角色分歧未达到3分，仍保留完整投票trace供复核": "顾问们的意见比较接近，最终按你的预算、意愿和现有食材确定了这份建议。",
        "trace": "讨论记录",
        "权重": "参考程度",
        "加权主分": "综合参考",
        "独立评分": "分别参考",
        "硬约束": "必须满足的条件",
        "候选": "方案",
        "本周预算减已支出": "本餐预算",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\[[A-Za-z][A-Za-z0-9_-]*\]", "", text)
    text = re.sub(r"(?<![A-Za-z0-9])P\d+(?!\d)[：:、\s]*", "", text)
    return html.escape(text)


def render_messages(st: Any, messages: list[Any]) -> None:
    colors = {"营养师": "#2f8f6b", "钱包": "#b8860b", "懒惰": "#8d6e63", "冰箱": "#3887a8", "口味": "#c85b45", "议长": "#174c3b"}
    for message in messages:
        old = " · 之前的建议" if message.error else ""
        color = colors.get(message.name, "#174c3b")
        nickname, title, _ = presentation(message.name)
        stance = "反对" if any(token in message.text for token in ("反对", "不接受", "出局")) else "让步" if any(token in message.text for token in ("但", "允许", "收一收")) else "支持" if any(token in message.text for token in ("支持", "同意", "保留满足感")) else "判断"
        safe_text = re.sub(r"\[M\d+\]", "", message.text)
        safe_text = _plain_text(safe_text)
        st.markdown(
            f"<div class='message-card' style='border-color:{color}55'><span class='agent-name' style='color:{color}'>{html.escape(message.emoji)} {html.escape(nickname)} · {html.escape(title)}</span> "
            f"<span style='background:{color};color:white;border-radius:10px;padding:2px 7px;font-size:.75rem'>{stance}</span> "
            f"<span class='weight'>第{message.round}轮{old}</span><br>{safe_text}</div>",
            unsafe_allow_html=True,
        )


def render_proposals(st: Any, proposals: list[Any], weighted_scores: dict[str, float], normalized: dict[str, float], winner: str) -> None:
    for proposal in proposals:
        status = "✅ 可以考虑" if proposal.feasible else f"⛔ {_plain_text(proposal.constraint_note)}"
        marker = "🏆 " if proposal.id == winner else ""
        st.markdown(f"### {marker}{_plain_text(proposal.mode)} · {_plain_text(proposal.dish)}")
        st.write(_plain_text(proposal.reason))
        st.caption(f"预计花费 ¥{proposal.estimated_cost:.2f} · {status}")
        if proposal.missing_ingredients:
            st.caption("还需要：" + "、".join(proposal.missing_ingredients))
        st.write("做法：" + _plain_text(" → ".join(proposal.steps)))


def render_vote_matrix(st: Any, votes: list[Any], proposals: list[Any]) -> None:
    st.subheader("顾问各自的看法")
    rows = []
    for vote in votes:
        nickname, title, _ = presentation(vote.agent)
        row = {"轮次": "初投" if vote.round == 1 else "复投", "顾问": f"{vote.emoji} {nickname} · {title}"}
        row.update({proposal.dish: "更推荐" if vote.scores.get(proposal.id, 0) >= 7 else "保留意见" for proposal in proposals})
        row["理由"] = _plain_text(vote.rationale)
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_result(st: Any, result: Any) -> None:
    rec = result.recommendation
    st.markdown(f"<div class='recipe-note'><h2>今晚就吃：{_plain_text(rec.dish)}</h2><p>{_plain_text(rec.mode)} · {_plain_text(rec.reason)}</p></div>", unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric("预计花费", f"¥{rec.estimated_cost:.2f}")
    cols[1].metric("做法", f"{len(rec.steps)}步")
    cols[2].metric("提醒", "请核对食材")
    st.write("提醒：" + _plain_text(rec.nutrition))
    st.subheader("跟着做，准备开饭")
    for index, step in enumerate(rec.steps, 1):
        st.write(f"{index}. {_plain_text(step)}")
    st.write("顾问怎样达成一致：" + _plain_text(rec.disagreements))
