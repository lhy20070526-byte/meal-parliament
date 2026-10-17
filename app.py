"""Streamlit entrypoint for the Meal Parliament demonstration."""

from __future__ import annotations

import json
import hashlib
import re
import time as monotonic_clock
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from core.llm_config import LLMConfigError, LLMConnectionError, check_connection, get_llm_config
from core.models import UserInput
from core.needs import NeedsError, parse_needs
from core.parliament import Parliament
from core.weights import calculate_weights
from core.vision import EXAMPLE_ITEMS, VisionError, VisionResult, recognize_fridge_image
from ui.hunger import hunger_slider
from ui.components import PRESENTATIONS, presentation, render_messages, render_proposals, render_result, render_role_seats, render_vote_matrix, render_weights

st.set_page_config(page_title="🍽️ 外卖/做饭决策议会", page_icon="🍽️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --cream:#fff9f0; --green:#315b45; --ink:#392d24; --muted:#716356; --tomato:#b84d32; }
.stApp { background:var(--cream); color:var(--ink); }
[data-testid="stSidebar"] { background:#f5eddf; border-right:1px solid #e5d8c6; }
[data-testid="stSidebar"] h2 { color:#68492f; }
[data-testid="stMainBlockContainer"] { max-width:1120px; padding-top:3.5rem; }
h1 { font-size:clamp(1.8rem, 3.2vw, 2.7rem) !important; line-height:1.3 !important; color:#4f392a; }
h3 { font-size:1.1rem !important; color:#69462f; }
p, li { line-height:1.75; }
button { min-height:42px; }
button[kind="primary"] { background:var(--tomato); color:white; border:1px solid var(--tomato); border-radius:14px; }
button[kind="primary"]:hover { background:#993d27; border-color:#993d27; }
button[kind="secondary"] { border-radius:12px; border-color:#d4c2ac; background:#fffcf6; }
input, textarea { caret-color:var(--tomato); }
::selection { background:#f4c6aa; color:#392d24; }
:focus-visible { outline:3px solid #b84d32 !important; outline-offset:3px; }
[data-testid="stTabs"] [role="tablist"] { gap:20px; }
[data-testid="stTabs"] [role="tab"] { font-size:1rem; color:#715c49; min-height:48px; }
[data-testid="stTabs"] [aria-selected="true"] { color:#a6412c; font-weight:700; }
.message-card { background:#fffdf8; border:1px solid #e4d9c9; border-radius:18px; padding:18px 20px; margin:12px 0; line-height:1.8; }
.agent-name { color:var(--green); font-weight:700; }
.weight { color:var(--muted); font-size:.82rem; float:right; }
.welcome-table { display:flex; gap:28px; align-items:center; padding:28px 0 32px; border-bottom:1px solid #e5d8c6; }
.welcome-table svg { flex:0 0 112px; }
.welcome-table h2 { font-size:1.65rem; line-height:1.4; padding:0 0 12px; color:#68432c; }
.welcome-table p { margin:0; max-width:42ch; color:#715c49; }
.recipe-note { background:#f7e8d6; padding:24px; border-radius:20px 20px 20px 4px; margin:20px 0; }
.recipe-note h2 { font-size:1.6rem; margin:0 0 12px; color:#65402a; }
.recipe-note p { margin:0; }
@media (max-width: 700px) {
 .weight { float:none; display:block; }
 [data-testid="stMainBlockContainer"] { padding:4rem 1.1rem 2rem; }
 .welcome-table { gap:16px; align-items:flex-start; }
 .welcome-table svg { flex-basis:62px; width:62px; height:80px; }
 .welcome-table h2 { font-size:1.25rem; }
 .message-card { padding:14px; }
 .recipe-note { padding:20px; }
}
/* Keep the sidebar toggle in Streamlit's header; hide only its toolbar and menu. */
header[data-testid="stHeader"] { background:transparent; }
header[data-testid="stHeader"] [data-testid="stToolbarActions"],
header[data-testid="stHeader"] [data-testid="stMainMenu"],
header[data-testid="stHeader"] [data-testid="stDecoration"],
header[data-testid="stHeader"] [data-testid="stStatusWidget"] { display:none !important; }
[data-testid="stAppDeployButton"], [data-testid="stFileUploaderDropzoneInstructions"] { display:none !important; }
[data-testid="stExpandSidebarButton"]::after { content:"填写用餐需求"; font-size:.9rem; margin-left:6px; }
[data-testid="stFileUploaderDropzone"] { position:relative; }
[data-testid="stFileUploaderDropzone"] * { color:transparent !important; font-size:0 !important; }
[data-testid="stFileUploaderDropzone"] button { color:#174c3b !important; font-size:0 !important; }
[data-testid="stFileUploaderDropzone"] button span { display:none !important; }
[data-testid="stFileUploaderDropzone"] button::after { content:"选择图片"; color:#174c3b; font-size:.9rem; }

[data-testid="stFileChipDeleteBtn"] button::after { content:"移除"; }
[data-testid="stFileChipName"]::after { content:"已选择照片"; color:#174c3b; font-size:.85rem; }
[data-testid="stCameraInput"] button { color:#174c3b !important; font-size:0 !important; }
[data-testid="stCameraInput"] button span { display:none !important; }
[data-testid="stCameraInput"] button::after { content:"拍照"; color:#174c3b; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

DEFAULT_INPUT = dict(
    hunger=5, steps=0, screen_hours=0.0, meal_budget=40.0,
    fridge="", recent_diet="", taste="", special_needs="无", wish="",
    cooking_willingness="看情况", shopping_willingness="看情况",
)


def _clear_vision() -> None:
    st.session_state.vision_source_hash = None
    st.session_state.vision_result = None
    st.session_state.vision_items = None
    st.session_state.vision_applied_hash = None
    st.session_state.vision_edit_revision = st.session_state.get("vision_edit_revision", 0) + 1
    st.session_state.vision_source_type = None


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    if isinstance(exc, VisionError) or "识图" in text or "图片" in text:
        return text
    if "未配置" in text or "密钥" in text:
        return "智能建议服务还没有准备好，请稍后再试或先使用示例体验。"
    if "少于两个" in text or "候选" in text:
        return "目前没有找到至少两种安全可行的吃法，请补充冰箱食材或调整预算和意愿。"
    if "引用" in text:
        return "顾问讨论没有顺利完成，请再试一次。"
    return "这次商量没有完成，请检查输入后再试。"


def _display_text(text: str) -> str:
    """Remove internal citation labels and implementation wording from UI copy."""
    text = re.sub(r"\[[A-Za-z][A-Za-z0-9_-]*\]", "", text)
    text = re.sub(r"(?<![A-Za-z0-9])P\d+(?!\d)[：:、\s]*", "", text)
    for source, target in {
        "各角色分歧未达到3分，仍保留完整投票trace供复核": "顾问们的意见比较接近，最终按你的预算、意愿和现有食材确定了这份建议。",
        "trace": "讨论记录",
        "权重": "参考程度",
        "加权主分": "综合参考",
        "独立评分": "分别参考",
        "硬约束": "必须满足的条件",
        "候选": "方案",
        "本周预算减已支出": "本餐预算",
    }.items():
        text = text.replace(source, target)
    return text.strip()


def _vision_section(llm_config) -> None:
    st.markdown("### 看看冰箱里有什么")
    st.session_state.setdefault("vision_edit_revision", 0)
    st.caption("拍照或上传一张常见图片；只有点击识别后才会交给智能建议服务。")
    source_type = st.radio("图片来源", ["上传照片", "拍照"], horizontal=True, key="fridge_source_type")
    if source_type != st.session_state.get("vision_source_type"):
        st.session_state.vision_source_type = source_type
        st.session_state.vision_result = None
        st.session_state.vision_items = None
        st.session_state.vision_applied_hash = None
        st.session_state.vision_edit_revision += 1
    if source_type == "拍照":
        source = st.camera_input("拍一张冰箱照片", key="fridge_camera")
    else:
        source = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png", "webp"], max_upload_size=8, key="fridge_upload")
        st.caption("每张照片不超过8兆，支持常见照片。")
    raw = source.getvalue() if source is not None else None
    source_hash = hashlib.sha256(raw).hexdigest() if raw else None
    if source_hash != st.session_state.get("vision_source_hash"):
        st.session_state.vision_source_hash = source_hash
        st.session_state.vision_result = None
        st.session_state.vision_items = None
        st.session_state.vision_applied_hash = None
        st.session_state.vision_edit_revision = st.session_state.get("vision_edit_revision", 0) + 1
    if raw and st.button("识别这张照片", key="recognize_fridge"):
        try:
            with st.spinner("正在看看照片里的食材…"):
                result = recognize_fridge_image(raw)
            st.session_state.vision_result = result.model_dump(mode="json")
            st.session_state.vision_items = [item.model_dump(mode="json") for item in result.items]
            st.session_state.vision_edit_revision += 1
        except Exception as exc:
            st.session_state.vision_result = {"status": "error", "notice": _friendly_error(exc), "items": [], "uncertainties": []}
            st.session_state.vision_items = None
    if not llm_config.configured or llm_config.parliament_mode in {"mock", "demo"}:
        st.caption("当前为示例体验，示例内容不代表识别上传照片。")
        if st.button("加载示例食材", key="load_example_food"):
            st.session_state.vision_result = VisionResult(status="mock", notice="示例内容不代表识别上传照片。", items=EXAMPLE_ITEMS, uncertainties=["照片看不出新鲜程度，请确认食材仍可食用。"]).model_dump(mode="json")
            st.session_state.vision_items = [item.model_dump(mode="json") for item in EXAMPLE_ITEMS]
            st.session_state.vision_edit_revision += 1
    result_data = st.session_state.get("vision_result") or {}
    if result_data.get("status") == "error":
        st.error(result_data.get("notice", "识别没有完成，请重试。"))
    items = st.session_state.get("vision_items")
    if items is not None:
        st.info(result_data.get("notice") or "请确认下面的食材清单，再用于今晚的商量。")
        edited: list[dict] = []
        revision = st.session_state.get("vision_edit_revision", 0)
        for index, item in enumerate(items):
            cols = st.columns([2, 1])
            name = cols[0].text_input("食材名称", value=item.get("name", ""), key=f"vision_name_{revision}_{index}")
            quantity = cols[1].text_input("大概数量", value=item.get("quantity", "未标注"), key=f"vision_quantity_{revision}_{index}")
            if name.strip():
                edited.append({"name": name.strip(), "quantity": quantity.strip() or "未标注"})
        uncertainties = result_data.get("uncertainties", [])
        if uncertainties:
            st.caption("需要你确认：" + "；".join(uncertainties))
        st.caption("照片看不出新鲜程度，请确认食材仍可食用。")
        if st.button("确认食材并用于今晚商量", type="primary", disabled=not edited, key="apply_fridge_items"):
            fridge = "、".join(f"{item['name']}（{item['quantity']}）" for item in edited)
            st.session_state.input_fridge = fridge
            st.session_state.input_values["fridge"] = fridge
            st.session_state.vision_applied_hash = source_hash
            st.success("已加入冰箱清单；你仍可以手动修改下面的文字。")


def _input_form() -> UserInput:
    default = DEFAULT_INPUT.copy()
    default["time"] = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%H:%M")
    values = st.session_state.setdefault("input_values", default)
    for field, value in values.items():
        st.session_state.setdefault(f"input_{field}", value)
    llm_config = get_llm_config()
    with st.sidebar:
        st.markdown("## 今晚，想怎么吃？")
        st.caption("用自己的话说说今晚的想法，我们帮你整理，再一起商量吃什么。")
        input_method = st.radio("你想怎么告诉我们？", ["直接说需求", "按问题填写"], horizontal=True, key="input_method")
        if input_method == "直接说需求":
            description = st.text_area(
                "今晚有什么需求？", max_chars=4000, height=120,
                placeholder="例如：这顿最多花30元，不想买菜，愿意做点简单的。冰箱有鸡蛋和番茄，想吃清淡些。",
                key="needs_description",
            )
            st.caption("点击下面的按钮后，会将这段需求交给智能建议服务理解。")
            if st.button("帮我整理需求", use_container_width=True):
                try:
                    with st.spinner("正在理解你的需求…"):
                        parsed = parse_needs(description)
                    if "hunger" in parsed:
                        st.session_state.needs_hunger_revision = st.session_state.get("needs_hunger_revision", 0) + 1
                    for field, value in parsed.items():
                        st.session_state[f"input_{field}"] = value
                        values[field] = value
                    st.session_state.input_wish = description.strip()
                    values["wish"] = description.strip()
                    st.success("已整理，请检查下面的内容；没有提到的项目保持原样。" if parsed else "已记下你的想法，请再补充下面的用餐信息。")
                except NeedsError as exc:
                    st.warning(str(exc))
        st.markdown("### 确认你的用餐需求" if input_method == "直接说需求" else "### 回答几个小问题")
        st.selectbox("今天愿意做饭吗？", ["愿意", "看情况", "不愿意"], key="input_cooking_willingness")
        st.selectbox("今天愿意买菜吗？", ["愿意", "看情况", "不愿意"], key="input_shopping_willingness")
        st.number_input("这顿最多花多少（元）", min_value=0.0, step=5.0, key="input_meal_budget")
        st.markdown("### 现在有多饿？")
        hunger_value = hunger_slider(int(st.session_state.input_hunger), key=f"meal_hunger_slider_{st.session_state.get('needs_hunger_revision', 0)}")
        if hunger_value != st.session_state.input_hunger:
            st.session_state.input_hunger = hunger_value
            values["hunger"] = hunger_value
            st.rerun()
        values["hunger"] = hunger_value
        _vision_section(llm_config)
        st.text_input("用餐时间（北京时间）", help="请填写24小时制时间，例如21:30", key="input_time")
        st.number_input("今日步数", min_value=0, step=500, key="input_steps")
        st.number_input("今天看屏幕多久（小时）", min_value=0.0, step=0.5, key="input_screen_hours")
        st.text_area("冰箱里有什么", height=72, key="input_fridge")
        st.text_area("近3天饮食", height=68, key="input_recent_diet")
        st.text_input("喜欢什么口味？有什么不吃的？", key="input_taste")
        st.selectbox("饮食上有特别要求吗？", ["无", "减脂", "增肌", "素食"], key="input_special_needs")
        st.text_area("还有什么想告诉顾问的？（可选）", key="input_wish")
        if st.button("开始商量今晚吃什么", type="primary", use_container_width=True, key="start_top"):
            st.session_state.start_requested = True
        if st.button("重置", use_container_width=True, key="reset_bottom"):
            st.session_state.result = None
            st.session_state.error = None
            st.session_state.events = []
            st.session_state.replay_cursor = 0
            st.session_state.replay_playing = False
            st.session_state.replay_paused = False
            st.session_state.replay_next_at = None
            st.session_state.replay_notice = None
            _clear_vision()
            st.rerun()
    fields = ["time", "hunger", "steps", "screen_hours", "meal_budget", "fridge", "recent_diet", "taste", "special_needs", "wish", "cooking_willingness", "shopping_willingness"]
    return UserInput(**{field: st.session_state[f"input_{field}"] for field in fields})


try:
    user = _input_form()
except ValidationError as exc:
    st.sidebar.error("输入有误：时间请填写例如21:30，数值也需在允许范围内。请修改后再开始商量。")
    st.stop()
weights = calculate_weights(user)
st.title("🍽️ 外卖 / 做饭决策议会")
llm_config = get_llm_config()
parliament_mode = llm_config.parliament_mode
if llm_config.provider not in {"openai", "deepseek"}:
    mode_caption = "当前为示例体验。"
elif parliament_mode in {"mock", "demo"} or (parliament_mode == "auto" and not llm_config.configured):
    mode_caption = "当前为示例体验。"
elif llm_config.configured:
    mode_caption = "在线建议已开启；确认后，输入和主动识别的照片会交给智能建议服务。"
else:
    mode_caption = "当前为示例体验。"
st.caption("五位虚拟顾问会为你商量这一餐。" + mode_caption)

tab_hall, tab_weights, tab_final = st.tabs(["顾问讨论", "方案比较", "今晚吃什么"])
with tab_hall:
    progress_placeholder = st.empty()
with tab_weights:
    weight_placeholder = st.empty()

with st.sidebar:
    st.divider()
    st.markdown("### 智能顾问")
    if llm_config.configured:
        st.caption("状态：在线建议已开启")
    else:
        st.caption("状态：当前为示例体验")
    if st.button("检查服务连接", use_container_width=True, disabled=not llm_config.configured, key="check_llm_connection"):
        try:
            checked_model = check_connection()
        except LLMConnectionError:
            st.error("服务暂时没有回应，请稍后再试。")
        except LLMConfigError:
            st.error("在线建议还没有准备好；当前可以继续使用示例体验。")
        except Exception:
            st.error("服务连接没有成功，请稍后再试。")
        else:
            st.success("在线建议已开启。")
    st.divider()
    st.markdown("### 今天更在意什么")
    for name, item in weights.items():
        nickname, title, _ = presentation(name)
        st.markdown(f"**{item['emoji']} {nickname} · {title}**")
        st.progress(min(float(item["value"]) / 3.0, 1.0), text="参考程度")

if st.session_state.pop("start_requested", False):
        st.session_state.result = None
        parliament = Parliament()
        events: list[dict] = []
        stream_lines: list[str] = []
        stream_weights = deepcopy(weights)
        def on_event(event: dict) -> None:
            events.append(event)
            if event.get("kind") == "initialized":
                stream_weights.clear()
                stream_weights.update(deepcopy(event.get("weights", {})))
            elif event.get("kind") == "summary":
                for adjustment in event.get("adjustments", []):
                    item = stream_weights.get(adjustment.get("agent"))
                    if item:
                        item["value"] = round(item["value"] + adjustment.get("delta", 0), 4)
                        item.setdefault("reasons", []).append(adjustment.get("reason", "议长有限微调"))
            if event.get("kind") == "message":
                item = event["message"]
                nickname, title, _ = presentation(item["name"])
                safe_text = re.sub(r"\[M\d+\]", "", item["text"])
                stream_lines.append(f"**{item['emoji']} {nickname} · {title}**\n\n{safe_text}")
            elif event.get("kind") == "summary":
                stream_lines.append(f"**🎙️ 餐桌主持人整理**\n\n{_display_text(event['text'])}")
            elif event.get("kind") == "agent_error":
                stream_lines.append("⚠️ 一位顾问暂时没有完成发言。")
            with progress_placeholder.container():
                st.markdown("\n\n---\n\n".join(stream_lines[-10:]))
                st.caption("顾问正在交换意见…")
            with weight_placeholder.container():
                render_weights(st, stream_weights)
        with st.spinner("顾问正在商量这一餐…"):
            try:
                st.session_state.result = parliament.run(user, on_event=on_event)
                st.session_state.events = events
                st.session_state.error = None
                st.session_state.replay_cursor = 1
                st.session_state.replay_playing = False
                st.session_state.replay_paused = False
                st.session_state.replay_next_at = None
                st.session_state.replay_notice = None
            except Exception as exc:
                st.session_state.error = _friendly_error(exc)
                st.session_state.events = events
        progress_placeholder.empty()
        weight_placeholder.empty()

result = st.session_state.get("result")
error = st.session_state.get("error")
if error:
    st.error(error)
    st.caption("已完成的讨论会保留在本次页面中；修改输入后可以重新商量。")
if result and result.input_fingerprint != user.fingerprint:
            st.warning("你的输入已改变，下面的建议属于之前那一轮；请重新开始商量。")


def _focus(text: str) -> str:
    if any(token in text for token in ("预算", "支出", "元")):
        return "争论焦点：预算上限与营养/满足感的取舍"
    if any(token in text for token in ("过期", "冰箱", "剩饭", "食材")):
        return "争论焦点：食材安全、库存利用与执行成本"
    if any(token in text for token in ("口味", "辣", "日料", "清淡")):
        return "争论焦点：口味满足感与今晚的可执行性"
    return "争论焦点：今晚怎样用最少摩擦吃到合适的一餐"


def _conflict_digest(result: object) -> str:
    text = " ".join(message.text for message in result.messages)
    points: list[str] = []
    if any(token in text for token in ("预算", "剩余", "元")):
        points.append("钱顾问守住这一餐的预算")
    if any(token in text for token in ("过期", "冰箱", "剩饭")):
        points.append("陈管家要求先确认食材安全")
    if any(token in text for token in ("饿", "洗切", "十五分钟", "锅")):
        points.append("周顾问提醒照顾今天的精力")
    if any(token in text for token in ("口味", "辣", "欲望", "日料")):
        points.append("唐顾问希望保留你喜欢的味道")
    if any(token in text for token in ("蔬菜", "蛋白质")):
        points.append("林营养师提醒补足蔬菜和蛋白质")
    if not points:
        points.append("五位顾问正在围绕你的输入寻找可行选择")
    return "餐桌主持人抓住的重点：" + "；".join(points) + "。"


if result and hasattr(st, "fragment"):
    @st.fragment(run_every="2s")
    def _replay_clock() -> None:
        if st.session_state.get("replay_playing"):
            cursor = st.session_state.get("replay_cursor", 0)
            now = monotonic_clock.monotonic()
            next_at = st.session_state.get("replay_next_at")
            if next_at is None:
                st.session_state.replay_next_at = now + 2.0
                return
            if cursor >= len(result.messages):
                st.session_state.replay_playing = False
                st.session_state.replay_next_at = None
                st.session_state.replay_notice = "已完成，点击“今晚吃什么”查看建议。"
                return
            if now < next_at:
                return
            st.session_state.replay_cursor = cursor + 1
            if cursor + 1 >= len(result.messages):
                st.session_state.replay_playing = False
                st.session_state.replay_next_at = None
                st.session_state.replay_notice = "已完成，点击“今晚吃什么”查看建议。"
            else:
                st.session_state.replay_next_at = now + 2.0
            st.rerun()
    _replay_clock()

with tab_hall:
    if result:
        st.subheader("六位顾问")
        messages = result.messages
        cursor = max(0, min(st.session_state.get("replay_cursor", len(messages)), len(messages)))
        controls = st.columns(4)
        if controls[0].button("逐条下一条", key="replay_step", use_container_width=True):
            st.session_state.replay_playing = False
            st.session_state.replay_paused = False
            st.session_state.replay_next_at = None
            st.session_state.replay_cursor = min(cursor + 1, len(messages))
            st.rerun()
        if controls[1].button("自动播放/继续", key="replay_play", use_container_width=True):
            st.session_state.replay_playing = True
            st.session_state.replay_paused = False
            st.session_state.replay_next_at = monotonic_clock.monotonic() + 2.0
            st.session_state.replay_notice = None
            st.rerun()
        if controls[2].button("暂停回放", key="replay_pause", use_container_width=True):
            st.session_state.replay_paused = True
            st.session_state.replay_playing = False
            st.session_state.replay_next_at = None
            st.rerun()
        if controls[3].button("快速看结果", key="replay_fast", use_container_width=True):
            st.session_state.replay_playing = False
            st.session_state.replay_next_at = None
            st.session_state.replay_cursor = len(messages)
            st.session_state.replay_notice = "已完成，点击“今晚吃什么”查看建议。"
            st.rerun()
        visible_messages = messages[:cursor]
        active_name = visible_messages[-1].name if visible_messages else None
        render_role_seats(st, active_name)
        if visible_messages:
            active = visible_messages[-1]
            nickname, title, _ = presentation(active.name)
            st.info(f"正在说：{active.emoji} {nickname} · {title}\n\n{_focus(active.text)}")
        else:
            st.caption("回放尚未开始；点击“逐条下一条”或“快速看结果”。")
        if st.session_state.get("replay_paused"):
            st.caption("回放已暂停；可继续逐条查看，或直接快速看结果。")
        if st.session_state.get("replay_notice"):
            st.success(st.session_state.replay_notice)
        render_messages(st, visible_messages)
        if cursor >= len(messages):
            st.subheader("餐桌主持人总结")
            st.info(_conflict_digest(result))
            with st.expander("看看顾问的完整整理", expanded=False):
                st.write(_display_text(result.summary))
        else:
            st.caption("顾问讨论完成后，餐桌主持人会整理冲突重点。")
        if result.extra_round:
            st.warning("顾问意见比较接近，又商量了一轮。")
    else:
        st.markdown("""<div class="welcome-table"><svg viewBox="0 0 140 140" width="112" height="132" aria-hidden="true"><path d="M42 33c-14-15 12-16 0-29M69 29c-14-15 12-16 0-29M96 33c-14-15 12-16 0-29" fill="none" stroke="#a68b6a" stroke-width="3" stroke-linecap="round"/><ellipse cx="70" cy="65" rx="59" ry="23" fill="#d2dfbd"/><path d="M12 64q6 56 58 56t58-56" fill="#ce6945"/><ellipse cx="70" cy="64" rx="52" ry="18" fill="#ffe4a8"/><path d="M33 63q10-13 24 0t25 0t25 0" fill="none" stroke="#e5b458" stroke-width="5"/><path d="M41 55l13-8m33 9l15-6" stroke="#5d7849" stroke-width="6" stroke-linecap="round"/><path d="M50 91q20 14 40 0" fill="none" stroke="#fff3dc" stroke-width="4" stroke-linecap="round"/><path d="M47 123h46" stroke="#9a5035" stroke-width="7" stroke-linecap="round"/></svg><div><h2>好好吃饭，<br>从照顾此刻的你开始。</h2><p>有多饿、想不想开火、这顿打算花多少？告诉我们，几位顾问一起帮你想。</p></div></div>""", unsafe_allow_html=True)
        st.info("打开“填写用餐需求”，填好后点击“开始商量今晚吃什么”。")
with tab_weights:
    render_weights(st, weights if not result else result.weights)
    st.caption("这里会把预算、时间、营养和口味放在一起比较，帮助你做决定。")
    if result:
        st.subheader("方案比较")
        render_proposals(st, result.proposals, result.weighted_scores, result.normalized_scores, result.winning_proposal_id)
with tab_final:
    if result:
        render_result(st, result)
        advice_text = "\n".join([
            f"今晚建议：{_display_text(result.recommendation.dish)}（{_display_text(result.recommendation.mode)}）",
            f"预计花费：¥{result.recommendation.estimated_cost:.2f}",
            f"为什么：{_display_text(result.recommendation.reason)}",
            "做法：" + _display_text(" → ".join(result.recommendation.steps)),
            f"提醒：{_display_text(result.recommendation.nutrition)}",
            f"顾问怎样达成一致：{_display_text(result.recommendation.disagreements)}",
        ])
        st.download_button("保存这份用餐建议", advice_text, file_name="今晚吃什么-用餐建议.txt", mime="text/plain", use_container_width=True)
    else:
        st.info("顾问商量结束后，这里会显示今晚的用餐建议。")
st.caption("提醒：照片看不出新鲜程度，请确认食材仍可食用；营养建议不能替代专业诊疗。")
