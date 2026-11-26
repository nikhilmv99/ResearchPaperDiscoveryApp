"""Admin dashboard: real query executions recorded in Langfuse (read through the backend)."""
import json

import altair as alt
import pandas as pd
import streamlit as st

RANGES = {"Last hour": 1, "Last 24 hours": 24, "Last 7 days": 24 * 7, "Last 30 days": 24 * 30}
KIND = {"GENERATION": "LLM", "EMBEDDING": "Retrieval", "RETRIEVER": "Retrieval", "TOOL": "Web"}
KINDS = ["LLM", "Retrieval", "Web", "Workflow"]


def _dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except AttributeError:  # older Streamlit
        return False


def _kind_colors() -> list[str]:
    return (["#3987e5", "#d95926", "#199e70", "#6f6e69"] if _dark()
            else ["#2a78d6", "#eb6834", "#1baf7a", "#a3a29b"])


def _secs(v) -> str:
    if v is None or pd.isna(v):
        return "running"
    return f"{v * 1000:.0f} ms" if v < 1 else f"{v:.1f} s"

def _compact(n) -> str:
    n = int(n)
    for div, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if n >= div:
            return f"{n / div:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{n:,}"

def _overview(df: pd.DataFrame, steps: pd.DataFrame) -> None:
    done = df["seconds"].dropna()
    st.markdown("""<style>
.st-key-trace_kpis [data-testid="stMetricValue"],
.st-key-trace_kpis [data-testid="stMetricValue"] *{font-size:1.3rem!important;}
.st-key-trace_kpis [data-testid="stMetricLabel"] *{font-size:.8rem!important;}
</style>""", unsafe_allow_html=True)
    with st.container(key="trace_kpis"):
        c = st.columns([1, 1, 1, 1, 1.5])
        c[0].metric("Queries", f"{len(df):,}")
        c[1].metric("Failed", f"{int(df['failed'].sum()):,}")
        c[2].metric("Median time", _secs(done.median()) if len(done) else "-")
        c[3].metric("95th percentile", _secs(done.quantile(0.95, interpolation="higher")) if len(done) else "-")
        c[4].metric("Tokens in / out",
                    f"{_compact(df['tokens_in'].sum())} / {_compact(df['tokens_out'].sum())}",
                    help=f"{df['tokens_in'].sum():,} in / {df['tokens_out'].sum():,} out")

    left, right = st.columns(2)
    with left:
        st.markdown("**Time per query**")
        d = df.dropna(subset=["seconds"]).assign(Result=lambda x: x["failed"].map({True: "Failed", False: "OK"}))
        ok, bad = _kind_colors()[0], "#d03b3b"
        st.altair_chart(alt.Chart(d).mark_point(filled=True, size=80, opacity=1).encode(
            x=alt.X("time:T", title=None, scale=alt.Scale(padding=12)),
            y=alt.Y("seconds:Q", title="Seconds", scale=alt.Scale(zero=True)),
            color=alt.Color("Result:N", scale=alt.Scale(domain=["OK", "Failed"], range=[ok, bad]),
                            legend=alt.Legend(orient="top", title=None)),
            shape=alt.Shape("Result:N", scale=alt.Scale(domain=["OK", "Failed"], range=["circle", "cross"]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("question:N", title="Query"), alt.Tooltip("name:N", title="Type"),
                     alt.Tooltip("seconds:Q", title="Seconds", format=".2f"),
                     alt.Tooltip("time:T", title="Time", format="%d %b %H:%M:%S")],
        ).properties(height=260), width="stretch")
    with right:
        st.markdown("**Median time per step**")
        if steps.empty:
            st.caption("No steps recorded.")
        else:
            top = steps.sort_values("median_s", ascending=False).head(10)
            st.altair_chart(alt.Chart(top).mark_bar(color=_kind_colors()[0], size=16, cornerRadiusEnd=4).encode(
                x=alt.X("median_s:Q", title="Seconds"),
                y=alt.Y("name:N", sort="-x", title=None, axis=alt.Axis(labelLimit=180)),
                tooltip=[alt.Tooltip("name:N", title="Step"), alt.Tooltip("count:Q", title="Runs"),
                         alt.Tooltip("median_s:Q", title="Median (s)", format=".3f"),
                         alt.Tooltip("p95_s:Q", title="95th pct (s)", format=".3f")],
            ).properties(height=260), width="stretch")


def _waterfall(steps: list[dict]) -> None:
    end = max([s["start_s"] + (s["seconds"] or 0) for s in steps] + [0.001])
    rows = []
    for i, s in enumerate(steps, 1):
        mark = {"ERROR": "✕ ", "WARNING": "! "}.get(s["level"], "")
        rows.append({
            # the number keeps repeated steps (e.g. three web searches) on separate rows
            "row": f"{i:02d}  {'· ' * s['depth']}{mark}{s['name']}",
            "start": s["start_s"],
            # very short steps get a visible sliver; the tooltip and label keep the real time
            "end": max(s["start_s"] + (s["seconds"] or 0), s["start_s"] + end * 0.004),
            "Kind": KIND.get(s["type"], "Workflow"), "time": _secs(s["seconds"]), "status": s["status"] or "",
        })
    df = pd.DataFrame(rows)
    x = alt.X("start:Q", title="Seconds since the query started", scale=alt.Scale(domain=[0, end * 1.12]))
    y = alt.Y("row:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, labelAlign="left", labelPadding=200))
    tip = [alt.Tooltip("row:N", title="Step"), "Kind:N", alt.Tooltip("time:N", title="Took"),
           alt.Tooltip("start:Q", title="Started at (s)", format=".3f"), alt.Tooltip("status:N", title="Status")]
    base = alt.Chart(df).encode(y=y, tooltip=tip)
    bars = base.mark_bar(size=14, cornerRadius=3).encode(
        x=x, x2="end:Q", color=alt.Color("Kind:N", scale=alt.Scale(domain=KINDS, range=_kind_colors()),
                                         legend=alt.Legend(orient="top", title=None)))
    ink = "#c3c2b7" if _dark() else "#52514e"
    labels = base.mark_text(align="left", dx=5, color=ink).encode(x="end:Q", text="time:N")
    st.altair_chart((bars + labels).properties(height=28 * len(rows) + 60), width="stretch")


def _show(label: str, value) -> None:
    if value in (None, "", {}, []):
        return
    st.caption(label)
    if isinstance(value, str):
        st.code(value, language=None, wrap_lines=True)
    elif isinstance(value, list) and value and all(isinstance(m, dict) and "role" in m for m in value):
        for m in value:  # chat messages
            st.code(f"[{m['role']}]\n{m.get('content', '')}", language=None, wrap_lines=True)
    else:
        st.code(json.dumps(value, indent=2, ensure_ascii=False, default=str), language="json", wrap_lines=True)


def _detail(call, row: dict) -> None:
    st.subheader(row["question"] or row["name"])
    cache = st.session_state.setdefault("trace_details", {})
    if row["trace_id"] not in cache:
        with st.spinner("Loading the trace..."):
            d = call("get", f"/traces/{row['trace_id']}")
        if not d:
            return
        cache[row["trace_id"]] = d
    d = cache[row["trace_id"]]
    st.caption(f"{row['name']}  |  {row['user'] or 'unknown user'}  |  {_secs(row['seconds'])}  |  "
               f"{row['llm_calls']} LLM call(s)  |  {row['tokens_in']:,} tokens in, {row['tokens_out']:,} out")
    if row["failed"]:
        st.error(row["error"] or "The query failed.")
    if d["url"]:
        st.link_button("Open in Langfuse", d["url"])
    _waterfall(d["steps"])
    st.markdown("**Steps**")
    for i, s in enumerate(d["steps"], 1):
        mark = {"ERROR": "✕ ", "WARNING": "! "}.get(s["level"], "")
        with st.expander(f"{i:02d}  {mark}{s['name']}  -  {_secs(s['seconds'])}"):
            if s["status"]:
                st.warning(s["status"])
            if s["model"]:
                st.caption(f"Model: {s['model']}  |  tokens: {s['tokens_in']:,} in, {s['tokens_out']:,} out")
            _show("Input", s["input"])
            _show("Output", s["output"])
            _show("Metadata", {k: v for k, v in s["metadata"].items()
                               if k in ("queue_wait_s", "retries") and v not in ([], None)})


def render(call) -> None:
    c1, c2 = st.columns([3, 1])
    label = c1.selectbox("Time range", list(RANGES), index=1, key="trace_range")
    # Streamlit reruns every tab on every click, so Langfuse is only asked when the button is pressed.
    if c2.button("Load traces", type="primary", width="stretch"):
        with st.spinner("Reading traces from Langfuse..."):
            data = call("get", "/traces", params={"hours": RANGES[label]})
        if data is not None:
            st.session_state.trace_data, st.session_state.trace_label = data, label
            st.session_state.trace_details = {}
    data = st.session_state.get("trace_data")
    if data is None:
        st.info("Press Load traces to see recorded queries.")
        return
    if not data.get("enabled", True):
        st.info(
            "Langfuse tracing is disabled. "
            "Configure both Langfuse keys in the backend .env file."
        )
        return
    st.caption(f"Showing: {st.session_state.trace_label.lower()}. "
               "A new query can take a few seconds to appear.")
    if data["truncated"]:
        st.warning("There are more traces than can be shown, so these numbers are partial. Pick a shorter range.")
    if not data["traces"]:
        st.info("No queries recorded in this time range. Run a search or ask a question, then load again.")
        return

    df = pd.DataFrame(data["traces"])
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df["seconds"] = pd.to_numeric(df["seconds"])
    _overview(df, pd.DataFrame(data["steps"]))

    st.markdown("**Queries** (select one to see how it ran)")
    ctx = getattr(st, "context", None)
    tz = getattr(ctx, "timezone", None) or "UTC"      # the chart uses the browser's time zone too
    table = df.assign(time=df["time"].dt.tz_convert(tz).dt.tz_localize(None), status=[("✕ failed" if f else f"! {n} issue(s)" if n else "ok")
                              for f, n in zip(df["failed"], df["step_problems"])])
    picked = st.dataframe(
        table[["time", "status", "name", "question", "user", "seconds", "llm_calls", "tokens_in", "tokens_out"]],
        hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="trace_table",
        column_config={
            "time": st.column_config.DatetimeColumn("Time", format="D MMM, HH:mm:ss"),
            "name": "Type", "question": st.column_config.TextColumn("Query", width="medium"), "user": "User",
            "seconds": st.column_config.NumberColumn("Seconds", format="%.2f"), "llm_calls": "LLM calls",
            "tokens_in": "Tokens in", "tokens_out": "Tokens out", "status": "Status"})
    rows = picked.selection.rows
    if rows and rows[0] < len(data["traces"]):
        _detail(call, data["traces"][rows[0]])