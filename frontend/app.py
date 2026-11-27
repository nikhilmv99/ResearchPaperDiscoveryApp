import base64
import hashlib
import hmac
import html
import json
import os
import time

import pandas as pd
import requests
import streamlit as st
from traces_tab import render as render_traces

API = os.getenv("BACKEND_URL", "http://localhost:8000")
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-me-in-production-secret-key").encode("utf-8")
SESSION_TTL_HOURS = float(os.getenv("SESSION_TTL_HOURS", "1"))

st.set_page_config(page_title="Research Paper Discovery", page_icon="📚", layout="wide")


# ----------------------------------------------------------------------------- persistent auth helpers
def _b64_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def make_auth_param(token: str, username: str, role: str, ttl_hours: float = SESSION_TTL_HOURS) -> str:
    payload = {
        "t": token,
        "u": username,
        "r": role,
        "exp": int(time.time() + ttl_hours * 3600),
    }
    body = _b64_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _b64_encode(hmac.new(SESSION_SECRET, body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"


def parse_auth_param(param_val: str | None) -> dict | None:
    if not param_val or "." not in param_val:
        return None
    body, sig = param_val.split(".", 1)
    expected_sig = _b64_encode(hmac.new(SESSION_SECRET, body.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected_sig):
        return None
    try:
        data = json.loads(_b64_decode(body).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if int(data.get("exp", 0)) <= int(time.time()):
        return None
    if not data.get("t") or not data.get("u") or not data.get("r"):
        return None
    return data


def set_session_auth(token: str, username: str, role: str, ttl_hours: float = SESSION_TTL_HOURS):
    st.session_state.update(token=token, username=username, role=role, active_chat=None)
    st.query_params["auth"] = make_auth_param(token, username, role, ttl_hours=ttl_hours)


def clear_session_auth():
    for k in ("token", "username", "role", "active_chat"):
        st.session_state.pop(k, None)
    if "auth" in st.query_params:
        del st.query_params["auth"]


def restore_session_from_url():
    if "token" in st.session_state:
        return
    restored = parse_auth_param(st.query_params.get("auth"))
    if restored:
        st.session_state.update(
            token=restored["t"],
            username=restored["u"],
            role=restored["r"],
        )
    elif "auth" in st.query_params:
        del st.query_params["auth"]


# ----------------------------------------------------------------------------- theme (CSS)
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
:root{
  --bg:#0E1420;
  --surface:#161F30;
  --surface-soft:rgba(255,255,255,.035);
  --line:rgba(148,163,184,.16);
  --accent:#38BDF8;
  --accent-2:#818CF8;
  --text:#F1F5F9;
  --muted:#94A3B8;
}
h1,h2,h3,h4,p,label,li,td,th,input,textarea,button,[role="tab"]{
  font-family:'Plus Jakarta Sans',sans-serif;
}
.stApp{
  background:
    radial-gradient(900px 480px at 10% -8%,rgba(56,189,248,.11),transparent 60%),
    radial-gradient(850px 460px at 95% 0%,rgba(129,140,248,.11),transparent 55%),
    var(--bg);
}
header[data-testid="stHeader"]{background:transparent;}
footer{visibility:hidden;}
.block-container{max-width:1100px;padding-top:1.6rem;padding-bottom:3rem;}

.hero{padding:.25rem 0 1rem;}
.hero .kicker{
  font:600 .72rem 'JetBrains Mono',monospace;
  letter-spacing:.18em;
  color:var(--accent);
  text-transform:uppercase;
}
.hero h1{
  font-size:2.35rem;
  font-weight:700;
  letter-spacing:-.02em;
  margin:.25rem 0 .2rem;
  padding:0;
  color:var(--text);
}
.hero h1 span{
  background:linear-gradient(90deg,var(--accent),var(--accent-2));
  -webkit-background-clip:text;
  background-clip:text;
  color:transparent;
}
.hero p{color:var(--muted);margin:0;font-size:.98rem;}

.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:.35rem 0 1.25rem;}
.stat{
  background:var(--surface);
  border:1px solid var(--line);
  border-radius:14px;
  padding:14px 18px;
}
.stat-v{font-size:1.55rem;font-weight:700;color:var(--text);}
.stat-k{
  color:var(--muted);
  font:500 .7rem 'JetBrains Mono',monospace;
  letter-spacing:.12em;
  text-transform:uppercase;
  margin-top:2px;
}
.login-kicker{
  font:600 .72rem 'JetBrains Mono',monospace;
  letter-spacing:.2em;
  color:var(--muted);
  margin:.4rem 0 .6rem;
  text-align:center;
}

.chat-header{
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:4px 0 10px;
  border-bottom:1px solid var(--line);
  margin-bottom:14px;
}
.chat-title{
  font-size:1.25rem;
  font-weight:600;
  color:var(--text);
}
.chat-sub{
  font-size:.82rem;
  color:var(--muted);
}
.gemini-welcome{
  text-align:center;
  padding:2.6rem 1rem 1.8rem;
}
.gemini-welcome h2{
  font-size:2rem;
  font-weight:700;
  margin:0 0 .35rem;
  background:linear-gradient(90deg,var(--accent),var(--accent-2));
  -webkit-background-clip:text;
  background-clip:text;
  color:transparent;
}
.gemini-welcome p{
  color:var(--muted);
  font-size:.96rem;
  margin:0;
}

.stTabs [data-baseweb="tab-list"]{
  gap:6px;
  background:var(--surface);
  border:1px solid var(--line);
  border-radius:12px;
  padding:4px;
}
.stTabs [data-baseweb="tab"]{
  height:38px;
  border-radius:9px;
  padding:0 16px;
  color:var(--muted);
  background:transparent;
  font-weight:500;
}
.stTabs [aria-selected="true"]{
  color:var(--text)!important;
  background:rgba(56,189,248,.14);
  border:1px solid rgba(56,189,248,.28);
}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}

.stButton>button,.stFormSubmitButton>button,button[data-testid^="stBaseButton"]{
  border-radius:10px;
  border:1px solid var(--line);
  background:var(--surface-soft);
  color:var(--text);
  font-weight:500;
  transition:background .15s ease,border-color .15s ease;
}
.stButton>button:hover,.stFormSubmitButton>button:hover,button[data-testid^="stBaseButton"]:hover{
  border-color:rgba(56,189,248,.45);
  background:rgba(56,189,248,.1);
  color:#fff;
}
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"],
.stButton>button[kind="primary"],
.stFormSubmitButton>button[kind="primary"]{
  background:linear-gradient(135deg,#38BDF8,#6366F1);
  border:0;
  color:#07111F;
  font-weight:600;
}
button[data-testid="stBaseButton-primary"]:hover,
.stButton>button[kind="primary"]:hover{
  opacity:.95;
  color:#07111F;
}

[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div{
  background:var(--surface)!important;
  border-radius:12px!important;
  border:1px solid var(--line)!important;
}
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea{background:transparent!important;}
[data-baseweb="input"]:focus-within,[data-baseweb="textarea"]:focus-within,[data-baseweb="select"]>div:focus-within{
  border-color:var(--accent)!important;
}

[data-testid="stForm"],[data-testid="stExpander"]{
  background:var(--surface);
  border:1px solid var(--line)!important;
  border-radius:14px!important;
}
[data-testid="stExpander"] details{border:0;}

[data-testid="stChatMessage"]{
  background:transparent;
  border:0;
  padding:10px 4px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  background:var(--surface);
  border:1px solid var(--line);
  border-radius:18px;
  padding:14px 18px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){
  background:rgba(22,31,48,.55);
  border:1px solid rgba(148,163,184,.1);
  border-radius:18px;
  padding:14px 18px;
}
[data-testid="stChatInput"]{
  background:var(--surface)!important;
  border:1px solid var(--line)!important;
  border-radius:16px!important;
}

[data-testid="stFileUploaderDropzone"]{
  border:1.5px dashed rgba(56,189,248,.38)!important;
  background:rgba(56,189,248,.04)!important;
  border-radius:14px!important;
}
[data-testid="stAlert"]{border-radius:12px;border:1px solid var(--line);}
[data-testid="stTable"] th{color:var(--accent);}

[data-testid="stSidebar"]{
  min-width:270px;
  background:#111827;
  border-right:1px solid var(--line);
}
.sidebar-section{
  font:600 .68rem 'JetBrains Mono',monospace;
  letter-spacing:.14em;
  color:var(--muted);
  text-transform:uppercase;
  margin:14px 0 8px;
}
.profile{
  display:flex;
  align-items:center;
  gap:11px;
  padding:12px;
  border:1px solid var(--line);
  border-radius:12px;
  background:var(--surface);
  margin-bottom:10px;
}
.avatar{
  width:38px;
  height:38px;
  border-radius:10px;
  display:grid;
  place-items:center;
  font-weight:700;
  color:#07111F;
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
}
.pname{font-weight:600;font-size:.95rem;margin-bottom:2px;color:var(--text);}
.badge{
  font:600 .64rem 'JetBrains Mono',monospace;
  letter-spacing:.1em;
  padding:2px 7px;
  border-radius:999px;
  border:1px solid var(--line);
  color:var(--muted);
}
.badge.admin{
  color:#FBCFE8;
  border-color:rgba(244,114,182,.35);
  background:rgba(244,114,182,.12);
}
.status{
  font:500 .74rem 'JetBrains Mono',monospace;
  margin:4px 2px 12px;
  display:flex;
  align-items:center;
  gap:8px;
}
.status .dot{width:8px;height:8px;border-radius:50%;background:#F87171;}
.status.on{color:#34D399;} .status.on .dot{background:#34D399;} .status.off{color:#F87171;}

.chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;}
.chip{
  font:600 .68rem 'JetBrains Mono',monospace;
  letter-spacing:.05em;
  padding:3px 9px;
  border-radius:999px;
  border:1px solid var(--line);
  color:var(--muted);
}
.chip.cyan{color:var(--accent);border-color:rgba(56,189,248,.35);background:rgba(56,189,248,.08);}
.chip.violet{color:#C7D2FE;border-color:rgba(129,140,248,.35);background:rgba(129,140,248,.1);}
.chip.green{color:#34D399;border-color:rgba(52,211,153,.35);background:rgba(52,211,153,.1);}
::-webkit-scrollbar{width:9px;height:9px;}
::-webkit-scrollbar-thumb{background:rgba(148,163,184,.2);border-radius:8px;}
::-webkit-scrollbar-thumb:hover{background:rgba(56,189,248,.4);}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- small helpers
def esc(x) -> str:
    """Escape anything that goes into our own HTML (names, titles, API text)."""
    return html.escape(str(x), quote=True)


def chip(text, kind="") -> str:
    return f'<span class="chip {kind}">{esc(text)}</span>'


def hero(sub="Search literature, compare findings, or start a new chat in the sidebar to interrogate your PDFs."):
    st.markdown(f'<div class="hero"><div class="kicker">RESEARCH WORKSPACE</div>'
                f'<h1>Research Paper <span>Discovery</span></h1><p>{esc(sub)}</p></div>', unsafe_allow_html=True)


@st.cache_data(ttl=15, show_spinner=False)
def backend_online() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=2).ok
    except requests.RequestException:
        return False


def call(method, path, **kw):
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        r = requests.request(method, f"{API}{path}", headers=headers, timeout=300, **kw)
        if r.status_code == 401:  # backend token expired or invalid
            clear_session_auth()
            st.rerun()
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        try:
            msg = e.response.json().get("detail", e.response.text)
        except ValueError:
            msg = e.response.text
        if e.response.status_code in (400, 403, 429):
            st.warning(str(msg)[:300])
        else:
            st.error(f"{e.response.status_code}: {str(msg)[:300]}")
    except requests.RequestException as e:
        st.error(f"Backend not reachable: {e}")
    return None


# ----------------------------------------------------------------------------- login
def login_form():
    hero()
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown('<div class="login-kicker">SIGN IN TO CONTINUE</div>', unsafe_allow_html=True)
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary"):
                try:
                    r = requests.post(f"{API}/auth/login", json={"username": username, "password": password}, timeout=30)
                except requests.RequestException as e:
                    st.error(f"Backend not reachable: {e}")
                    return
                if r.ok:
                    d = r.json()
                    set_session_auth(d["access_token"], d["username"], d["role"], ttl_hours=SESSION_TTL_HOURS)
                    st.rerun()
                else:
                    try:
                        st.error(str(r.json().get("detail", "Login failed")))
                    except ValueError:
                        st.error("Login failed")


restore_session_from_url()

if "token" not in st.session_state:
    login_form()
    st.stop()

is_admin = st.session_state.role == "admin"
chat_list = call("get", "/chats") or []
titles = {c["id"]: c["title"] for c in chat_list}  # newest first

if st.session_state.get("active_chat") not in titles:
    st.session_state.active_chat = None

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    role = st.session_state.role
    st.markdown(
        f'<div class="profile"><div class="avatar">{esc(st.session_state.username[:1].upper())}</div>'
        f'<div><div class="pname">{esc(st.session_state.username)}</div>'
        f'<span class="badge {"admin" if role == "admin" else ""}">{esc(role.upper())}</span></div></div>',
        unsafe_allow_html=True)
    online = backend_online()
    st.markdown(f'<div class="status {"on" if online else "off"}"><span class="dot"></span>'
                f'Backend {"online" if online else "offline"}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Ask Papers</div>', unsafe_allow_html=True)
    if st.button("＋ New chat", type="primary", use_container_width=True):
        new = call("post", "/chats", json={})
        if new:
            st.session_state.active_chat = new["id"]
            st.rerun()

    for cid, ctitle in titles.items():
        is_current = cid == st.session_state.get("active_chat")
        row_title, row_menu = st.columns([0.82, 0.18])
        label = f"{'▸ ' if is_current else ''}{ctitle[:25]}"
        if row_title.button(label, key=f"open_{cid}", type="primary" if is_current else "secondary", use_container_width=True):
            st.session_state.active_chat = cid
            st.rerun()
        with row_menu.popover("⋮", help="Chat actions"):
            renamed = st.text_input("Rename chat", value=ctitle, key=f"rename_input_{cid}")
            if st.button("Save", key=f"save_btn_{cid}", use_container_width=True) and renamed.strip():
                if call("patch", f"/chats/{cid}", json={"title": renamed.strip()}) is not None:
                    st.rerun()
            if st.button("Delete chat", key=f"del_btn_{cid}", use_container_width=True):
                if call("delete", f"/chats/{cid}") is not None:
                    if st.session_state.get("active_chat") == cid:
                        st.session_state.active_chat = None
                    st.rerun()

    st.divider()
    if st.button("Log out", use_container_width=True):
        clear_session_auth()
        st.rerun()

lib = call("get", "/papers") or []
active_chat_id = st.session_state.get("active_chat")

# =============================================================================
# VIEW 1: DEDICATED GEMINI-STYLE CHAT UI (when a chat is active)
# =============================================================================
if active_chat_id:
    chat_title = titles.get(active_chat_id, "New chat")

    head_left, head_right = st.columns([0.92, 0.08])
    with head_left:
        st.markdown(
            f'<div class="chat-header"><div><div class="chat-title">{esc(chat_title)}</div>'
            f'<div class="chat-sub">Ask questions grounded in your uploaded research papers</div></div></div>',
            unsafe_allow_html=True,
        )
    with head_right:
        if st.button("✕", key="close_chat_ui", help="Back to Research Workspace", use_container_width=True):
            st.session_state.active_chat = None
            st.rerun()

    uploaded = [p for p in lib if p["source"] == "upload"]
    names = {p["paper_id"]: p["title"] for p in uploaded}

    _, center_col, _ = st.columns([0.08, 0.84, 0.08])
    with center_col:
        if not uploaded:
            st.info("No uploaded PDFs yet. Click ✕ at the top right and open Upload PDF to add papers.")
            chosen = []
        else:
            chosen = st.multiselect(
                "📎 Select uploaded papers (leave empty to search all uploaded PDFs)",
                list(names),
                format_func=names.get,
                key=f"papers_{active_chat_id}",
            )

        data = call("get", f"/chats/{active_chat_id}") or {"messages": []}

        if not data["messages"]:
            st.markdown(
                f'<div class="gemini-welcome">'
                f'<h2>Hello, {esc(st.session_state.username)}</h2>'
                f'<p>Select uploaded PDFs above if you want to narrow the scope, then ask anything below.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            for m in data["messages"]:
                with st.chat_message(m["role"], avatar="🧑‍🔬" if m["role"] == "user" else "✨"):
                    st.write(m["content"])
                    if m["sources"]:
                        with st.expander(f"Sources ({len(m['sources'])})"):
                            for src in m["sources"]:
                                st.markdown(f"**[{src['n']}] {src['title']}**")
                                st.caption(src["snippet"])

        prompt = st.chat_input("Ask about your uploaded research papers...")
        if prompt and prompt.strip():
            with st.chat_message("user", avatar="🧑‍🔬"):
                st.write(prompt.strip())
            with st.chat_message("assistant", avatar="✨"):
                with st.spinner("Reading papers and drafting response..."):
                    res = call("post", f"/chats/{active_chat_id}/ask", json={"question": prompt.strip(), "paper_ids": chosen})
            if res:
                st.rerun()

# =============================================================================
# VIEW 2: COMMON RESEARCH WORKSPACE UI (when no chat is open)
# =============================================================================
else:
    hero()
    n_up = sum(1 for p in lib if p["source"] == "upload")
    st.markdown('<div class="stats">'
                f'<div class="stat"><div class="stat-v">{n_up}</div><div class="stat-k">Uploaded papers</div></div>'
                f'<div class="stat"><div class="stat-v">{len(lib) - n_up}</div><div class="stat-k">Web papers</div></div>'
                f'<div class="stat"><div class="stat-v">{esc(st.session_state.role.capitalize())}</div><div class="stat-k">Access level</div></div>'
                '</div>', unsafe_allow_html=True)

    tab_names = ["Search", "Compare"]

    if is_admin:
        tab_names.extend([
            "Vector Search",
            "Upload PDF",
            "Traces",
        ])

    tabs = st.tabs(tab_names)

    tab_search = tabs[0]
    tab_compare = tabs[1]

    tab_vectors = tabs[2] if is_admin else None
    tab_upload = tabs[3] if is_admin else None
    tab_traces = tabs[4] if is_admin else None

    with tab_search:
        q = st.text_input("Ask a research question", placeholder="How can small language models run on phones?")
        use_web = st.checkbox("Also search Google Scholar / arXiv / Semantic Scholar", value=True)
        if st.button("Search", type="primary") and q.strip():
            with st.spinner("Planning query, searching, ranking..."):
                data = call("post", "/search", json={"question": q, "use_web": use_web})
            if data:
                plan = data["plan"]
                st.caption(f"Query: {plan['semantic_query']}  |  Keywords: {', '.join(plan['keywords'])}"
                           + (f"  |  From {plan['year_from']}" if plan["year_from"] else ""))
                if not data["results"]:
                    st.info("No matching papers found. Try different wording or turn on web search.")
                for p in data["results"]:
                    badges = chip(f"score {p['score']}", "cyan")
                    if p["fresh"]:
                        badges += chip("NEW", "green")
                    badges += chip(p.get("source") or "web", "violet")
                    if p.get("citations") is not None:
                        badges += chip(f"{p['citations']} citations")
                    with st.expander(f"{p['title']} ({p['year'] or 'n/a'})"):
                        st.markdown(f'<div class="chips">{badges}</div>', unsafe_allow_html=True)
                        st.write(p["authors"])
                        st.write(p["abstract"])
                        url = p.get("url") or ""
                        if url.startswith(("http://", "https://")):
                            st.markdown(f"[Open paper ↗]({url})")

    with tab_compare:
        labels = {p["paper_id"]: f"{p['title']} ({p['year'] or p['source']})" for p in lib}
        chosen = st.multiselect("Select 2-3 papers", list(labels), format_func=labels.get, max_selections=3)
        focus = st.text_input("Optional focus", placeholder="e.g. memory footprint")
        if st.button("Compare", type="primary") and len(chosen) >= 2:
            with st.spinner("Reading papers and comparing (may take a minute)..."):
                res = call("post", "/compare", json={"paper_ids": chosen, "focus": focus or None})
            if res and "papers" in res:
                df = pd.DataFrame(res["papers"]).set_index("title").drop(columns=["id"], errors="ignore").T
                st.table(df)
                st.info(res.get("summary", ""))
            elif res:
                st.warning(res.get("error", "Unexpected response"))
                st.code(res.get("raw", ""))

    if is_admin and tab_upload is not None:
        with tab_upload:
            flash = st.session_state.pop("flash", None)
            if flash:
                st.success(flash)
            f = st.file_uploader("Drop a research paper (PDF)", type="pdf")
            if f and st.button("Add to library", type="primary"):
                with st.spinner("Extracting and indexing..."):
                    res = call("post", "/upload", files={"file": (f.name, f.getvalue(), "application/pdf")})
                if res:
                    st.session_state.flash = f"Added: {res['title']}"
                    st.rerun()
    if is_admin and tab_vectors is not None:
        with tab_vectors:
            paper_titles = {p["paper_id"]: p["title"] for p in lib}

            stats = call("get", "/vectors/stats")
            if stats:
                st.caption(
                    f"Collection: **{esc(stats['collection'])}**  |  "
                    f"Vectors: **{stats['points_count']}**  |  "
                    f"Dim: **{stats['vector_size']}**  |  "
                    f"Status: **{esc(stats['status'])}**"
                )

            v_query = st.text_input(
                "Vector query (optional — leave blank to list chunks)",
                placeholder="e.g. quantization memory latency (or leave empty to inspect chunks)",
                key="vec_query",
            )

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                v_kind = st.selectbox(
                    "Chunk type",
                    options=[None, "abstract", "fulltext"],
                    format_func=lambda x: "All chunks" if x is None else x,
                    key="vec_kind",
                )
            with c2:
                v_top_k = st.number_input("Max chunks", min_value=1, max_value=200, value=20, step=1, key="vec_top_k")
            with c3:
                use_year = st.checkbox("Filter by year", value=False, key="vec_use_year")
                v_year_from = (
                    st.number_input("Year from", min_value=1900, max_value=2100, value=2022, step=1, key="vec_year_from")
                    if use_year
                    else None
                )

            v_papers = st.multiselect(
                "Select paper(s)",
                options=list(paper_titles),
                format_func=paper_titles.get,
                key="vec_papers",
            )

            if st.button("Fetch / Search vectors", type="primary", key="vec_search_btn"):
                payload = {
                    "query": v_query.strip() if (v_query and v_query.strip()) else None,
                    "top_k": int(v_top_k),
                    "year_from": int(v_year_from) if v_year_from is not None else None,
                    "paper_ids": v_papers,
                    "kind": v_kind,
                }
                with st.spinner("Querying Qdrant..."):
                    res = call("post", "/vectors/search", json=payload)

                if res is not None:
                    hits = res.get("results", [])
                    if not hits:
                        st.info("No matching chunks found.")
                    for idx, h in enumerate(hits, start=1):
                        pid = h.get("paper_id", "")
                        ptitle = paper_titles.get(pid, pid or "Unknown paper")
                        badges = ""
                        if h.get("score") is not None:
                            badges += chip(f"score {round(h['score'], 4)}", "cyan")
                        badges += chip(h.get("kind") or "chunk", "violet")
                        if h.get("year"):
                            badges += chip(str(h["year"]))
                        with st.expander(f"#{idx} — {ptitle}"):
                            st.markdown(f'<div class="chips">{badges}</div>', unsafe_allow_html=True)
                            st.write(h.get("text", ""))
                            if pid:
                                st.caption(f"Paper ID: {pid}")
    if is_admin and tab_traces is not None:
        with tab_traces:
            render_traces(call)    