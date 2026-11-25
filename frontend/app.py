import os

import pandas as pd
import requests
import streamlit as st

import traces_tab

API = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="Research Paper Discovery", layout="wide")
st.title("Research Paper Discovery")

def login_form():
    st.subheader("Sign in")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            try:
                r = requests.post(f"{API}/auth/login", json={"username": username, "password": password}, timeout=30)
            except requests.RequestException as e:
                st.error(f"Backend not reachable: {e}")
                return
            if r.ok:
                d = r.json()
                st.session_state.update(token=d["access_token"], username=d["username"], role=d["role"])
                st.rerun()
            else:
                try:
                    st.error(str(r.json().get("detail", "Login failed")))
                except ValueError:
                    st.error("Login failed")


if "token" not in st.session_state:
    login_form()
    st.stop()

is_admin = st.session_state.role == "admin"
with st.sidebar:
    st.write(f"Signed in as **{st.session_state.username}** ({st.session_state.role})")
    if st.button("Log out"):
        for k in ("token", "username", "role"):
            st.session_state.pop(k, None)
        st.rerun()

def call(method, path, **kw):
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        r = requests.request(method, f"{API}{path}", headers=headers, timeout=300, **kw)
        if r.status_code == 401:  # token expired or invalid
            for k in ("token", "username", "role"):
                st.session_state.pop(k, None)
            st.rerun()
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        if e.response.status_code in (400, 429) and isinstance(detail, str):
            st.warning(detail)  # e.g. stopped by a guard rail
        else:
            st.error(f"{e.response.status_code}: {e.response.text[:300]}")
    except requests.RequestException as e:
        st.error(f"Backend not reachable: {e}")
    return None


tab_names = ["Ask", "Search", "Compare", "Upload PDF"] + (["Traces"] if is_admin else [])
tab_ask, tab_search, tab_compare, tab_upload, *admin_tabs = st.tabs(tab_names)

with tab_ask:
    if st.session_state.pop("drop_active", False):
        st.session_state.pop("active_chat", None)
    chat_list = call("get", "/chats") or []
    titles = {c["id"]: c["title"] for c in chat_list}   # newest first

    if st.button("New chat"):
        new = call("post", "/chats", json={})
        if new:
            st.session_state.active_chat = new["id"]
            st.rerun()

    if not titles:
        st.info("No chats yet. Click New chat to start one.")
    else:
        if st.session_state.get("active_chat") not in titles:
            st.session_state.active_chat = next(iter(titles))
        st.radio("Chats", list(titles), format_func=lambda i: titles[i][:28], horizontal=True,
                 key="active_chat", label_visibility="collapsed")
        chat_id = st.session_state.active_chat

        with st.expander("Chat options"):
            new_title = st.text_input("Rename", value=titles[chat_id], key=f"rename_{chat_id}")
            c1, c2 = st.columns(2)
            if c1.button("Save name", key=f"save_{chat_id}") and new_title.strip():
                if call("patch", f"/chats/{chat_id}", json={"title": new_title.strip()}) is not None:
                    st.rerun()
            if c2.button("Delete this chat", key=f"del_{chat_id}"):
                if call("delete", f"/chats/{chat_id}") is not None:
                    st.session_state.drop_active = True
                    st.rerun()

        uploaded = call("get", "/papers", params={"source": "upload"}) or []
        names = {p["paper_id"]: p["title"] for p in uploaded}
        if not uploaded:
            st.info("No uploaded papers yet. An admin can add PDFs in the Upload tab.")
        chosen = st.multiselect("Limit to these papers (empty = all uploaded)", list(names),
                                format_func=names.get, key=f"papers_{chat_id}")

        data = call("get", f"/chats/{chat_id}") or {"messages": []}
        for m in data["messages"]:
            with st.chat_message(m["role"]):
                st.write(m["content"])
                if m["sources"]:
                    with st.expander(f"Sources ({len(m['sources'])})"):
                        for src in m["sources"]:
                            st.markdown(f"**[{src['n']}] {src['title']}**")
                            st.caption(src["snippet"])

        with st.form(f"ask_{chat_id}", clear_on_submit=True):
            q = st.text_input("Your question", key=f"q_{chat_id}")
            sent = st.form_submit_button("Ask")
        if sent and q.strip():
            with st.spinner("Searching the papers and answering..."):
                res = call("post", f"/chats/{chat_id}/ask", json={"question": q.strip(), "paper_ids": chosen})
            if res:
                st.rerun()

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
            for p in data["results"]:
                with st.expander(f"{p['title']} ({p['year'] or 'n/a'})  -  score {p['score']}"
                                 + ("  [new]" if p["fresh"] else "")):
                    st.write(p["authors"])
                    st.write(p["abstract"])
                    st.markdown(f"[Open paper]({p['url']})" if p["url"] else "")

with tab_upload:
    if not is_admin:
        st.info("Only an admin can upload papers.")
        st.stop()
    f = st.file_uploader("PDF", type="pdf")
    if f and st.button("Add to library"):
        with st.spinner("Extracting and indexing..."):
            res = call("post", "/upload", files={"file": (f.name, f.getvalue(), "application/pdf")})
        if res:
            st.success(f"Added: {res['title']}")

with tab_compare:
    lib = call("get", "/papers") or []
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
            for w in res.get("warnings", []):
                st.warning(w)
        elif res:
            st.warning(res.get("error", "Unexpected response"))

if admin_tabs:
    with admin_tabs[0]:
        traces_tab.render(call)
