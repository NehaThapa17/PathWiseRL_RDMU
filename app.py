"""
PathWise RL — Personalized Learning Path Recommender using Reinforcement Learning
Final Exam Project: Reasoning and Decision Making under Uncertainty
Demonstrates: MDP, Sequential Decision Making, Exploration vs Exploitation, Utility Theory
"""

import os
import sys
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# ─── Absolute paths — works from any working directory ────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PathWise RL",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    required = [
        "concepts.csv", "prerequisites.csv", "students.csv",
        "initial_mastery.csv", "quiz_history.csv", "learning_objectives.csv",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        gen_script = os.path.join(BASE_DIR, "generate_data.py")
        ran = False
        for py_cmd in [sys.executable, "py", "python", "python3"]:
            try:
                subprocess.run(
                    [py_cmd, gen_script],
                    check=True, capture_output=True, cwd=BASE_DIR,
                )
                ran = True
                break
            except Exception:
                continue
        if not ran:
            st.error("❌ Data files missing and could not auto-generate. Run `py generate_data.py` first.")
            st.stop()

    def read(name):
        return pd.read_csv(os.path.join(DATA_DIR, name))

    return (
        read("concepts.csv"),
        read("prerequisites.csv"),
        read("students.csv"),
        read("initial_mastery.csv"),
        read("quiz_history.csv"),
        read("learning_objectives.csv"),
    )

concepts, prerequisites, students, initial_mastery, quiz_history_base, learning_objectives = load_all_data()

# ─── Session State Init ────────────────────────────────────────────────────────
if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = quiz_history_base.copy()
if "mastery_override" not in st.session_state:
    st.session_state.mastery_override = {}
if "q_table" not in st.session_state:
    st.session_state.q_table = {}
if "episode_rewards" not in st.session_state:
    st.session_state.episode_rewards = []
if "policy_trained" not in st.session_state:
    st.session_state.policy_trained = False

# ─── Core Helper Functions ────────────────────────────────────────────────────

def get_mastery(student_id, concept_id):
    """Return current mastery, preferring session-state overrides."""
    key = (student_id, concept_id)
    if key in st.session_state.mastery_override:
        return st.session_state.mastery_override[key]
    row = initial_mastery[
        (initial_mastery["student_id"] == student_id) &
        (initial_mastery["concept_id"] == concept_id)
    ]
    return float(row["initial_mastery"].values[0]) if len(row) else 0.0

def get_mastery_dict(student_id):
    return {cid: get_mastery(student_id, cid) for cid in concepts["concept_id"]}

def get_prereqs(concept_id):
    return prerequisites[prerequisites["target"] == concept_id]["source"].tolist()

def mastery_bin(m):
    return "Low" if m < 0.40 else ("Medium" if m < 0.70 else "High")

def time_bin(t):
    return "Low" if t < 60 else ("Medium" if t <= 180 else "High")

def state_key(concept_id, mastery, remaining_time):
    return f"{concept_id}_{mastery_bin(mastery)}_{time_bin(remaining_time)}"

# ─── Reward Function (Utility Theory) ─────────────────────────────────────────

def compute_reward(concept_id, mastery_dict, student_row, remaining_time):
    """
    Composite reward function demonstrating Utility Theory:
    Balances learning gain, prerequisite readiness, time cost, and difficulty fit.
    """
    c = concepts[concepts["concept_id"] == concept_id].iloc[0]
    cur_m = mastery_dict.get(concept_id, 0.0)

    expected_mastery_gain = c["base_mastery_gain"] * (1 - cur_m)

    prereqs = get_prereqs(concept_id)
    if prereqs:
        prerequisite_score = float(np.mean([mastery_dict.get(p, 0.0) for p in prereqs]))
        missing_prereq_penalty = 1 if any(mastery_dict.get(p, 0.0) < 0.50 for p in prereqs) else 0
    else:
        prerequisite_score = 1.0
        missing_prereq_penalty = 0

    content_match = 1 if c["content_type"] == student_row["learning_style"] else 0
    difficulty_gap = abs(c["difficulty"] - student_row["preferred_difficulty"])

    reward = (
        10.0 * expected_mastery_gain
        + 2.0 * prerequisite_score
        + 1.5 * content_match
        + 0.5 * c["exam_importance"]
        - 0.03 * c["estimated_time_min"]
        - 0.8 * difficulty_gap
        - 3.0 * missing_prereq_penalty
    )
    return float(reward)

# ─── Q-learning Trainer ────────────────────────────────────────────────────────

def train_q_learning(student_id, student_row, obj_concepts, alpha, gamma, epsilon, n_episodes, time_budget):
    """
    Tabular Q-learning for path recommendation.
    State  = (current_concept, mastery_bin, time_bin)
    Action = next_concept_to_study
    """
    q_table = {}
    episode_rewards = []

    for _ in range(n_episodes):
        mastery_d = get_mastery_dict(student_id).copy()
        remaining = float(time_budget)
        visited = set()
        ep_reward = 0.0

        # Start at a random unvisited objective concept
        avail_start = [c for c in obj_concepts if c not in visited]
        if not avail_start:
            break
        current = np.random.choice(avail_start)

        for _step in range(len(obj_concepts)):
            cur_m = mastery_d.get(current, 0.0)
            s = state_key(current, cur_m, remaining)

            # Valid actions: unvisited & fit in remaining time
            valid = [
                c for c in obj_concepts
                if c not in visited
                and concepts[concepts["concept_id"] == c]["estimated_time_min"].values[0] <= remaining
            ]
            if not valid:
                break

            # ε-greedy (Exploration vs Exploitation)
            if np.random.random() < epsilon or not any((s, c) in q_table for c in valid):
                action = np.random.choice(valid)
            else:
                action = max(valid, key=lambda c: q_table.get((s, c), 0.0))

            reward = compute_reward(action, mastery_d, student_row, remaining)

            # Apply action (update mastery & time)
            c_row = concepts[concepts["concept_id"] == action].iloc[0]
            gain = c_row["base_mastery_gain"] * (1 - mastery_d.get(action, 0.0))
            mastery_d[action] = min(1.0, mastery_d.get(action, 0.0) + gain)
            remaining -= c_row["estimated_time_min"]
            visited.add(action)
            ep_reward += reward

            # Next state
            next_m = mastery_d.get(action, 0.0)
            s_next = state_key(action, next_m, max(0.0, remaining))
            next_valid = [
                c for c in obj_concepts
                if c not in visited
                and concepts[concepts["concept_id"] == c]["estimated_time_min"].values[0] <= remaining
            ]
            max_next_q = max((q_table.get((s_next, c), 0.0) for c in next_valid), default=0.0)

            # Q-learning update rule
            old_q = q_table.get((s, action), 0.0)
            q_table[(s, action)] = old_q + alpha * (reward + gamma * max_next_q - old_q)

            current = action
            if remaining <= 0:
                break

        episode_rewards.append(ep_reward)

    return q_table, episode_rewards

# ─── Path Recommender ──────────────────────────────────────────────────────────

def recommend_path(student_id, student_row, obj_concepts, q_table, time_budget, target_mastery):
    """
    Sequential greedy path recommender.
    Combines Q-learning policy with prerequisite order and time constraints.
    Returns 3-7 concepts.
    """
    mastery_d = get_mastery_dict(student_id)
    remaining = float(time_budget)
    visited = set()
    path, rewards = [], []

    # Focus on concepts below target mastery
    candidates = [c for c in obj_concepts if mastery_d.get(c, 0.0) < target_mastery]
    if not candidates:
        candidates = obj_concepts[:]

    # Build topological order from prerequisite graph
    G = nx.DiGraph()
    for _, row in prerequisites.iterrows():
        if row["source"] in obj_concepts and row["target"] in obj_concepts:
            G.add_edge(row["source"], row["target"])
    try:
        topo = [c for c in nx.topological_sort(G) if c in candidates]
    except nx.NetworkXUnfeasible:
        topo = candidates[:]
    extra = [c for c in candidates if c not in topo]
    ordered = topo + extra

    current = ordered[0] if ordered else (candidates[0] if candidates else obj_concepts[0])

    for _ in range(7):
        available = [
            c for c in ordered
            if c not in visited
            and concepts[concepts["concept_id"] == c]["estimated_time_min"].values[0] <= remaining
        ]
        if not available:
            break

        # Rank available concepts using Q-table or fallback reward
        if q_table:
            cur_m = mastery_d.get(current, 0.0)
            s = state_key(current, cur_m, remaining)
            best = max(available, key=lambda c: q_table.get((s, c), compute_reward(c, mastery_d, student_row, remaining)))
        else:
            best = max(available, key=lambda c: compute_reward(c, mastery_d, student_row, remaining))

        r = compute_reward(best, mastery_d, student_row, remaining)
        path.append(best)
        rewards.append(r)
        visited.add(best)

        c_row = concepts[concepts["concept_id"] == best].iloc[0]
        gain = c_row["base_mastery_gain"] * (1 - mastery_d.get(best, 0.0))
        mastery_d[best] = min(1.0, mastery_d.get(best, 0.0) + gain)
        remaining -= c_row["estimated_time_min"]
        current = best

        if len(path) >= 3 and remaining <= 0:
            break

    return path, rewards, mastery_d

def compute_kpis(student_id, path, updated_mastery, time_budget):
    initial_avg = float(np.mean([
        get_mastery(student_id, c) for c in concepts["concept_id"]
    ]))
    final_avg = float(np.mean([
        updated_mastery.get(c, get_mastery(student_id, c)) for c in concepts["concept_id"]
    ]))
    total_time = sum(
        concepts[concepts["concept_id"] == c]["estimated_time_min"].values[0] for c in path
    )
    mastery_rate = float(np.mean([updated_mastery.get(c, 0.0) >= 0.70 for c in path])) if path else 0.0
    learning_efficiency = (final_avg - initial_avg) / total_time * 100 if total_time > 0 else 0.0

    hist = st.session_state.quiz_history
    stud_hist = hist[hist["student_id"] == student_id]
    satisfaction = float(stud_hist["satisfaction_score"].mean()) if len(stud_hist) > 0 else 0.75

    return {
        "learning_efficiency": round(learning_efficiency, 4),
        "mastery_rate": round(mastery_rate, 2),
        "satisfaction": round(satisfaction, 2),
        "total_time": int(total_time),
        "mastery_gain": round(final_avg - initial_avg, 4),
    }

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🧠 PathWise RL")
st.markdown("### Personalized Learning Path Recommender using Reinforcement Learning")
st.info(
    "**Course Concepts Demonstrated:** "
    "🔵 Markov Decision Process &nbsp;|&nbsp; "
    "🟣 Sequential Decision Making &nbsp;|&nbsp; "
    "🟠 Exploration vs Exploitation &nbsp;|&nbsp; "
    "🟢 Utility Theory / Reward Function"
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    student_opts = {
        f"{r['student_name']} ({r['student_id']}) — {r['prior_level']}": r["student_id"]
        for _, r in students.iterrows()
    }
    sel_student_label = st.selectbox("👤 Select Student", list(student_opts.keys()))
    sel_student_id    = student_opts[sel_student_label]
    student_row       = students[students["student_id"] == sel_student_id].iloc[0].copy()

    obj_opts = {
        f"{r['objective_name']} ({r['objective_id']})": r["objective_id"]
        for _, r in learning_objectives.iterrows()
    }
    sel_obj_label = st.selectbox("🎯 Select Learning Objective", list(obj_opts.keys()))
    sel_obj_id    = obj_opts[sel_obj_label]
    obj_row       = learning_objectives[learning_objectives["objective_id"] == sel_obj_id].iloc[0]
    obj_concepts  = obj_row["required_concepts"].split(",")

    st.divider()
    st.subheader("📐 Learning Parameters")

    target_mastery     = st.slider("Target Mastery", 0.60, 0.95, float(student_row["target_mastery"]), 0.05)
    time_budget        = st.slider("⏰ Time Budget (min)", 30, 360, int(student_row["daily_time_budget_min"]), 15)
    pref_difficulty    = st.slider("🎲 Preferred Difficulty", 1, 5, int(student_row["preferred_difficulty"]))
    student_row["preferred_difficulty"] = pref_difficulty

    st.divider()
    st.subheader("🤖 RL Hyperparameters")

    epsilon    = st.slider("Exploration Rate ε", 0.00, 0.50, 0.10, 0.01,
                           help="Probability of choosing a random concept (exploration)")
    alpha      = st.slider("Learning Rate α", 0.05, 0.80, 0.30, 0.05,
                           help="How fast Q-values update")
    gamma      = st.slider("Discount Factor γ", 0.50, 0.99, 0.90, 0.01,
                           help="Weight given to future rewards")
    n_episodes = st.slider("Q-learning Episodes", 50, 1000, 300, 50)

    st.divider()

    train_btn = st.button("🚀 Train / Update Policy", type="primary", use_container_width=True)
    reset_btn = st.button("🔄 Reset Student Mastery", use_container_width=True)

    if train_btn:
        with st.spinner(f"Training {n_episodes} episodes…"):
            q_t, ep_r = train_q_learning(
                sel_student_id, student_row, obj_concepts,
                alpha, gamma, epsilon, n_episodes, time_budget,
            )
            st.session_state.q_table        = q_t
            st.session_state.episode_rewards = ep_r
            st.session_state.policy_trained  = True
        st.success(f"✅ Policy trained — {n_episodes} episodes completed.")

    if reset_btn:
        overrides_to_remove = [k for k in st.session_state.mastery_override if k[0] == sel_student_id]
        for k in overrides_to_remove:
            del st.session_state.mastery_override[k]
        st.session_state.quiz_history = quiz_history_base.copy()
        st.success("↩️ Mastery reset to initial values.")

# ─── Shared Computations ──────────────────────────────────────────────────────
mastery_dict = get_mastery_dict(sel_student_id)
q_table      = st.session_state.q_table

path, path_rewards, projected_mastery = recommend_path(
    sel_student_id, student_row, obj_concepts,
    q_table, time_budget, target_mastery,
)
kpis = compute_kpis(sel_student_id, path, projected_mastery, time_budget)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_ov, tab_gr, tab_rp, tab_rl, tab_au, tab_dt = st.tabs([
    "📊 Overview",
    "🕸️ Concept Graph",
    "🛣️ Recommended Path",
    "🤖 RL Policy",
    "⚡ Adaptive Update",
    "📂 Data",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_ov:
    st.subheader(f"Student Profile: {student_row['student_name']}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Student Details**")
        st.markdown(f"- **ID:** `{sel_student_id}`")
        st.markdown(f"- **Level:** `{student_row['prior_level']}`")
        st.markdown(f"- **Learning Style:** `{student_row['learning_style']}`")
        st.markdown(f"- **Motivation:** `{student_row['motivation_level']:.0%}`")
    with c2:
        st.markdown("**Learning Objective**")
        st.markdown(f"- {obj_row['objective_name']}")
        st.markdown(f"- **Concepts:** `{', '.join(obj_concepts)}`")
        st.markdown(f"- **Recommended Time:** `{obj_row['recommended_time_min']} min`")
    with c3:
        st.markdown("**Mastery Status**")
        obj_avg = float(np.mean([mastery_dict.get(c, 0.0) for c in obj_concepts]))
        st.markdown(f"- **Current Avg Mastery:** `{obj_avg:.0%}`")
        st.markdown(f"- **Target Mastery:** `{target_mastery:.0%}`")
        st.markdown(f"- **Time Budget:** `{time_budget} min`")
        gap = target_mastery - obj_avg
        color = "🔴" if gap > 0.3 else ("🟡" if gap > 0.1 else "🟢")
        st.markdown(f"- **Mastery Gap:** {color} `{gap:.0%}`")

    st.divider()
    st.subheader("📈 Key Performance Indicators")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "🎓 Learning Efficiency",
        f"{kpis['learning_efficiency']:.4f}",
        "mastery gain / 100 min",
    )
    m2.metric(
        "📊 Mastery Rate",
        f"{kpis['mastery_rate']:.0%}",
        f"{int(kpis['mastery_rate'] * len(path))} / {len(path)} concepts ≥ 70%",
    )
    m3.metric(
        "😊 Student Satisfaction",
        f"{kpis['satisfaction']:.2f}",
        "avg quiz satisfaction",
    )
    m4.metric(
        "⏱️ Est. Completion",
        f"{kpis['total_time']} min",
        f"of {time_budget} min budget",
    )

    st.divider()
    st.subheader("Objective Concept Mastery Progress")
    for cid in obj_concepts:
        cname = concepts[concepts["concept_id"] == cid]["concept_name"].values[0]
        m = mastery_dict.get(cid, 0.0)
        icon = "🟢" if m >= 0.70 else ("🟡" if m >= 0.40 else "🔴")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.progress(m, text=f"{icon} {cid} — {cname}")
        with col_b:
            st.markdown(f"**{m:.0%}**")

    # Faculty Explanation (Academic Section)
    st.divider()
    with st.expander("📚 Model Explanation for Faculty", expanded=False):
        st.markdown("""
### Why This is a Markov Decision Process (MDP)

| MDP Component | PathWise RL Definition |
|---|---|
| **State (S)** | Current concept + mastery level bin (Low/Med/High) + remaining time bin (Low/Med/High) |
| **Action (A)** | Next concept to recommend from the learning objective |
| **Reward (R)** | Composite utility: mastery gain + prerequisite readiness − time cost − difficulty mismatch |
| **Transition T(s,a,s')** | After studying concept a, mastery updates and time decreases → new state |
| **Policy π(s)** | Mapping from state → best next concept (learned via Q-learning) |

The **Markov property** holds because the next state depends only on the current concept,
current mastery, and remaining time — not on the full history of studied concepts.

---
### Reward Function (Utility Theory)

```
reward = 10 × expected_mastery_gain
       + 2  × prerequisite_score
       + 1.5 × content_match
       + 0.5 × exam_importance
       - 0.03 × estimated_time_min
       - 0.8  × difficulty_gap
       - 3    × missing_prerequisite_penalty
```

Each coefficient encodes the **relative utility** of each factor to the student's learning outcome.

---
### Q-learning Update Rule

```
Q(s, a) ← Q(s, a) + α × [r + γ × max_a' Q(s', a') − Q(s, a)]
```

- **α (learning rate):** How much new information overrides old estimates.
- **γ (discount factor):** How much future rewards are valued vs immediate rewards.
- Over many episodes, Q-values converge to the true expected cumulative reward.

---
### Exploration vs Exploitation (ε-Greedy)

- With probability **ε**: choose a **random** concept → *explore* unknown paths.
- With probability **1−ε**: choose the concept with the **highest Q-value** → *exploit* learned knowledge.
- Higher ε → more diverse paths discovered; Lower ε → more consistent greedy recommendations.

---
### Sequential Decision Making

Each concept studied changes the state:
1. Mastery level of that concept increases.
2. Remaining time decreases.
3. New concepts become available (prerequisites unlocked).

This chain of decisions — where each action shapes future choices — is the essence of
**Sequential Decision Making under Uncertainty**.

---
### Adaptive Updates

When a student submits a quiz score, mastery is updated:
```
new_mastery = old_mastery + base_gain × (score / 100) × (1 − old_mastery)
if score ≥ 80: new_mastery += 0.05   # bonus for strong performance
if score < 50: new_mastery -= 0.03   # penalty for weak performance
new_mastery = clip(new_mastery, 0, 1)
```
This immediately re-ranks the recommended path, demonstrating the system's **adaptivity**.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONCEPT GRAPH
# ══════════════════════════════════════════════════════════════════════════════
with tab_gr:
    st.subheader("🕸️ Concept Dependency Graph")
    st.markdown(
        "🔴 Low mastery (<40%) &nbsp;|&nbsp; 🟠 Medium mastery (40–70%) &nbsp;|&nbsp; "
        "🟢 High mastery (>70%) &nbsp;|&nbsp; 🔵 Border = in recommended path"
    )

    G = nx.DiGraph()
    for _, row in concepts.iterrows():
        G.add_node(row["concept_id"], **row.to_dict())
    for _, row in prerequisites.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["dependency_strength"])

    # Use shell layout for cleaner look
    pos = nx.spring_layout(G, seed=7, k=2.2)

    # ── Edge traces ──
    reg_ex, reg_ey = [], []
    path_ex, path_ey = [], []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_path_edge = (u in path and v in path and
                        abs(path.index(u) - path.index(v)) == 1)
        if is_path_edge:
            path_ex += [x0, x1, None]
            path_ey += [y0, y1, None]
        else:
            reg_ex += [x0, x1, None]
            reg_ey += [y0, y1, None]

    fig_g = go.Figure()

    fig_g.add_trace(go.Scatter(
        x=reg_ex, y=reg_ey, mode="lines",
        line=dict(width=1.2, color="#cccccc"),
        hoverinfo="none", showlegend=False,
    ))
    if path_ex:
        fig_g.add_trace(go.Scatter(
            x=path_ex, y=path_ey, mode="lines",
            line=dict(width=3.5, color="#1a6fc4"),
            hoverinfo="none", name="Recommended Path Edge",
        ))

    # ── Nodes ──
    node_ids   = list(G.nodes())
    node_x     = [pos[n][0] for n in node_ids]
    node_y     = [pos[n][1] for n in node_ids]
    node_clrs  = []
    node_sizes = []
    node_bclrs = []
    node_bwids = []
    hover_txts = []

    for cid in node_ids:
        row = concepts[concepts["concept_id"] == cid].iloc[0]
        m = mastery_dict.get(cid, 0.0)

        node_clrs.append(
            "#2ecc71" if m >= 0.70 else ("#f39c12" if m >= 0.40 else "#e74c3c")
        )
        node_sizes.append(row["exam_importance"] * 10 + 25)

        if cid in path:
            node_bclrs.append("#1a6fc4")
            node_bwids.append(4)
        elif cid in obj_concepts:
            node_bclrs.append("#8e44ad")
            node_bwids.append(2)
        else:
            node_bclrs.append("#999999")
            node_bwids.append(1)

        in_obj  = "⭐ In Objective" if cid in obj_concepts else ""
        in_path = "🛣️ In Recommended Path" if cid in path else ""
        hover_txts.append(
            f"<b>{cid}: {row['concept_name']}</b><br>"
            f"Skill Area: {row['skill_area']}<br>"
            f"Difficulty: {row['difficulty']}/5<br>"
            f"Time: {row['estimated_time_min']} min<br>"
            f"Content Type: {row['content_type']}<br>"
            f"Mastery: {m:.0%}<br>"
            f"Exam Importance: {row['exam_importance']}/5<br>"
            f"{in_obj} {in_path}"
        )

    fig_g.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_clrs,
            line=dict(color=node_bclrs, width=node_bwids),
            opacity=0.88,
        ),
        text=node_ids,
        textposition="top center",
        hovertext=hover_txts,
        hoverinfo="text",
        name="Concepts",
    ))

    fig_g.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(fig_g, width="stretch")

    with st.expander("📋 Concept Details Table"):
        display_df = concepts.copy()
        display_df["Current Mastery"] = display_df["concept_id"].apply(
            lambda c: f"{mastery_dict.get(c, 0.0):.0%}"
        )
        display_df["In Objective"] = display_df["concept_id"].apply(
            lambda c: "✅" if c in obj_concepts else "—"
        )
        display_df["In Path"] = display_df["concept_id"].apply(
            lambda c: "🛣️" if c in path else "—"
        )
        st.dataframe(display_df, width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RECOMMENDED PATH
# ══════════════════════════════════════════════════════════════════════════════
with tab_rp:
    st.subheader("🛣️ Personalized Recommended Learning Path")

    if not path:
        st.warning(
            "⚠️ No path could be built within the available time budget. "
            "Try increasing the time budget or selecting a smaller objective."
        )
    else:
        # ── Path table ──
        path_rows = []
        cur_m_sim = get_mastery_dict(sel_student_id).copy()
        cum_time  = 0
        for i, cid in enumerate(path):
            c_row = concepts[concepts["concept_id"] == cid].iloc[0]
            m_bef  = cur_m_sim.get(cid, 0.0)
            gain   = c_row["base_mastery_gain"] * (1 - m_bef)
            m_aft  = min(1.0, m_bef + gain)
            cum_time += c_row["estimated_time_min"]

            prereqs = get_prereqs(cid)
            prereqs_ok = all(cur_m_sim.get(p, 0.0) >= 0.50 for p in prereqs) if prereqs else True

            path_rows.append({
                "Step": i + 1,
                "ID": cid,
                "Concept": c_row["concept_name"],
                "Difficulty": f"{c_row['difficulty']}/5",
                "Time (min)": c_row["estimated_time_min"],
                "Cumul. Time": cum_time,
                "Mastery Before": f"{m_bef:.0%}",
                "Mastery After": f"{m_aft:.0%}",
                "Reward": round(path_rewards[i], 2),
                "Prereqs ✓": "✅" if prereqs_ok else "⚠️ Weak",
            })
            cur_m_sim[cid] = m_aft

        path_df = pd.DataFrame(path_rows)
        st.dataframe(path_df, width="stretch", hide_index=True)

        if not all(r["Prereqs ✓"] == "✅" for r in path_rows):
            st.warning(
                "⚠️ Some concepts have prerequisites with mastery below 50%. "
                "The system still recommends them with a penalty applied to the reward."
            )

        # ── Timeline ──
        st.subheader("⏱️ Learning Timeline (Gantt View)")
        timeline_rows = []
        start_t = 0
        for row in path_rows:
            timeline_rows.append({
                "Concept": f"Step {row['Step']}: {row['Concept']}",
                "Start": start_t,
                "Duration": row["Time (min)"],
                "End": start_t + row["Time (min)"],
                "Difficulty": int(row["Difficulty"][0]),
            })
            start_t += row["Time (min)"]

        tl_df = pd.DataFrame(timeline_rows)
        fig_tl = go.Figure()
        colors = {1: "#2ecc71", 2: "#27ae60", 3: "#f39c12", 4: "#e67e22", 5: "#e74c3c"}
        for _, tr in tl_df.iterrows():
            fig_tl.add_trace(go.Bar(
                x=[tr["Duration"]],
                y=[tr["Concept"]],
                base=tr["Start"],
                orientation="h",
                marker_color=colors.get(tr["Difficulty"], "#3498db"),
                name=tr["Concept"],
                hovertemplate=(
                    f"<b>{tr['Concept']}</b><br>"
                    f"Start: {tr['Start']} min<br>"
                    f"Duration: {tr['Duration']} min<br>"
                    f"End: {tr['End']} min<extra></extra>"
                ),
                showlegend=False,
            ))
        fig_tl.update_layout(
            barmode="stack",
            height=250 + len(path) * 30,
            xaxis_title="Minutes",
            title="Study Timeline (color = difficulty: green→red)",
            margin=dict(l=0, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_tl, width="stretch")

        # ── Mastery Progression ──
        st.subheader("📈 Expected Mastery Progression")
        mp_rows = []
        cur_m_sim2 = get_mastery_dict(sel_student_id).copy()
        avg_bef = float(np.mean([cur_m_sim2.get(c, 0.0) for c in obj_concepts]))
        mp_rows.append({"Point": "Start", "Avg Mastery": avg_bef})

        for i, cid in enumerate(path):
            c_row = concepts[concepts["concept_id"] == cid].iloc[0]
            m_bef = cur_m_sim2.get(cid, 0.0)
            gain  = c_row["base_mastery_gain"] * (1 - m_bef)
            cur_m_sim2[cid] = min(1.0, m_bef + gain)
            avg_now = float(np.mean([cur_m_sim2.get(c, 0.0) for c in obj_concepts]))
            mp_rows.append({
                "Point": f"After Step {i+1}\n({c_row['concept_name']})",
                "Avg Mastery": avg_now,
            })

        mp_df = pd.DataFrame(mp_rows)
        fig_mp = px.line(
            mp_df, x="Point", y="Avg Mastery",
            markers=True, title="Average Objective Mastery After Each Step",
            color_discrete_sequence=["#1a6fc4"],
        )
        fig_mp.add_hline(
            y=target_mastery, line_dash="dash", line_color="green",
            annotation_text=f"Target: {target_mastery:.0%}",
        )
        fig_mp.update_layout(yaxis_range=[0, 1.05], height=350)
        st.plotly_chart(fig_mp, width="stretch")

        # ── Reward bar chart ──
        st.subheader("🏆 Q-learning Reward by Concept")
        rw_df = pd.DataFrame({
            "Concept": [
                f"Step {i+1}: {concepts[concepts['concept_id']==c]['concept_name'].values[0]}"
                for i, c in enumerate(path)
            ],
            "Reward": path_rewards,
        })
        fig_rw = px.bar(
            rw_df, x="Concept", y="Reward",
            color="Reward", color_continuous_scale="Blues",
            title="Expected Reward per Concept (higher = better choice)",
            text_auto=".2f",
        )
        fig_rw.update_layout(height=320)
        st.plotly_chart(fig_rw, width="stretch")

        # ── Step Explanations ──
        st.subheader("💡 Why Each Concept Was Selected")
        cur_m_exp = get_mastery_dict(sel_student_id).copy()
        for i, cid in enumerate(path):
            c_row = concepts[concepts["concept_id"] == cid].iloc[0]
            m = cur_m_exp.get(cid, 0.0)
            prereqs = get_prereqs(cid)
            reasons = []
            if m < target_mastery:
                reasons.append(f"Mastery {m:.0%} is below target {target_mastery:.0%}")
            if cid in obj_concepts:
                reasons.append("Required by the selected learning objective")
            if c_row["exam_importance"] >= 5:
                reasons.append("Rated as high exam importance (5/5)")
            if path_rewards[i] > 4:
                reasons.append(f"High Q-learning reward score: {path_rewards[i]:.2f}")
            if c_row["content_type"] == student_row["learning_style"]:
                reasons.append(f"Content type ({c_row['content_type']}) matches student's learning style")
            if not prereqs or all(cur_m_exp.get(p, 0.0) >= 0.50 for p in prereqs):
                reasons.append("Prerequisites sufficiently mastered")
            else:
                reasons.append("⚠️ Some prerequisites have low mastery — penalty applied")

            with st.expander(f"Step {i+1}: {c_row['concept_name']} ({cid}) | Reward: {path_rewards[i]:.2f}"):
                st.markdown(
                    f"**Difficulty:** {c_row['difficulty']}/5 &nbsp;|&nbsp; "
                    f"**Time:** {c_row['estimated_time_min']} min &nbsp;|&nbsp; "
                    f"**Content Type:** {c_row['content_type']} &nbsp;|&nbsp; "
                    f"**Skill Area:** {c_row['skill_area']}"
                )
                for r in reasons:
                    st.markdown(f"- {r}")

            gain = c_row["base_mastery_gain"] * (1 - m)
            cur_m_exp[cid] = min(1.0, m + gain)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RL POLICY
# ══════════════════════════════════════════════════════════════════════════════
with tab_rl:
    st.subheader("🤖 Reinforcement Learning Policy")

    with st.expander("ℹ️ How Q-learning Works in PathWise RL", expanded=False):
        st.markdown("""
**State:** `(concept_id, mastery_bin, time_bin)` — e.g., `C07_Low_Medium`
- Mastery bins: Low (<40%), Medium (40–70%), High (>70%)
- Time bins: Low (<60 min), Medium (60–180 min), High (>180 min)

**Action:** Next concept to study from the selected objective.

**Q-learning update:**
```
Q(s, a) ← Q(s, a) + α × [reward + γ × max_a' Q(s', a') − Q(s, a)]
```

**ε-greedy policy:** With prob. ε → random concept (explore); else → max Q-value concept (exploit).
        """)

    if not st.session_state.policy_trained:
        st.info("👈 Click **Train / Update Policy** in the sidebar to train the Q-learning agent.")
    else:
        episode_rewards = st.session_state.episode_rewards
        q_table_local   = st.session_state.q_table

        # ── Convergence chart ──
        st.subheader("📉 Episode Reward Convergence")
        smooth_window = max(1, len(episode_rewards) // 20)
        ep_df = pd.DataFrame({
            "Episode": range(1, len(episode_rewards) + 1),
            "Total Reward": episode_rewards,
            "Smoothed (avg)": pd.Series(episode_rewards).rolling(smooth_window, min_periods=1).mean(),
        })
        fig_conv = px.line(
            ep_df, x="Episode",
            y=["Total Reward", "Smoothed (avg)"],
            title="Q-learning Episode Rewards (converging = policy stabilizing)",
            color_discrete_map={"Total Reward": "#aec6e8", "Smoothed (avg)": "#1a6fc4"},
        )
        fig_conv.update_layout(height=340, legend=dict(orientation="h"))
        st.plotly_chart(fig_conv, width="stretch")

        # ── Q-value heatmap ──
        st.subheader("🔥 Q-value Heatmap (State × Action)")
        if q_table_local:
            all_states  = sorted(set(k[0] for k in q_table_local))
            all_actions = sorted(set(k[1] for k in q_table_local))

            display_states = all_states[:25]
            q_matrix = [
                [q_table_local.get((s, a), 0.0) for a in all_actions]
                for s in display_states
            ]
            action_labels = [
                concepts[concepts["concept_id"] == a]["concept_name"].values[0]
                if a in concepts["concept_id"].values else a
                for a in all_actions
            ]

            fig_ht = px.imshow(
                q_matrix,
                x=action_labels,
                y=display_states,
                color_continuous_scale="RdYlGn",
                title="Q-values: top 25 states (rows) × actions (columns)",
                labels={"color": "Q-value"},
                aspect="auto",
            )
            fig_ht.update_layout(height=560)
            st.plotly_chart(fig_ht, width="stretch")

        # ── Top actions bar ──
        st.subheader("🏅 Best Concepts by Max Q-value")
        action_best_q = {}
        for (s, a), q in q_table_local.items():
            action_best_q[a] = max(action_best_q.get(a, float("-inf")), q)

        if action_best_q:
            top_df = pd.DataFrame(action_best_q.items(), columns=["Concept ID", "Max Q-value"])
            top_df["Concept Name"] = top_df["Concept ID"].map(
                dict(zip(concepts["concept_id"], concepts["concept_name"]))
            )
            top_df = top_df.sort_values("Max Q-value", ascending=False)

            fig_top = px.bar(
                top_df, x="Concept Name", y="Max Q-value",
                color="Max Q-value", color_continuous_scale="Blues",
                title="Max Q-value per Action Concept (higher = more often recommended)",
                text_auto=".2f",
            )
            fig_top.update_layout(height=340)
            st.plotly_chart(fig_top, width="stretch")

        # ── Policy table ──
        st.subheader("📋 Learned Policy Table")
        state_best = {}
        for (s, a), q in q_table_local.items():
            if s not in state_best or q > state_best[s][1]:
                state_best[s] = (a, q)

        policy_rows = []
        for s, (a, q) in state_best.items():
            parts = s.split("_")
            cid_s = parts[0] if parts else s
            ml    = parts[1] if len(parts) > 1 else "?"
            tl    = parts[2] if len(parts) > 2 else "?"
            a_name = (
                concepts[concepts["concept_id"] == a]["concept_name"].values[0]
                if a in concepts["concept_id"].values else a
            )
            policy_rows.append({
                "State": s,
                "At Concept": cid_s,
                "Mastery Level": ml,
                "Time Level": tl,
                "Best Next → Concept": a_name,
                "Q-value": round(q, 3),
            })

        policy_df = (
            pd.DataFrame(policy_rows)
            .sort_values("Q-value", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(policy_df, width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ADAPTIVE UPDATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_au:
    st.subheader("⚡ Adaptive Learning Update")
    st.markdown(
        "Simulate completing a concept and submitting a quiz. "
        "The system updates mastery and recalculates the recommended path."
    )

    if not path:
        st.warning("No recommended path available. Adjust parameters in the sidebar first.")
    else:
        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Simulate Concept Completion")
            concept_opts = {
                f"{cid}: {concepts[concepts['concept_id']==cid]['concept_name'].values[0]}": cid
                for cid in path
            }
            sel_concept_label = st.selectbox("Select Completed Concept", list(concept_opts.keys()))
            sel_concept_id    = concept_opts[sel_concept_label]
            c_row_upd         = concepts[concepts["concept_id"] == sel_concept_id].iloc[0]

            quiz_score_upd = st.slider("Quiz Score", 0, 100, 75, 5)
            time_spent_upd = st.slider("Actual Time Spent (min)", 5, 120, int(c_row_upd["estimated_time_min"]), 5)
            satisfaction_upd = st.slider("Satisfaction Rating", 0.0, 1.0, 0.80, 0.05)

            old_m = mastery_dict.get(sel_concept_id, 0.0)
            new_m = old_m + c_row_upd["base_mastery_gain"] * (quiz_score_upd / 100) * (1 - old_m)
            if quiz_score_upd >= 80:
                new_m += 0.05
            elif quiz_score_upd < 50:
                new_m -= 0.03
            new_m = float(np.clip(new_m, 0.0, 1.0))

            delta_m = new_m - old_m
            st.markdown("---")
            st.markdown(f"**Before:** `{old_m:.0%}` &nbsp;→&nbsp; **After:** `{new_m:.0%}`")
            st.metric("Mastery Change", f"{new_m:.0%}", f"{delta_m:+.0%}")

            if st.button("✅ Apply Learning Update", type="primary", use_container_width=True):
                # Apply mastery override
                st.session_state.mastery_override[(sel_student_id, sel_concept_id)] = new_m

                # Record to quiz history
                prev_attempts = st.session_state.quiz_history[
                    (st.session_state.quiz_history["student_id"] == sel_student_id) &
                    (st.session_state.quiz_history["concept_id"] == sel_concept_id)
                ]
                new_attempt_no = len(prev_attempts) + 1
                new_entry = pd.DataFrame([{
                    "student_id":       sel_student_id,
                    "concept_id":       sel_concept_id,
                    "attempt_no":       new_attempt_no,
                    "quiz_score":       quiz_score_upd,
                    "time_spent_min":   time_spent_upd,
                    "completed":        1 if quiz_score_upd >= 50 else 0,
                    "satisfaction_score": satisfaction_upd,
                }])
                st.session_state.quiz_history = pd.concat(
                    [st.session_state.quiz_history, new_entry], ignore_index=True
                )
                st.success(f"✅ Mastery updated: {old_m:.0%} → {new_m:.0%} | Path will recalculate.")
                st.rerun()

        with right:
            st.markdown("### Current vs Target Mastery")
            bar_data = []
            for cid in obj_concepts:
                cname = concepts[concepts["concept_id"] == cid]["concept_name"].values[0]
                init_m = initial_mastery[
                    (initial_mastery["student_id"] == sel_student_id) &
                    (initial_mastery["concept_id"] == cid)
                ]["initial_mastery"].values
                init_val = float(init_m[0]) if len(init_m) > 0 else 0.0
                curr_val = mastery_dict.get(cid, 0.0)
                bar_data.append({
                    "Concept": f"{cid}: {cname}",
                    "Initial Mastery": init_val,
                    "Current Mastery": curr_val,
                })

            bar_df = pd.DataFrame(bar_data)
            fig_ba = go.Figure()
            fig_ba.add_trace(go.Bar(
                name="Initial Mastery",
                x=bar_df["Concept"], y=bar_df["Initial Mastery"],
                marker_color="#aec6e8",
            ))
            fig_ba.add_trace(go.Bar(
                name="Current Mastery",
                x=bar_df["Concept"], y=bar_df["Current Mastery"],
                marker_color="#1a6fc4",
            ))
            fig_ba.add_hline(
                y=target_mastery, line_dash="dash", line_color="green",
                annotation_text=f"Target: {target_mastery:.0%}",
            )
            fig_ba.update_layout(
                barmode="group",
                yaxis_range=[0, 1.1],
                height=420,
                title="Initial vs Current Mastery per Objective Concept",
                yaxis_title="Mastery",
                xaxis_title="",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_ba, width="stretch")

        st.divider()
        st.subheader("📜 In-Session Quiz History")
        hist_view = st.session_state.quiz_history[
            st.session_state.quiz_history["student_id"] == sel_student_id
        ].copy()
        hist_view = hist_view.merge(concepts[["concept_id", "concept_name"]], on="concept_id", how="left")
        st.dataframe(hist_view.tail(15), width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_dt:
    st.subheader("📂 All Datasets")

    with st.expander("📚 Concepts (12 rows)", expanded=True):
        st.dataframe(concepts, width="stretch", hide_index=True)

    with st.expander("🔗 Prerequisites (12 edges)"):
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.dataframe(prerequisites, width="stretch", hide_index=True)
        with col_b:
            fig_dep = px.bar(
                prerequisites.merge(
                    concepts[["concept_id", "concept_name"]].rename(columns={"concept_id": "target", "concept_name": "Target Name"}),
                    on="target"
                ),
                x="Target Name", y="dependency_strength",
                color="dependency_strength", color_continuous_scale="Blues",
                title="Dependency Strength by Target Concept",
            )
            st.plotly_chart(fig_dep, width="stretch")

    with st.expander("👥 Students (5 profiles)"):
        st.dataframe(students, width="stretch", hide_index=True)

    with st.expander("🎯 Initial Mastery (60 rows = 5 students × 12 concepts)"):
        im_pivot = initial_mastery.pivot(index="student_id", columns="concept_id", values="initial_mastery")
        st.dataframe(im_pivot.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)

    with st.expander("📝 Quiz History (including in-session updates)"):
        full_hist = st.session_state.quiz_history.merge(
            concepts[["concept_id", "concept_name"]], on="concept_id", how="left"
        )
        st.dataframe(full_hist, width="stretch", hide_index=True)

    with st.expander("🎓 Learning Objectives (5 objectives)"):
        st.dataframe(learning_objectives, width="stretch", hide_index=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "PathWise RL — Final Exam Project | "
    "Reasoning and Decision Making under Uncertainty | "
    "Built with Streamlit, Plotly, NetworkX, NumPy, Pandas"
)
