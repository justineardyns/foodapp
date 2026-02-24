#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st
import base64
import requests

import os

# ---------- Fixed tag set (click-only) ----------
# key = wat we opslaan in DB, label = wat je ziet
TAG_LIBRARY = [
    {"key": "pasta", "label": "🍝 Pasta"},
    {"key": "rijst", "label": "🍚 Rijst"},
    {"key": "aardappelen", "label": "🥔 Aardappelen"},
    {"key": "noedels", "label": "🍜 Noedels"},
    {"key": "wrap", "label": "🌮 Wrap / Taco"},
    {"key": "brood", "label": "🥖 Brood / Sandwich"},
    {"key": "salade", "label": "🥗 Salade"},
    {"key": "soep", "label": "🍲 Soep"},
    {"key": "couscous", "label": "🥘 Couscous"},
    {"key": "bulgur", "label": "🌾 Bulgur"},
]

APP_DIR = Path(__file__).parent

TAG_IMAGE = {
    "pasta": str(APP_DIR / "pasta.jpg"),
    "rijst": str(APP_DIR / "rijst.jpg"),
    "aardappelen": str(APP_DIR / "aardappelen.jpg"),
    "noedels": str(APP_DIR / "noedels.jpg"),
    "wrap": str(APP_DIR / "wraps.jpg"),
    "brood": str(APP_DIR / "brood.jpg"),
    "salade": str(APP_DIR / "salade.jpg"),
    "soep": str(APP_DIR / "soep.jpg"),
    "couscous": str(APP_DIR / "couscous.jpg"),
    "bulgur": str(APP_DIR / "bulgur.jpg"),
}

TAG_KEY_TO_LABEL = {t["key"]: t["label"] for t in TAG_LIBRARY}
TAG_LABEL_TO_KEY = {t["label"]: t["key"] for t in TAG_LIBRARY}
TAG_WEEK_CAPS = {
    "pasta": 4,     
    "rijst": 2,
    "scampi": 2,
    "comfort": 3,
}

ALL_TAG_LABELS = [t["label"] for t in TAG_LIBRARY]


# Default pantry seed (only used if pantry is empty)
DEFAULT_PANTRY = sorted(set([
    # --- Carbs / basics ---
    "pasta", "rijst", "noedels", "couscous", "quinoa", "bulgur",
    "aardappelen", "zoete aardappel", "brood", "wraps", "tortilla chips",

    # --- Proteins ---
    "kip", "gehakt", "worst", "bacon", "steak", "kalkoen",
    "scampi", "zalm", "tonijn", "kabeljauw", "vissticks",
    "ei", "tofu", "kikkererwten", "linzen", "bonen",

    # --- Dairy / fridge ---
    "room", "kookroom", "crème fraîche", "melk", "boter", "yoghurt",
    "parmezaan", "mozzarella", "feta", "kaas", "cheddar",
    "ricotta", "mascarpone",

    # --- Sauces / liquids ---
    "tomatensaus", "passata", "tomatenpuree",
    "sojasaus", "oestersaus", "teriyakisaus", "sweet chili",
    "mayonaise", "ketchup", "mosterd",
    "bouillon", "groentebouillon", "kippenbouillon",
    "olijfolie", "zonnebloemolie", "sesamolie",
    "azijn", "balsamico", "citroensap",

    # --- Vegetables ---
    "ui", "look", "lente-ui", "sjalot",
    "paprika", "champignons", "spinazie", "broccoli", "bloemkool",
    "courgette", "aubergine", "wortel", "prei", "selder",
    "tomaat", "komkommer", "sla", "rucola",
    "erwten", "boontjes", "maïs", "appelmoes",

    # --- Herbs / spices ---
    "zout", "peper", "paprikapoeder", "chilivlokken", "cayenne",
    "komijn", "koriander", "kerrie", "curry", "garam masala",
    "oregano", "basilicum", "tijm", "rozemarijn", "peterselie",
    "dille", "kaneel",

    # --- Asian-ish essentials ---
    "kokosmelk", "currypasta", "gember", "limoen",
    "sriracha", "pindakaas", "nori", "miso",

    # --- Italian-ish essentials ---
    "pesto", "kappertjes", "olijven",

    # --- Convenience ---
    "diepvriesgroenten", "diepvries spinazie", "diepvries scampi",
    "frietjes", "aardappelkroketten",

    # --- Extras / salad ---
    "citroen", "limoen", "honing",
    "noten", "pijnboompitten", "sesamzaad",
]))

# --- GitHub backup (gratis persistence) ---
GITHUB_REPO = "justineardyns/foodapp"  
GITHUB_BRANCH = "main"
GITHUB_DB_PATH = "data/meals.db"  # waar je db in je repo komt

def _gh_headers():
    token = st.secrets.get("GITHUB_TOKEN", "")
    if not token:
        return None
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

def github_download_db_if_exists(local_path: str):
    """
    Download meals.db uit GitHub (als die bestaat) en zet lokaal.
    """
    headers = _gh_headers()
    if not headers:
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_DB_PATH}"
    r = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH}, timeout=20)

    if r.status_code == 200:
        data = r.json()
        content_b64 = data.get("content", "")
        if content_b64:
            raw = base64.b64decode(content_b64)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(raw)
            return True

    return False

def github_upload_db(local_path: str, commit_message="Update meals.db"):
    """
    Upload lokale meals.db naar GitHub (create/update).
    """
    headers = _gh_headers()
    if not headers:
        return False

    if not os.path.exists(local_path):
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_DB_PATH}"

    # check of file bestaat (sha nodig om te updaten)
    sha = None
    r0 = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH}, timeout=20)
    if r0.status_code == 200:
        sha = r0.json().get("sha")

    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "message": commit_message,
        "content": content_b64,
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=headers, json=payload, timeout=30)
    return r.status_code in (200, 201)

def persist_db(conn=None, reason="autosave"):
    """
    Throttle: niet 10 commits per minuut.
    """
    now = datetime.now().timestamp()
    last = st.session_state.get("_last_db_push_ts", 0.0)
    if now - last < 2.0:  # max 1 push per 2 sec
        return

    ok = github_upload_db(DB_PATH, commit_message=f"Update meals.db ({reason})")
    if ok:
        st.session_state["_last_db_push_ts"] = now


# -------------------- DB --------------------
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn
    


def init_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS recipes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        tags TEXT DEFAULT '',            -- comma separated tag keys, e.g. "pasta,scampi"
        ingredients TEXT DEFAULT '[]',   -- JSON list of {"item": "...", "qty": "", "unit": ""}
        instructions TEXT DEFAULT '',
        url TEXT DEFAULT '',
        notes TEXT DEFAULT '',
        active INTEGER DEFAULT 1
    )""")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS ratings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rated_on TEXT NOT NULL,
        recipe_id INTEGER NOT NULL,
        rating REAL NOT NULL,
        FOREIGN KEY(recipe_id) REFERENCES recipes(id)
    )""")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS meal_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        meal_on TEXT NOT NULL,
        meal_type TEXT NOT NULL,   -- 'home' or 'takeaway'
        recipe_id INTEGER,
        FOREIGN KEY(recipe_id) REFERENCES recipes(id)
    )""")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS pantry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        active INTEGER DEFAULT 1
    )""")
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS week_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        week_start TEXT NOT NULL,         -- YYYY-MM-DD
        created_at TEXT NOT NULL,         -- timestamp
        plan_json TEXT NOT NULL           -- JSON list van jouw plan
    )""")

    conn.commit()


def migrate_db(conn):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(recipes)").fetchall()}
    if "url" not in cols:
        conn.execute("ALTER TABLE recipes ADD COLUMN url TEXT DEFAULT ''")
    if "notes" not in cols:
        conn.execute("ALTER TABLE recipes ADD COLUMN notes TEXT DEFAULT ''")

    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "week_plans" not in tables:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS week_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start TEXT NOT NULL,
            created_at TEXT NOT NULL,
            plan_json TEXT NOT NULL
        )""")
    conn.commit()


def q(conn, sql, params=()):
    cur = conn.execute(sql, params)
    return cur.fetchall()


def exec_(conn, sql, params=(), persist_reason="write"):
    conn.execute(sql, params)
    conn.commit()
    persist_db(reason=persist_reason)


# -------------------- Pantry --------------------

def get_pantry(conn):
    rows = q(conn, "SELECT name FROM pantry WHERE active=1 ORDER BY name")
    return [r[0].strip().lower() for r in rows]

def add_pantry_item(conn, name: str):
    name = (name or "").strip().lower()
    if not name:
        return
    try:
        exec_(conn, "INSERT INTO pantry(name, active) VALUES(?,1)", (name,))
    except sqlite3.IntegrityError:
        exec_(conn, "UPDATE pantry SET active=1 WHERE name=?", (name,))

def sync_pantry(conn):
    existing = set(get_pantry(conn))
    for item in DEFAULT_PANTRY:
        if item.strip().lower() not in existing:
            add_pantry_item(conn, item)



# -------------------- Helpers --------------------
def parse_ingredients(text):
    try:
        data = json.loads(text or "[]")
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def ingredients_to_text(ings):
    return json.dumps(ings, ensure_ascii=False)


def normalize_tag_keys(tags_str: str):
    # tags are saved as comma separated keys
    keys = [t.strip().lower() for t in (tags_str or "").split(",") if t.strip()]
    # keep only known keys (from TAG_LIBRARY)
    keys = [k for k in keys if k in TAG_KEY_TO_LABEL]
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out


def tag_keys_to_str(keys):
    keys = [k for k in (keys or []) if k in TAG_KEY_TO_LABEL]
    return ",".join(keys)


def get_all_recipes(conn):
    rows = q(conn, """
        SELECT id, title, tags, ingredients, instructions, url, notes
        FROM recipes
        WHERE active=1
        ORDER BY title
    """)
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "title": r[1],
            "tags": normalize_tag_keys(r[2] or ""),
            "ingredients": parse_ingredients(r[3]),
            "instructions": r[4] or "",
            "url": r[5] or "",
            "notes": r[6] or "",
        })
    return out


def get_avg_recipe_scores(conn):
    rows = q(conn, """
        SELECT recipe_id, AVG(rating) as avg_rating, COUNT(*) as n
        FROM ratings
        GROUP BY recipe_id
    """)
    return {rid: {"avg": float(avg), "n": int(n)} for (rid, avg, n) in rows}


def last_cooked_dates(conn):
    rows = q(conn, """
        SELECT recipe_id, MAX(meal_on) as last_on
        FROM meal_log
        WHERE meal_type='home' AND recipe_id IS NOT NULL
        GROUP BY recipe_id
    """)
    return {rid: last_on for (rid, last_on) in rows}


def recency_penalty(last_on_str, today=None):
    if not last_on_str:
        return 0.0
    today = today or date.today()
    try:
        d = datetime.strptime(last_on_str, "%Y-%m-%d").date()
        days = (today - d).days
        if days < 3:
            return 2.0
        if days < 7:
            return 1.0
        return 0.0
    except Exception:
        return 0.0


def pick_weighted(candidates, weights):
    if not candidates:
        return None
    total = sum(weights) if weights else 0
    if total <= 0:
        return random.choice(candidates)
    r = random.random() * total
    acc = 0.0
    for c, w in zip(candidates, weights):
        acc += w
        if acc >= r:
            return c
    return candidates[-1]



def generate_weekmenu(conn, start_date: date, takeaway_days: int = 1, min_avg_keep: float = 5.0):
    recipes = get_all_recipes(conn)
    if not recipes:
        return []

    recipe_scores = get_avg_recipe_scores(conn)
    last_dates = last_cooked_dates(conn)

    # ---- Taste model ----
    training = build_training_rows(conn)
    ing_aff, tag_aff, global_mean = compute_affinities(training)

    # ---- Filter low rated recipes ----
    allowed = []
    for r in recipes:
        rs = recipe_scores.get(r["id"])
        if rs and rs["n"] > 0 and rs["avg"] <= float(min_avg_keep):
            continue
        allowed.append(r)

    if not allowed:
        allowed = recipes

    # ---- Takeaway days ----
    takeaway_days = max(0, min(7, int(takeaway_days)))
    idxs = list(range(7))
    random.shuffle(idxs)
    takeaway_set = set(idxs[:takeaway_days])

    plan = []
    used_ids = set()
    tag_counts = defaultdict(int)
    last_day_tags = set()

    for day_i in range(7):
        d = (start_date + timedelta(days=day_i)).isoformat()

        if day_i in takeaway_set:
            plan.append({"date": d, "type": "takeaway", "recipe_id": None})
            last_day_tags = set()
            continue

        candidates = []
        weights = []

        for r in allowed:
            rid = r["id"]
            if rid in used_ids:
                continue

            tags = set(r["tags"])

            # ---- Hard weekly cap ----
            blocked = False
            for t in tags:
                cap = TAG_WEEK_CAPS.get(t, 7)
                if tag_counts[t] >= cap:
                    blocked = True
                    break
            if blocked:
                continue

            # ---- AI score ----
            predicted = predict_recipe_score(r, ing_aff, tag_aff, global_mean)
            real_avg = recipe_scores.get(rid, {}).get("avg")
            real_n = recipe_scores.get(rid, {}).get("n", 0)

            if real_n > 0 and real_avg is not None:
                base_score = 0.7 * real_avg + 0.3 * predicted
            else:
                base_score = predicted if predicted is not None else 6.0

            # ---- Recency penalty ----
            pen = recency_penalty(last_dates.get(rid), today=start_date)
            w = max(0.1, base_score - pen)

            # ---- Avoid similar consecutive days ----
            overlap = len(tags & last_day_tags)
            if overlap > 0:
                w *= (0.65 ** overlap)

            # ---- Soft cap penalty ----
            for t in tags:
                cap = TAG_WEEK_CAPS.get(t, 7)
                if cap > 0:
                    ratio = tag_counts[t] / cap
                    w *= max(0.3, 1.0 - 0.6 * ratio)

            candidates.append(r)
            weights.append(w)

        if not candidates:
            plan.append({"date": d, "type": "takeaway", "recipe_id": None})
            last_day_tags = set()
            continue

        chosen = pick_weighted(candidates, weights)

        used_ids.add(chosen["id"])

        chosen_tags = set(chosen["tags"])
        for t in chosen_tags:
            tag_counts[t] += 1

        last_day_tags = chosen_tags

        plan.append({"date": d, "type": "home", "recipe_id": chosen["id"]})

    return plan



def replace_plan_entry(plan, idx: int, new_type: str, new_recipe_id: int | None):
    if not plan or idx < 0 or idx >= len(plan):
        return plan
    plan2 = list(plan)
    e = dict(plan2[idx])
    e["type"] = new_type
    e["recipe_id"] = new_recipe_id if new_type == "home" else None
    plan2[idx] = e
    return plan2

def get_recipe_rating_summary(conn, recipe_id: int):
    row = q(conn, "SELECT AVG(rating), COUNT(*) FROM ratings WHERE recipe_id=?", (recipe_id,))
    avg, n = row[0] if row else (None, 0)

    last = q(conn, """
        SELECT rated_on, rating
        FROM ratings
        WHERE recipe_id=?
        ORDER BY rated_on DESC, id DESC
        LIMIT 5
    """, (recipe_id,))

    return {
        "avg": float(avg) if avg is not None else None,
        "n": int(n),
        "last": [(d, float(r)) for d, r in last],
    }

def save_week_plan(conn, week_start: date, plan: list):
    exec_(
        conn,
        "INSERT INTO week_plans(week_start, created_at, plan_json) VALUES(?,?,?)",
        (week_start.isoformat(), datetime.now().isoformat(timespec="seconds"), json.dumps(plan, ensure_ascii=False)),
    )

def load_latest_week_plan(conn):
    rows = q(conn, """
        SELECT week_start, plan_json
        FROM week_plans
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT 1
    """)
    if not rows:
        return None
    week_start, plan_json = rows[0]
    try:
        plan = json.loads(plan_json)
        # basic sanity check
        if isinstance(plan, list) and plan:
            return {"week_start": week_start, "plan": plan}
    except Exception:
        return None
    return None

def overwrite_latest_week_plan(conn, week_start: str, plan: list):
    # update most recent record for same week_start; if not found insert new
    row = q(conn, """
        SELECT id FROM week_plans
        WHERE week_start=?
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT 1
    """, (week_start,))
    if row:
        plan_id = row[0][0]
        exec_(conn, "UPDATE week_plans SET plan_json=? WHERE id=?",
              (json.dumps(plan, ensure_ascii=False), plan_id))
    else:
        # if none exists, save fresh
        exec_(conn, "INSERT INTO week_plans(week_start, created_at, plan_json) VALUES(?,?,?)",
              (week_start, datetime.now().isoformat(timespec="seconds"), json.dumps(plan, ensure_ascii=False)))
        
def get_top_cooked_recipes(conn, limit=5):
    rows = q(conn, """
        SELECT recipe_id, COUNT(*) as n
        FROM meal_log
        WHERE meal_type='home' AND recipe_id IS NOT NULL
        GROUP BY recipe_id
        ORDER BY n DESC
        LIMIT ?
    """, (int(limit),))
    return [(int(rid), int(n)) for rid, n in rows]

def get_top_rated_recipes(conn, limit=5, min_n=2):
    rows = q(conn, """
        SELECT recipe_id, AVG(rating) as avg, COUNT(*) as n
        FROM ratings
        GROUP BY recipe_id
        HAVING n >= ?
        ORDER BY avg DESC
        LIMIT ?
    """, (int(min_n), int(limit)))
    return [(int(rid), float(avg), int(n)) for rid, avg, n in rows]

def get_calendar_events(conn, start_date: date, end_date: date):
    """
    Returns dict:
      {
        "YYYY-MM-DD": {
            "type": "home"/"takeaway",
            "recipe_id": int|None,
            "title": str,
            "score": float|None
        }, ...
      }
    Neemt de laatste gelogde entry per dag (als je meerdere logs op 1 dag hebt).
    """
    rows = q(conn, """
        SELECT ml.meal_on, ml.meal_type, ml.recipe_id,
               (SELECT r.title FROM recipes r WHERE r.id = ml.recipe_id) as title,
               (SELECT rt.rating
                FROM ratings rt
                WHERE rt.recipe_id = ml.recipe_id AND rt.rated_on = ml.meal_on
                ORDER BY rt.id DESC
                LIMIT 1) as score,
               ml.id
        FROM meal_log ml
        WHERE ml.meal_on BETWEEN ? AND ?
        ORDER BY ml.meal_on ASC, ml.id ASC
    """, (start_date.isoformat(), end_date.isoformat()))

    # laatste per dag wint
    out = {}
    for meal_on, meal_type, recipe_id, title, score, _id in rows:
        out[meal_on] = {
            "type": meal_type,
            "recipe_id": int(recipe_id) if recipe_id is not None else None,
            "title": title or ("Afhaal" if meal_type == "takeaway" else "—"),
            "score": float(score) if score is not None else None,
        }
    return out


# -------------------- AI / Taste Analysis --------------------

def build_training_rows(conn):
    """
    Returns rows:
    [
        {
            "recipe_id": int,
            "rating": float,
            "ingredients": [...],
            "tags": [...]
        }
    ]
    """
    recipes = get_all_recipes(conn)
    by_id = {r["id"]: r for r in recipes}

    rows = q(conn, """
        SELECT recipe_id, rating
        FROM ratings
    """)

    training = []

    for rid, rating in rows:
        recipe = by_id.get(rid)
        if not recipe:
            continue

        training.append({
            "recipe_id": rid,
            "rating": float(rating),
            "ingredients": [ing["item"] for ing in recipe.get("ingredients", [])],
            "tags": recipe.get("tags", [])
        })

    return training


def compute_affinities(training_rows, alpha=5):
    """
    Ingredient + tag affinities with smoothing.
    """
    if not training_rows:
        return {}, {}, None

    global_mean = sum(r["rating"] for r in training_rows) / len(training_rows)

    ingredient_scores = defaultdict(list)
    tag_scores = defaultdict(list)

    for row in training_rows:
        for ing in row["ingredients"]:
            ingredient_scores[ing].append(row["rating"])
        for tag in row["tags"]:
            tag_scores[tag].append(row["rating"])

    def smooth(avg_list):
        n = len(avg_list)
        if n == 0:
            return None
        raw_avg = sum(avg_list) / n
        return (sum(avg_list) + alpha * global_mean) / (n + alpha)

    ing_aff = {k: smooth(v) for k, v in ingredient_scores.items()}
    tag_aff = {k: smooth(v) for k, v in tag_scores.items()}

    return ing_aff, tag_aff, global_mean


def predict_recipe_score(recipe, ing_aff, tag_aff, global_mean):
    """
    Predict score based on ingredient + tag affinity.
    """
    if global_mean is None:
        return None

    ing_scores = [
        ing_aff.get(ing["item"])
        for ing in recipe.get("ingredients", [])
        if ing_aff.get(ing["item"]) is not None
    ]

    tag_scores = [
        tag_aff.get(tag)
        for tag in recipe.get("tags", [])
        if tag_aff.get(tag) is not None
    ]

    ing_part = sum(ing_scores)/len(ing_scores) if ing_scores else global_mean
    tag_part = sum(tag_scores)/len(tag_scores) if tag_scores else global_mean

    return 0.6 * tag_part + 0.4 * ing_part


# -------------------- UI --------------------
st.set_page_config(page_title="Mijn Weekmenu", layout="wide")

st.markdown(
    """
    <h1 style='margin-bottom: 0.2rem;'>Mijn Weekmenu</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
      /* Page background */
      .stApp {
        background: linear-gradient(180deg, #fff7f2 0%, #ffffff 40%);
      }

      /* Tabs: zachtere highlight */
      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
      }
      .stTabs [aria-selected="true"] {
        color: #d14b3b !important;
      }
      .stTabs [data-baseweb="tab-highlight"] {
        background-color: #d14b3b !important;
      }

      /* Buttons */
      .stButton>button {
        border-radius: 14px;
        border: 1px solid #f2c9c2;
        background: #ffe7e2;
        color: #6b1f16;
      }
      .stButton>button:hover {
        border-color: #e7a39a;
        background: #ffd9d2;
      }

      /* Headings iets zachter */
      h1, h2, h3 {
        letter-spacing: -0.2px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Restore DB from GitHub on cold start ---
if "db_restored" not in st.session_state:
    github_download_db_if_exists(DB_PATH)
    st.session_state["db_restored"] = True
conn = get_conn()
init_db(conn)
migrate_db(conn)
sync_pantry(conn)

# --- Auto-load last saved week plan into session_state ---
if "week_plan" not in st.session_state:
    latest = load_latest_week_plan(conn)
    if latest:
        st.session_state["week_plan"] = latest["plan"]
        st.session_state["week_start"] = latest["week_start"]

tabs = st.tabs(["Vandaag", "Week", "Recepten", "Analyse", "Kalender"])
weekday_short = ["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"]


# -------------------- Tab 1: Vandaag --------------------
with tabs[0]:
    st.subheader("Vandaag")

    recipes = get_all_recipes(conn)
    by_id = {r["id"]: r for r in recipes}

    today = date.today()
    today_str = today.isoformat()
    weekday_short = ["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"]

    st.caption(f"{weekday_short[today.weekday()]} {today.day}/{today.month}")

    if "week_plan" not in st.session_state or not st.session_state.get("week_plan"):
        latest = load_latest_week_plan(conn)
        if latest:
            st.session_state["week_plan"] = latest["plan"]
            st.session_state["week_start"] = latest["week_start"]
            
    plan = st.session_state.get("week_plan", [])
    today_entry = next((e for e in plan if e.get("date") == today_str), None)

    if not today_entry:
        st.info("Geen weekmenu geladen. Ga naar tab 2 en genereer je week.")
    else:
        # --- takeaway day ---
        if today_entry.get("type") == "takeaway":
            st.write("Afhaal")

            with st.form("today_takeaway_form"):
                submit_tw = st.form_submit_button("Log", type="primary", use_container_width=True)

            if submit_tw:
                exec_(conn, "INSERT INTO meal_log(meal_on, meal_type, recipe_id) VALUES(?,?,?)",
                      (today_str, "takeaway", None))
                st.success("Geloggd.")
                st.rerun()

                # --- home day ---
        else:
            rid = today_entry.get("recipe_id")
            r = by_id.get(rid)

            if not r:
                st.warning("Recept niet gevonden (misschien verwijderd). Pas je weekmenu aan in tab 2.")
            else:
                # Title (groter)
                st.markdown(f"## {r['title']}")

                # Foto: pak de "eerste" tag die we kennen (maar toon pas op het einde)
                img_path = None
                for k in (r.get("tags") or []):
                    if k in TAG_IMAGE:
                        img_path = TAG_IMAGE[k]
                        break

                # tags (read-only)
                if r.get("tags"):
                    tag_line = " ".join([TAG_KEY_TO_LABEL.get(k, k) for k in r["tags"]])
                    if tag_line:
                        st.caption(tag_line)

                # url or instructions (read-only)
                if r.get("url"):
                    st.markdown(f"[open]({r['url']})")
                elif r.get("instructions"):
                    st.write(r["instructions"])

                # ingredients (read-only)
                if r.get("ingredients"):
                    st.caption("Ingrediënten")
                    for ing in r["ingredients"]:
                        item = (ing.get("item") or "").strip()
                        qty = (ing.get("qty") or "").strip()
                        unit = (ing.get("unit") or "").strip()
                        line = f"- {item}" + (f" {qty} {unit}".strip() if (qty or unit) else "")
                        st.write(line)

                st.divider()

                # score + log (no choice)
                score = st.slider("Score", 1.0, 10.0, 8.0, 0.5)

                with st.form("today_home_form"):
                    submit_home = st.form_submit_button(
                        "Welke score geef je aan dit recept?",
                        type="primary",
                        use_container_width=True
                    )

                if submit_home:
                    exec_(conn, "INSERT INTO meal_log(meal_on, meal_type, recipe_id) VALUES(?,?,?)",
                          (today_str, "home", rid))
                    exec_(conn, "INSERT INTO ratings(rated_on, recipe_id, rating) VALUES(?,?,?)",
                          (today_str, rid, float(score)))
                    st.success("Opgeslagen.")
                    st.rerun()

                # Foto helemaal onderaan
                if img_path:
                    p = Path(img_path)
                    if p.exists():
                        st.divider()
                        st.image(str(p), use_container_width=True)
                    else:
                        st.caption(f"⚠️ Afbeelding niet gevonden: {p.name}")


# -------------------- Tab 2: Week --------------------
with tabs[1]:
    st.subheader("Week")

    recipes = get_all_recipes(conn)
    by_id = {r["id"]: r for r in recipes}
    titles = [r["title"] for r in recipes]
    by_title = {r["title"]: r for r in recipes}

    with st.form("week_form"):
        c1, c2 = st.columns([1.2, 1])
        with c1:
            start = st.date_input("Start", value=date.today())
        with c2:
            takeaway_days = st.slider("Afhaal (dagen)", 0, 7, 0)
    
        submit_week = st.form_submit_button("Genereer", type="primary", use_container_width=True)
    
    if submit_week:
        st.session_state["week_plan"] = generate_weekmenu(conn, start_date=start, takeaway_days=takeaway_days, min_avg_keep=5.0)
        st.session_state["week_start"] = start.isoformat()   # <-- toevoegen
        save_week_plan(conn, week_start=start, plan=st.session_state["week_plan"])  # <-- juist
        st.rerun()

    plan = st.session_state.get("week_plan", [])
    if not plan:
        st.caption("Genereer je weekmenu.")
    else:
        st.divider()
        for idx, e in enumerate(plan):
            d = datetime.strptime(e["date"], "%Y-%m-%d").date()
            label = f"{weekday_short[d.weekday()]} {d.day}/{d.month}"

            left, right = st.columns([1, 2])
            with left:
                st.write(label)

            with right:
                if e["type"] == "takeaway":
                    if titles:
                        choice = st.selectbox(
                            "recept",
                            options=["Afhaal"] + titles,
                            index=0,
                            key=f"wk_{idx}",
                            label_visibility="collapsed",
                        )
                        if choice != "Afhaal":
                            new_plan = replace_plan_entry(plan, idx, "home", by_title[choice]["id"])
                            st.session_state["week_plan"] = new_plan
                            
                            week_start_str = st.session_state.get("week_start")
                            if week_start_str:
                                overwrite_latest_week_plan(conn, week_start_str, new_plan)
                            
                            st.rerun()
                    else:
                        st.write("Afhaal")
                else:
                    rid = e.get("recipe_id")
                    current_title = by_id.get(rid, {}).get("title", "—")
                    if titles:
                        choice = st.selectbox(
                            "recept",
                            options=titles + ["Afhaal"],
                            index=(titles.index(current_title) if current_title in titles else 0),
                            key=f"wk_{idx}",
                            label_visibility="collapsed",
                        )
                        if choice == "Afhaal":
                            new_plan = replace_plan_entry(plan, idx, "takeaway", None)
                            st.session_state["week_plan"] = new_plan
                        
                            week_start_str = st.session_state.get("week_start")
                            if week_start_str:
                                overwrite_latest_week_plan(conn, week_start_str, new_plan)
                        
                            st.rerun()
                        else:
                            new_id = by_title[choice]["id"]
                            if new_id != rid:
                                new_plan = replace_plan_entry(plan, idx, "home", new_id)
                                st.session_state["week_plan"] = new_plan
                        
                                week_start_str = st.session_state.get("week_start")
                                if week_start_str:
                                    overwrite_latest_week_plan(conn, week_start_str, new_plan)
                        
                                st.rerun()
                    else:
                        st.write(current_title)


# -------------------- Tab 3: Recepten --------------------
with tabs[2]:
    st.subheader("Recepten")

    recipes = get_all_recipes(conn)
    by_title = {r["title"]: r for r in recipes}
    titles = sorted(by_title.keys())

    # --- 1) Zoek / laad recept ---
    choice = st.selectbox("Zoek", options=["Nieuw"] + titles, index=0)

    loaded = None if choice == "Nieuw" else by_title[choice]
    
    if loaded:
        s = get_recipe_rating_summary(conn, loaded["id"])
        avg_txt = f"{s['avg']:.1f}/10" if s["avg"] is not None else "—"
        st.caption(f"Score: {avg_txt}  •  {s['n']}x")
    
        if s["last"]:
            last_line = "  ".join([f"{d}: {r:.1f}" for d, r in s["last"]])
            st.caption(f"Laatste: {last_line}")


    # Defaults voor form velden
    default_title = "" if not loaded else loaded["title"]

    default_tag_labels = []
    if loaded:
        default_tag_labels = [TAG_KEY_TO_LABEL[t] for t in loaded["tags"] if t in TAG_KEY_TO_LABEL]

    pantry = get_pantry(conn)
    default_ings = []
    if loaded:
        # neem enkel ingrediënten die in pantry staan (anders kunnen ze niet geselecteerd worden)
        default_ings = [ing.get("item", "") for ing in (loaded.get("ingredients") or [])]
        default_ings = [x for x in default_ings if x in pantry]

    default_text = ""
    if loaded:
        # Als er een url is, toon die in het tekstveld, anders instructions
        default_text = loaded["url"] if loaded.get("url") else (loaded.get("instructions") or "")

    # --- 2) Form: add/update ---
    with st.form("recipe_form"):
        title = st.text_input("Naam", value=default_title)

        tag_labels = st.multiselect("Tags", options=ALL_TAG_LABELS, default=default_tag_labels)
        tag_keys = [TAG_LABEL_TO_KEY[l] for l in tag_labels if l in TAG_LABEL_TO_KEY]

        selected_ings = st.multiselect("Ingrediënten", options=pantry, default=default_ings)
        ingredients = [{"item": it, "qty": "", "unit": ""} for it in selected_ings]

        text = st.text_area("Bereiding / URL", value=default_text, height=140)

        c1, c2 = st.columns([1, 1])
        with c1:
            save = st.form_submit_button("Opslaan", type="primary", use_container_width=True)
        with c2:
            delete = st.form_submit_button("Verwijder", use_container_width=True)

    # --- 3) Actions ---
    if save:
        if not title.strip():
            st.error("Naam ontbreekt.")
        else:
            t = (text or "").strip()
            is_url = t.lower().startswith("http://") or t.lower().startswith("https://")
            url = t if is_url else ""
            instructions = "" if is_url else t

            if loaded:
                # UPDATE bestaand recept
                exec_(
                    conn,
                    "UPDATE recipes SET title=?, tags=?, ingredients=?, instructions=?, url=? WHERE id=?",
                    (
                        title.strip(),
                        tag_keys_to_str(tag_keys),
                        ingredients_to_text(ingredients),
                        instructions,
                        url,
                        loaded["id"],
                    ),
                )
            else:
                # INSERT nieuw recept
                exec_(
                    conn,
                    "INSERT INTO recipes(title, tags, ingredients, instructions, url, notes) VALUES(?,?,?,?,?,?)",
                    (
                        title.strip(),
                        tag_keys_to_str(tag_keys),
                        ingredients_to_text(ingredients),
                        instructions,
                        url,
                        "",
                    ),
                )

            st.success("Opgeslagen.")
            st.rerun()

    if delete:
        if not loaded:
            st.info("Kies eerst een bestaand recept om te verwijderen.")
        else:
            exec_(conn, "UPDATE recipes SET active=0 WHERE id=?", (loaded["id"],))
            st.success("Verwijderd.")
            st.rerun()

# -------------------- Tab 4: Analyse --------------------
with tabs[3]:
    st.header("Smaakprofiel")

    recipes = get_all_recipes(conn)
    by_id = {r["id"]: r for r in recipes}

    training = build_training_rows(conn)
    ing_aff, tag_aff, global_mean = compute_affinities(training)

    if not training:
        st.info("Nog geen scores om analyse te maken.")
    else:

        st.markdown("### Algemeen")

        st.write(f"Gemiddelde score: **{round(global_mean,2)}/10**")
        st.write(f"Aantal beoordelingen: **{len(training)}**")

        # --- Top ingrediënten ---
        st.markdown("### Beste ingrediënten")

        sorted_ings = sorted(
            ing_aff.items(),
            key=lambda x: x[1] if x[1] is not None else 0,
            reverse=True
        )[:5]

        for ing, score in sorted_ings:
            st.write(f"{ing} → {round(score,2)}/10")
            
            # --- Top 5 meest gemaakt ---
        st.markdown("### Meest gemaakt (thuis)")
    
        top_cooked = get_top_cooked_recipes(conn, limit=5)
        if not top_cooked:
            st.write("Nog geen thuis-maaltijden gelogd.")
        else:
            for rid, n in top_cooked:
                title = by_id.get(rid, {}).get("title", f"Recept {rid}")
                st.write(f"{title} → {n}x")
    
        # --- Top 5 best gescoord (optioneel maar leuk) ---
        st.markdown("### Best gescoord")
    
        top_rated = get_top_rated_recipes(conn, limit=5, min_n=2)
        if not top_rated:
            st.write("Nog te weinig scores (min. 2 per recept).")
        else:
            for rid, avg, n in top_rated:
                title = by_id.get(rid, {}).get("title", f"Recept {rid}")
                st.write(f"{title} → {avg:.1f}/10 ({n}x)")
                
# -------------------- Tab 5: Kalender --------------------
with tabs[4]:
    st.subheader("Kalender")

    recipes = get_all_recipes(conn)
    by_id = {r["id"]: r for r in recipes}

    # Kies maand
    today = date.today()
    month_pick = st.date_input("Kies datum", value=today, key="cal_month")

    month_start = month_pick.replace(day=1)
    next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
    month_end = next_month - timedelta(days=1)

    events = get_calendar_events(conn, month_start, month_end)

    st.caption(f"{month_start.strftime('%B %Y')}")

    # simpele “calendar grid” (Ma..Zo)
    weekday_labels = ["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"]
    st.write("")

    # start op maandag
    first_wd = month_start.weekday()  # ma=0
    days_in_month = month_end.day

    # grid: 6 weken x 7 dagen
    day_num = 1
    for week in range(6):
        cols = st.columns(7)
        for wd in range(7):
            with cols[wd]:
                if week == 0 and wd < first_wd:
                    st.write("")  # leeg vak
                    continue
                if day_num > days_in_month:
                    st.write("")
                    continue

                d = month_start.replace(day=day_num)
                d_str = d.isoformat()

                # dag header
                is_today = (d == today)
                st.markdown(f"**{day_num}**" + (" ⭐" if is_today else ""))

                ev = events.get(d_str)
                if not ev:
                    st.caption("—")
                else:
                    if ev["type"] == "takeaway":
                        st.caption("Afhaal")
                    else:
                        # titel + score
                        title = ev["title"]
                        score = ev["score"]
                        if score is None:
                            st.caption(f"{title}")
                        else:
                            st.caption(f"{title} • **{score:.1f}/10**")

                # (optioneel) details via expander
                if ev and ev["type"] == "home" and ev["recipe_id"]:
                    with st.expander("Details", expanded=False):
                        r = by_id.get(ev["recipe_id"])
                        if r:
                            if r.get("url"):
                                st.markdown(f"[open]({r['url']})")
                            elif r.get("instructions"):
                                st.write(r["instructions"])

                            if r.get("ingredients"):
                                st.caption("Ingrediënten")
                                for ing in r["ingredients"]:
                                    item = (ing.get("item") or "").strip()
                                    if item:
                                        st.write(f"- {item}")

                day_num += 1

    st.divider()

