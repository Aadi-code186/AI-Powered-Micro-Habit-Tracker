# Add these imports at the top of the file
import os
import sqlite3
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import random

import pandas as pd
import altair as alt
import streamlit as st
# Add these new imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from gemini import GeminiAPI

DB_PATH = "habits.db"

# ---------------------------
# DB HELPERS
# ---------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            start_date DATE NOT NULL
        );
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS habit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            habit_id INTEGER NOT NULL,
            log_date DATE NOT NULL,
            ts TIMESTAMP NOT NULL,
            FOREIGN KEY (habit_id) REFERENCES habits(id)
        );
    """)
    conn.commit()


# ---------------------------
# DATA ACCESS
# ---------------------------
def create_habit(name, start_date=None):
    if start_date is None:
        start_date = date.today()
    conn = get_conn()
    conn.execute("INSERT INTO habits (name, start_date) VALUES (?, ?)", (name, start_date))
    conn.commit()

def get_habits():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM habits ORDER BY id DESC", conn, parse_dates=["start_date"])
    return df

def delete_habit(habit_id):
    conn = get_conn()
    conn.execute("DELETE FROM habit_logs WHERE habit_id=?", (habit_id,))
    conn.execute("DELETE FROM habits WHERE id=?", (habit_id,))
    conn.commit()

def log_habit(habit_id, log_date=None):
    if log_date is None:
        log_date = date.today()
    ts = datetime.now()
    conn = get_conn()
    conn.execute(
        "INSERT INTO habit_logs (habit_id, log_date, ts) VALUES (?, ?, ?)",
        (habit_id, log_date, ts)
    )
    conn.commit()

def get_logs(habit_id=None):
    conn = get_conn()
    if habit_id:
        query = "SELECT * FROM habit_logs WHERE habit_id=? ORDER BY log_date"
        df = pd.read_sql_query(query, conn, params=(habit_id,), parse_dates=["log_date", "ts"])
    else:
        query = "SELECT * FROM habit_logs ORDER BY log_date"
        df = pd.read_sql_query(query, conn, parse_dates=["log_date", "ts"])
    return df

def get_logs_joined():
    conn = get_conn()
    query = """
        SELECT hl.*, h.name as habit_name, h.start_date
        FROM habit_logs hl
        JOIN habits h ON h.id = hl.habit_id
        ORDER BY hl.log_date
    """
    df = pd.read_sql_query(query, conn, parse_dates=["log_date", "ts", "start_date"])
    return df

# ---------------------------
# ANALYTICS
# ---------------------------
def calc_streaks(log_dates: pd.Series):
    """Return current streak and longest streak from a sorted series of unique dates."""
    if log_dates.empty:
        return 0, 0

    # Ensure unique, sorted
    # Fix: Apply date() to each element in the Series
    days = sorted(set(pd.to_datetime(log_dates).dt.date))
    longest = 1
    current = 1
    for i in range(1, len(days)):
        if (days[i] - days[i - 1]).days == 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    longest = max(longest, current)

    # current streak till today
    last_day = days[-1]
    if (date.today() - last_day).days == 0 or (date.today() - last_day).days == 1:
        # if last log is yesterday, still break streak, so only exact today counts as active
        if (date.today() - last_day).days == 0:
            current_streak = 1
            # go backwards
            for i in range(len(days) - 2, -1, -1):
                if (days[i + 1] - days[i]).days == 1:
                    current_streak += 1
                else:
                    break
        else:
            current_streak = 0
    else:
        current_streak = 0

    return current_streak, longest

def adherence_percentage(start_date: date, logs: pd.Series):
    if pd.isna(start_date):
        return 0.0
    total_days = (date.today() - start_date.date()).days + 1
    if total_days <= 0:
        return 0.0
    days_completed = len(set(logs.dt.date))
    return (days_completed / total_days) * 100

def bucket_time_of_day(ts: pd.Series):
    # morning [5-12), afternoon [12-17), evening [17-22), night otherwise
    def bucket(t):
        h = t.hour
        if 5 <= h < 12: return "Morning"
        if 12 <= h < 17: return "Afternoon"
        if 17 <= h < 22: return "Evening"
        return "Night"
    return ts.apply(bucket)

def build_weekday_heatmap(df):
    df["weekday"] = df["log_date"].dt.day_name()
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    counts = df.groupby(["habit_name","weekday"]).size().reset_index(name="count")
    counts["weekday"] = pd.Categorical(counts["weekday"], categories=weekday_order, ordered=True)
    chart = alt.Chart(counts).mark_rect().encode(
        x=alt.X('weekday:O', sort=weekday_order),
        y=alt.Y('habit_name:N'),
        color=alt.Color('count:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['habit_name', 'weekday', 'count']
    ).properties(height=200)
    return chart

def build_adherence_over_time(df_habits, df_logs):
    # compute rolling adherence per week for each habit
    frames = []
    for _, row in df_habits.iterrows():
        hid = row["id"]
        start = row["start_date"].date()
        logs = df_logs[df_logs["habit_id"] == hid]["log_date"].dt.date
        if len(logs) == 0:
            continue
        start_weeks = pd.date_range(start=start, end=date.today(), freq="W")
        for wk_end in start_weeks:
            days_elapsed = (wk_end.date() - start).days + 1
            if days_elapsed <= 0:
                continue
            completed = len([d for d in logs if d <= wk_end.date()])
            adherence = completed / days_elapsed * 100
            frames.append({"habit_id": hid, "habit_name": row["name"], "week_end": wk_end, "adherence_pct": adherence})
    if not frames:
        return None
    adh = pd.DataFrame(frames)
    chart = alt.Chart(adh).mark_line(point=True).encode(
        x='week_end:T',
        y=alt.Y('adherence_pct:Q', title='Adherence %'),
        color='habit_name:N',
        tooltip=['habit_name', 'week_end:T', alt.Tooltip('adherence_pct:Q', format='.1f')]
    ).properties(height=300)
    return chart

# ---------------------------
# FAKE DATA GENERATOR
# ---------------------------
def generate_fake_data(n_habits=3, days=28, seed=42):
    random.seed(seed)
    conn = get_conn()
    cursor = conn.cursor()

    # wipe
    cursor.execute("DELETE FROM habit_logs;")
    cursor.execute("DELETE FROM habits;")
    conn.commit()

    habit_names = ["Meditate 10min", "Read 20 pages", "Workout", "Write Journal", "Drink 2L Water"]
    for i in range(n_habits):
        name = random.choice(habit_names) + f" #{i+1}"
        start_date = date.today() - timedelta(days=days)
        cursor.execute("INSERT INTO habits (name, start_date) VALUES (?, ?)", (name, start_date))
        hid = cursor.lastrowid
        # simulate adherence ~ 60-90%
        adherence = random.uniform(0.6, 0.9)
        for d in range(days):
            if random.random() < adherence:
                log_day = start_date + timedelta(days=d)
                # random time (morning/afternoon/evening)
                hour = random.choice([7, 9, 11, 15, 18, 20, 22])
                ts = datetime.combine(log_day, datetime.min.time()) + timedelta(hours=hour, minutes=random.randint(0,59))
                cursor.execute("INSERT INTO habit_logs (habit_id, log_date, ts) VALUES (?, ?, ?)", (hid, log_day, ts))
    conn.commit()

# ---------------------------
# UI
# ---------------------------
def sidebar_actions():
    st.sidebar.header("Admin / Utilities")
    if st.sidebar.button("Generate Fake Data (28 days)"):
        generate_fake_data(n_habits=3, days=28)
        st.sidebar.success("Fake data generated.")
        st.rerun()  # Changed from st.experimental_rerun()

    if st.sidebar.button("Reset DB (delete everything)"):
        os.remove(DB_PATH) if os.path.exists(DB_PATH) else None
        init_db()
        st.sidebar.warning("Database reset.")
        st.rerun()  # Changed from st.experimental_rerun()


def show_create_habit():
    st.subheader("Create a Habit")
    with st.form("create_habit"):
        name = st.text_input("Habit name", placeholder="e.g., Meditate 10min")
        start_date = st.date_input("Start date", value=date.today())
        submitted = st.form_submit_button("Create")
        if submitted:
            if name.strip():
                create_habit(name.strip(), start_date)
                st.success(f"Habit '{name}' created.")
                st.rerun()  # Changed from st.experimental_rerun()
            else:
                st.warning("Please enter a habit name.")
                
def has_recent_log(habit_id, hours=12):
    """Check if habit has been logged within the last specified hours."""
    conn = get_conn()
    cutoff_time = datetime.now() - timedelta(hours=hours)
    query = """SELECT COUNT(*) as count FROM habit_logs 
              WHERE habit_id=? AND ts > ?"""
    result = conn.execute(query, (habit_id, cutoff_time)).fetchone()
    return result[0] > 0

def show_log_habit(df_habits):
    st.subheader("Log Today's Habit")
    if df_habits.empty:
        st.info("No habits yet. Create one above.")
        return

    habit_name = st.selectbox("Select habit", df_habits["name"])
    # Force current date only
    today = date.today()
    st.write(f"Log date: {today} (Today)")
    
    if st.button("Log"):
        hid = int(df_habits[df_habits["name"] == habit_name]["id"].iloc[0])
        
        # Check if habit was logged in the last 12 hours
        if has_recent_log(hid, hours=12):
            st.error(f"You've already logged '{habit_name}' within the last 12 hours. Please wait before logging again.")
        else:
            log_habit(hid, today)
            st.success(f"Logged '{habit_name}' for {today}.")
            st.rerun()  # Changed from st.experimental_rerun()

def show_habit_table(df_habits):
    if df_habits.empty:
        return

    st.subheader("Your Habits")
    for _, row in df_habits.iterrows():
        hid = row["id"]
        name = row["name"]
        start_date = row["start_date"].date()
        df_logs = get_logs(hid)
        cur_streak, longest = calc_streaks(df_logs["log_date"] if not df_logs.empty else pd.Series(dtype="datetime64[ns]"))
        adherence = adherence_percentage(row["start_date"], df_logs["log_date"] if not df_logs.empty else pd.Series(dtype="datetime64[ns]"))
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1:
            st.markdown(f"**{name}**  \n_Start: {start_date}_")
        with col2:
            st.metric("Current streak", cur_streak)
        with col3:
            st.metric("Longest streak", longest)
        with col4:
            st.metric("Adherence %", f"{adherence:.1f}%")
        with st.expander("Show recent logs"):
            st.dataframe(df_logs.tail(20))
        if st.button(f"Delete '{name}'", key=f"del-{hid}"):
            delete_habit(hid)
            st.warning(f"Deleted '{name}'.")
            st.rerun()  # Changed from st.experimental_rerun()
        st.markdown("---")

def show_analytics():
    st.header("Analytics & Insights")

    df = get_logs_joined()
    if df.empty:
        st.info("No logs yet. Generate fake data or start logging.")
        return

    # KPIs overall
    st.subheader("Overall KPIs")
    # group by habit
    kpi_rows = []
    for habit_id, g in df.groupby("habit_id"):
        cur_streak, longest = calc_streaks(g["log_date"])
        adh = adherence_percentage(g["start_date"].iloc[0], g["log_date"])
        # Add prediction for each habit
        success_prob = predict_habit_completion(df, habit_id)
        kpi_rows.append({
            "habit_id": habit_id,
            "habit_name": g["habit_name"].iloc[0],
            "current_streak": cur_streak,
            "longest_streak": longest,
            "adherence_pct": round(adh, 1),
            "tomorrow_prob": round(success_prob * 100, 1)
        })
    kpis = pd.DataFrame(kpi_rows).sort_values("adherence_pct", ascending=False)
    st.dataframe(kpis)

    # time of day distribution
    st.subheader("Time-of-Day Distribution")
    df["tod_bucket"] = bucket_time_of_day(df["ts"])
    tod_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('tod_bucket:N', title='Time of Day'),
        y='count():Q',
        color='habit_name:N',
        tooltip=['habit_name', 'tod_bucket', 'count()']
    ).properties(height=300)
    st.altair_chart(tod_chart, use_container_width=True)

    # weekday heatmap
    st.subheader("Weekday Heatmap")
    heat = build_weekday_heatmap(df)
    st.altair_chart(heat, use_container_width=True)

    # adherence over time
    st.subheader("Adherence Over Time (Weekly)")
    df_habits = get_habits()
    chart = build_adherence_over_time(df_habits, df)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough data to compute adherence trend yet.")

    # AI insights section (renamed from Auto-Insights)
    st.subheader("AI Insights")
    best = kpis.sort_values("adherence_pct", ascending=False).iloc[0]
    worst = kpis.sort_values("adherence_pct", ascending=True).iloc[0]
    
    # Display predictions
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tomorrow's Success Chance", f"{best.tomorrow_prob}%", 
                 f"{best.tomorrow_prob - 50:.1f}%", 
                 delta_color="normal")
        st.write(f"✅ **Best adherence:** **{best.habit_name}** at **{best.adherence_pct}%**.") 
    
    with col2:
        st.metric("Improvement Opportunity", f"{worst.tomorrow_prob}%", 
                 f"{worst.tomorrow_prob - 50:.1f}%", 
                 delta_color="normal")
        st.write(f"⚠️ **Lowest adherence:** **{worst.habit_name}** at **{worst.adherence_pct}%**")
    
    top_tod = (
        df.groupby("tod_bucket").size().sort_values(ascending=False).index[0]
        if not df.empty else "N/A"
    )
    st.write(f"⏰ You most frequently log habits in the **{top_tod}**.") 
    
    # Add Gemini-powered insight
    try:
        gemini_api = GeminiAPI()
        habit_summary = kpis.to_dict('records')
        ai_insight = gemini_api.generate_insight(habit_summary)
        st.markdown("### AI-Generated Insight")
        st.markdown(f"*{ai_insight}*")
    except Exception as e:
        st.warning(f"AI insight generation unavailable. Set up your Gemini API key to enable this feature.")

# ---------------------------
# MAIN
# ---------------------------
def main():
    st.set_page_config(page_title="Micro-Habit Tracker (AI-Enhanced)", layout="wide")
    st.title("🧠 Micro-Habit Tracker with AI Analytics")

    init_db()
    sidebar_actions()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📌 Habits", "🧾 Logs", "📊 Analytics"])

    with tab1:
        show_create_habit()
        st.markdown("---")
        df_habits = get_habits()
        show_habit_table(df_habits)

    with tab2:
        df_habits = get_habits()
        show_log_habit(df_habits)
        st.markdown("---")
        st.subheader("All Logs")
        df = get_logs_joined()
        st.dataframe(df if not df.empty else pd.DataFrame())

    with tab3:
        show_analytics()


def predict_habit_completion(df, habit_id):
    """Predict the probability of completing a habit tomorrow using logistic regression"""
    if df.empty or len(df) < 5:  # Need minimum data for prediction
        return 0.5  # Default 50% if not enough data
    
    # Filter for the specific habit
    habit_df = df[df['habit_id'] == habit_id].copy()
    if len(habit_df) < 5:
        return 0.5
    
    # Feature engineering
    habit_df['log_date'] = pd.to_datetime(habit_df['log_date'])
    habit_df['weekday'] = habit_df['log_date'].dt.weekday
    habit_df['month'] = habit_df['log_date'].dt.month
    habit_df['day'] = habit_df['log_date'].dt.day
    
    # Create a dataset of all dates from start to now
    start_date = pd.to_datetime(habit_df['start_date'].min())
    end_date = pd.to_datetime(date.today())
    all_dates = pd.DataFrame({'date': pd.date_range(start_date, end_date)})
    all_dates['weekday'] = all_dates['date'].dt.weekday
    all_dates['month'] = all_dates['date'].dt.month
    all_dates['day'] = all_dates['date'].dt.day
    
    # Mark dates when habit was completed
    all_dates['completed'] = all_dates['date'].isin(pd.to_datetime(habit_df['log_date']))
    
    # Prepare features and target
    X = all_dates[['weekday', 'month', 'day']]
    y = all_dates['completed']
    
    # Create a simple model
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), ['day']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['weekday', 'month'])
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced'))
    ])
    
    # Train the model
    model.fit(X, y)
    
    # Predict for tomorrow
    tomorrow = date.today() + timedelta(days=1)
    tomorrow_features = pd.DataFrame({
        'weekday': [tomorrow.weekday()],
        'month': [tomorrow.month],
        'day': [tomorrow.day]
    })
    
    # Return probability of completion
    return model.predict_proba(tomorrow_features)[0][1]  # Probability of class 1 (completion)

if __name__ == "__main__":
    main()
