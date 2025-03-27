from datetime import datetime, timedelta
import os
import json
import dayplot as dp
from collections import defaultdict
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import logging
import numpy as np

LOG_STAT_FILE = "logs/token_usage.json"
LOG_FILE = "logs/usage_log.json"
LOG_EVAL_FILE = "logs/evaluations.jsonl"
LOG_CHAT_HISTORY = "logs/chat_history.txt"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_eval_stats_plot():
    # Initialize distribution stats for 'values' fields
    if not os.path.exists("logs/evaluations.jsonl"):
        logger.warning("No data to plot.")
        return

    values_distributions = defaultdict(list)

    # Process the 'values' field again to get distributions
    with open("logs/evaluations.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                if "evaluation" in data and isinstance(data["evaluation"], dict):
                    for key, value in data["evaluation"].items():
                        values_distributions[key].append(value)
                values_distributions["liked (bool)"].append(int(eval(data["liked"]))*5) #convert bool to int and from 0-1 to 0-5
            except json.JSONDecodeError:
                continue

    numeric_data = {key: values for key, values in values_distributions.items() if all(isinstance(v, (int, float)) or v is None for v in values)}

    # Compute means and standard deviations
    categories = list(numeric_data.keys())
    means = [np.mean([v for v in values if (v is not None and v>=0)]) for values in numeric_data.values()]
    stds = [np.std([v for v in values if (v is not None and v>=0)]) for values in numeric_data.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=means,
        error_y=dict(type='data', array=stds, visible=True),
        marker_color='#10b981'
    ))
    fig.update_layout(
        title="Evaluation statistics",
        xaxis_title="Categories",
        yaxis_title="Mean and Standard Deviation",
        template="plotly_white"
    )
    return fig

def get_usage_stats():
    """Computes total users, total input/output tokens, averages, and cumulative daily token usage."""
    if not os.path.exists(LOG_STAT_FILE):
        return {
            "total_users": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "avg_input_tokens_per_user_per_day": 0,
            "avg_output_tokens_per_user_per_day": 0,
            "cumulative_tokens_per_day": [],
            "cumulative_input_tokens_per_day": [],
            "cumulative_output_tokens_per_day": []
        }

    with open(LOG_STAT_FILE, "r") as file:
        data = json.load(file)

    total_users = len(data)
    total_input_tokens = 0
    total_output_tokens = 0
    daily_totals = defaultdict(lambda: [0, 0])  # {date: [input_tokens, output_tokens]}

    for usage in data.values():
        for tokens, timestamp in usage["input_tokens"]:
            date = datetime.fromisoformat(timestamp).date()
            total_input_tokens += tokens
            daily_totals[date][0] += tokens

        for tokens, timestamp in usage["output_tokens"]:
            date = datetime.fromisoformat(timestamp).date()
            total_output_tokens += tokens
            daily_totals[date][1] += tokens

    # Compute averages
    active_days = len(daily_totals)
    avg_input_tokens_per_user_per_day = total_input_tokens / (total_users * active_days) if total_users > 0 and active_days > 0 else 0
    avg_output_tokens_per_user_per_day = total_output_tokens / (total_users * active_days) if total_users > 0 and active_days > 0 else 0

    # Cumulative token count per day
    sorted_dates = sorted(daily_totals.keys())
    cumulative_input = 0
    cumulative_output = 0
    cumulative_tokens_per_day = []
    cumulative_input_tokens_per_day = []
    cumulative_output_tokens_per_day = []

    for date in sorted_dates:
        cumulative_input += daily_totals[date][0]
        cumulative_output += daily_totals[date][1]
        cumulative_tokens_per_day.append((cumulative_input + cumulative_output, date.isoformat()))
        cumulative_input_tokens_per_day.append((cumulative_input, date.isoformat()))
        cumulative_output_tokens_per_day.append((cumulative_output, date.isoformat()))

    return {
        "total_users": total_users,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_input_tokens_per_user_per_day": round(avg_input_tokens_per_user_per_day),
        "avg_output_tokens_per_user_per_day": round(avg_output_tokens_per_user_per_day),
        "cumulative_tokens_per_day": cumulative_tokens_per_day,
        "cumulative_input_tokens_per_day": cumulative_input_tokens_per_day,
        "cumulative_output_tokens_per_day": cumulative_output_tokens_per_day,
        "daily_totals": daily_totals,
    }

def log_token_usage(ip_address: str, input_tokens: int, output_tokens: int):
    """Logs the input and output token usage for a given IP address with timestamps."""
    os.makedirs(os.path.dirname(LOG_STAT_FILE), exist_ok=True)

    if os.path.exists(LOG_STAT_FILE):
        with open(LOG_STAT_FILE, "r") as file:
            data = json.load(file)
    else:
        data = {}

    # Ensure IP has an entry
    if ip_address not in data:
        data[ip_address] = {"input_tokens": [], "output_tokens": []}

    # Append new token usage
    now = datetime.now().isoformat()
    data[ip_address]["input_tokens"].append((input_tokens, now))
    data[ip_address]["output_tokens"].append((output_tokens, now))

    # Save back to file
    with open(LOG_STAT_FILE, "w") as file:
        json.dump(data, file, indent=4)

def plot_cumulative_tokens():
    """Plots cumulative token usage over time with stacked bars for input and output tokens and a line for total cumulative tokens using Plotly."""
    stats = get_usage_stats()
    if not stats["cumulative_tokens_per_day"]:
        logger.warning("No data to plot.")
        return

    dates = [datetime.fromisoformat(d) for _, d in stats["cumulative_tokens_per_day"]]
    input_tokens = [t for t, _ in stats["cumulative_input_tokens_per_day"]]
    output_tokens = [t for t, _ in stats["cumulative_output_tokens_per_day"]]
    total_tokens = [t for t, _ in stats["cumulative_tokens_per_day"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(x=dates, y=input_tokens, name="Input Tokens", marker_color='royalblue',width=1000*3600*24*0.5))
    fig.add_trace(go.Bar(x=dates, y=output_tokens, name="Output Tokens", marker_color='lightsalmon',width=1000*3600*24*0.5))
    fig.add_trace(go.Scatter(x=dates, y=total_tokens, mode='lines+markers', name="Total", line=dict(color='darkslategrey')))

    fig.update_layout(
        title="Cumulative Token Usage Over Time",
        xaxis_title="Date",
        yaxis_title="Tokens [tokens/dd]",
        barmode='stack',
        xaxis=dict(tickangle=-45),
        legend_title="Legend",
        template="plotly_white"
    )

    return fig


def plot_daily_tokens_heatmap():
    """Plots cumulative token usage over time with stacked bars for input and output tokens and a line for total cumulative tokens using Plotly."""
    stats = get_usage_stats()
    if not stats["daily_totals"]:
        logger.warning("No data to plot.")
        return

    dates = stats["daily_totals"].keys()
    total_tokens = [t[0]+t[1] for _, t in stats["daily_totals"].items()]
    fig, ax = plt.subplots(figsize=(15, 6))
    dp.calendar(
        dates,
        total_tokens,
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now(),
        ax=ax,
    )
    fig.tight_layout()

    return fig

def read_usage_log(ip_address: str) -> (int, datetime, bool):
    if not os.path.exists(LOG_FILE):
        return 0, datetime.now(), False
    with open(LOG_FILE, "r") as file:
        data = json.load(file)
    if ip_address in data:
        entry = data[ip_address]
        return entry["tokens_count"], datetime.fromisoformat(entry["last_call"]), entry["banned_flag"]
    return 0, datetime.now(), False

def update_usage_log(ip_address: str, tokens_consumed: int, banned: bool):
    """Updates the usage log, modifying the entry for the given IP address."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as file:
            data = json.load(file)
    else:
        data = {}

    was_banned = data.get(ip_address, {}).get("banned_flag", False)
    tokens_count = data.get(ip_address, {}).get("tokens_count", 0) + tokens_consumed

    if was_banned and not banned:
        tokens_count = 0  # Reset tokens if ban is lifted

    data[ip_address] = {
        "tokens_count": tokens_count,
        "last_call": datetime.now().isoformat(),
        "banned_flag": banned
    }

    with open(LOG_FILE, "w") as file:
        json.dump(data, file, indent=4)


def export_history(history):
    with open(LOG_CHAT_HISTORY, 'w') as f:
        f.write(str(history))
    return LOG_CHAT_HISTORY

