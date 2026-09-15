import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models.pinn import PINN

st.set_page_config(page_title="Tsunami PINN", layout="wide", page_icon="🌊")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); }
    h1 { color: #4da6ff !important; text-shadow: 0 0 20px rgba(77,166,255,0.5); }
    .stMetric { background: rgba(77,166,255,0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(77,166,255,0.3); }
    .stMetric label { color: #8ab4f8 !important; }
    .stMetric value { color: #ffffff !important; font-size: 2em !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌊 Tsunami Wave Propagation — Physics-Informed AI")
st.markdown("*Real-time wave simulation powered by a neural network that obeys the shallow-water wave equations.*")

N_TIME_FRAMES = 100  # Precompute 100 frames between t=0 and t=1

@st.cache_resource
def load_model():
    model = PINN()
    model.load_state_dict(torch.load("models/pinn_tsunami.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_data(show_spinner=False)
def precompute_waves(resolution, n_time_frames):
    """Precompute wave heights for ALL time steps at once. Runs once per resolution change."""
    model = load_model()
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    t = torch.linspace(0, 1, n_time_frames)
    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)
    with torch.no_grad():
        eta, _, _ = model(x_flat, y_flat, t_flat)
    # Shape: (resolution, resolution, n_time_frames)
    waves = eta.reshape(resolution, resolution, n_time_frames).numpy()
    return waves, x.numpy(), y.numpy(), t.numpy()

@st.cache_data(show_spinner=False)
def precompute_cities(n_time_frames):
    """Precompute city impact curves. Runs once."""
    model = load_model()
    cities = {
        "Jakarta": (0.85, 0.25),
        "Colombo": (0.15, 0.55),
        "Mumbai":  (0.25, 0.85),
        "Perth":   (0.90, 0.75),
        "Bangkok": (0.70, 0.10),
    }
    t = torch.linspace(0, 1, n_time_frames).reshape(-1, 1)
    results = {}
    for name, (cx, cy) in cities.items():
        cx_b = torch.full_like(t, cx)
        cy_b = torch.full_like(t, cy)
        with torch.no_grad():
            eta, _, _ = model(cx_b, cy_b, t)
        eta_np = eta.numpy().flatten()
        results[name] = {
            "x": cx, "y": cy,
            "curve": eta_np,
            "peak": float(np.max(np.abs(eta_np))),
            "arrival": float(t.numpy().flatten()[np.argmax(np.abs(eta_np))]),
        }
    return results, t.numpy().flatten()

# --- Sidebar ---
st.sidebar.markdown("### ⚙️ Simulation Controls")
time_step = st.sidebar.slider("⏱ Time (normalized)", 0.0, 1.0, 0.0, 0.01)
resolution = st.sidebar.slider("🔍 Grid Resolution", 40, 120, 80)
view_mode = st.sidebar.radio("🎨 View Mode", ["2D Heatmap", "3D Surface", "Split View"])
color_theme = st.sidebar.selectbox("🌈 Color Palette", ["Thermal", "Ocean", "Plasma", "Turbo"])

themes = {"Thermal": "RdBu_r", "Ocean": "Blues", "Plasma": "Plasma", "Turbo": "Turbo"}
cmap = themes[color_theme]

# --- Precompute (cached) ---
waves_3d, x_np, y_np, t_np = precompute_waves(resolution, N_TIME_FRAMES)
city_data, city_t = precompute_cities(N_TIME_FRAMES)

# --- Slice the precomputed frame instantly ---
t_idx = int(np.clip(np.round(time_step * (N_TIME_FRAMES - 1)), 0, N_TIME_FRAMES - 1))
wave = np.nan_to_num(waves_3d[:, :, t_idx], nan=0.0, posinf=0.0, neginf=0.0)
X_np, Y_np = np.meshgrid(x_np, y_np, indexing="ij")
city_curve_idx = t_idx

# --- Build figures ---
if view_mode == "2D Heatmap":
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=wave, x=x_np, y=y_np,
        colorscale=cmap, zmid=0,
        colorbar=dict(title="Wave (m)", tickfont=dict(color="white")),
    ))
    fig.add_trace(go.Scatter(
        x=[c["x"] for c in city_data.values()],
        y=[c["y"] for c in city_data.values()],
        mode="markers+text",
        marker=dict(size=15, color="cyan", symbol="star", line=dict(color="white", width=2)),
        text=list(city_data.keys()), textposition="top right",
        textfont=dict(color="white", size=12), name="Cities"
    ))
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"Wave Height at t = {time_step:.2f}", font=dict(size=20, color="white")),
        xaxis=dict(title="X (space)", showgrid=False),
        yaxis=dict(title="Y (space)", showgrid=False, scaleanchor="x"),
        height=650, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,14,39,1)",
        uirevision="keep",  # keeps zoom/pan between slider moves
    )

elif view_mode == "3D Surface":
    fig = go.Figure(data=[go.Surface(
        z=wave, x=x_np, y=y_np,
        colorscale=cmap, cmid=0,
        colorbar=dict(title="Wave (m)", tickfont=dict(color="white")),
    )])
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"3D Wave Surface at t = {time_step:.2f}", font=dict(size=20, color="white")),
        scene=dict(
            xaxis=dict(title="X", backgroundcolor="rgba(10,14,39,1)", gridcolor="rgba(77,166,255,0.2)"),
            yaxis=dict(title="Y", backgroundcolor="rgba(10,14,39,1)", gridcolor="rgba(77,166,255,0.2)"),
            zaxis=dict(title="Wave Height", backgroundcolor="rgba(10,14,39,1)", gridcolor="rgba(77,166,255,0.2)"),
            bgcolor="rgba(10,14,39,1)",
        ),
        height=700, paper_bgcolor="rgba(0,0,0,0)",
        uirevision="keep",
    )

else:
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "heatmap"}, {"type": "surface"}]],
        subplot_titles=("2D Heatmap", "3D Surface"),
    )
    fig.add_trace(go.Heatmap(z=wave, colorscale=cmap, zmid=0, showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(z=wave, colorscale=cmap, cmid=0, showscale=False), row=1, col=2)
    fig.update_layout(
        template="plotly_dark", height=650,
        paper_bgcolor="rgba(0,0,0,0)", uirevision="keep",
    )

st.plotly_chart(fig, use_container_width=True)

# --- Live metrics ---
st.markdown("### 📊 Live Wave Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Peak Wave", f"{wave.max():.3f} m")
col2.metric("Wave Trough", f"{wave.min():.3f} m")
col3.metric("Energy", f"{np.sum(wave**2):.2f}")
col4.metric("Grid", f"{resolution}×{resolution}")

# --- City Impact Panel ---
st.markdown("### 🏙️ Coastal City Impact")
cols = st.columns(len(city_data))
for i, (name, c) in enumerate(city_data.items()):
    with cols[i]:
        hit = time_step >= c["arrival"]
        status = "🚨 HIT" if hit else "⏳ WAIT"
        color = "#ff4444" if hit else "#4da6ff"
        current_height = c["curve"][city_curve_idx]
        st.markdown(f"""
        <div style="background: rgba(77,166,255,0.08); padding: 15px;
                    border-radius: 12px; border-left: 4px solid {color};">
            <h4 style="color: {color}; margin: 0;">{name}</h4>
            <p style="color: #8ab4f8; margin: 5px 0;">{status}</p>
            <p style="color: white; margin: 5px 0;">Current: <b>{current_height:.3f} m</b></p>
            <p style="color: white; margin: 5px 0;">Peak: <b>{c['peak']:.3f} m</b></p>
            <p style="color: white; margin: 5px 0;">Arrival: t = <b>{c['arrival']:.2f}</b></p>
        </div>
        """, unsafe_allow_html=True)

# --- City wave height curves ---
with st.expander("📈 Wave Height Timeline for Each City"):
    fig_curves = go.Figure()
    for name, c in city_data.items():
        fig_curves.add_trace(go.Scatter(
            x=city_t, y=c["curve"], mode="lines", name=name,
            line=dict(width=2),
        ))
    fig_curves.add_vline(x=time_step, line_dash="dash", line_color="red",
                         annotation_text=f"t={time_step:.2f}")
    fig_curves.update_layout(
        template="plotly_dark",
        xaxis_title="Time (normalized)", yaxis_title="Wave Height (m)",
        height=400, paper_bgcolor="rgba(0,0,0,0)",
        uirevision="keep",
    )
    st.plotly_chart(fig_curves, use_container_width=True)

st.markdown("---")
st.caption("⚡ Physics-Informed Neural Network | Shallow-Water Equations | PyTorch + Streamlit + Plotly")