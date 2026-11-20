import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease | KNN Studio",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
.stApp { background: linear-gradient(135deg, #071426 0%, #0b1d32 55%, #10283c 100%); color: #edf6ff; }
[data-testid="stSidebar"] { background: #091a2b; border-right: 1px solid #1c3a52; }
.block-container { padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1450px; }
.hero {
  padding: 26px 30px; border-radius: 22px; margin-bottom: 22px;
  background: linear-gradient(115deg, rgba(13,148,136,.22), rgba(14,165,233,.12));
  border: 1px solid rgba(103,232,249,.22);
}
.hero h1 { font-family: 'Kanit', sans-serif; font-size: 2.25rem; margin: 0; color: #f0fbff; }
.hero p { color: #a9c4d8; margin: 7px 0 0 0; }
.metric-card {
  background: linear-gradient(145deg, #10283e, #0d2033);
  border: 1px solid #24445d; border-radius: 16px; padding: 17px 19px;
}
.metric-label { color: #9ab6ca; font-size: .86rem; }
.metric-value { color: #f2fbff; font-size: 1.8rem; font-weight: 700; margin-top: 3px; }
.metric-note { color: #70d9d0; font-size: .78rem; }
.section-title { color: #e8f6ff; font-family: 'Kanit', sans-serif; font-size: 1.35rem; font-weight: 600; margin: 8px 0 12px; }
.info-panel {
  background: rgba(14, 38, 57, .82); border: 1px solid #24465f;
  padding: 16px 18px; border-radius: 14px; color: #c4d9e8;
}
div[data-testid="stForm"] { background: rgba(14,38,57,.72); border: 1px solid #24465f; padding: 18px; border-radius: 16px; }
.stButton > button, .stFormSubmitButton > button {
  background: linear-gradient(90deg, #0d9488, #0891b2); color: white;
  border: 0; border-radius: 10px; font-weight: 600; min-height: 2.7rem;
}
.stButton > button:hover, .stFormSubmitButton > button:hover { border: 1px solid #67e8f9; color: white; }
h2, h3 { font-family: 'Kanit', sans-serif !important; color: #e8f6ff !important; }
p, label, .stMarkdown { color: #d7e7f2; }
[data-testid="stDataFrame"] { border: 1px solid #24465f; border-radius: 12px; }
small { color: #9ab6ca; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]
TARGET = "HeartDisease"
DISPLAY = {
    "Age": "อายุ (ปี)",
    "Sex": "เพศ (รหัส)",
    "ChestPainType": "ประเภทอาการเจ็บหน้าอก (รหัส)",
    "RestingBP": "ความดันโลหิตขณะพัก",
    "Cholesterol": "คอเลสเตอรอล",
    "FastingBS": "น้ำตาลขณะอดอาหาร (รหัส)",
    "RestingECG": "ผล ECG ขณะพัก (รหัส)",
    "MaxHR": "อัตราการเต้นหัวใจสูงสุด",
    "ExerciseAngina": "เจ็บหน้าอกขณะออกกำลังกาย (รหัส)",
    "Oldpeak": "Oldpeak",
    "ST_Slope": "ST Slope (รหัส)"
}

@st.cache_data
def load_default_data():
    return pd.read_csv("Heart3.csv")

def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.markdown("## 💙 Heart KNN Studio")
st.sidebar.caption("เครื่องมือจำแนกข้อมูลโรคหัวใจด้วย K-Nearest Neighbors")
page = st.sidebar.radio(
    "เมนูหลัก",
    ["ภาพรวมข้อมูล", "สำรวจข้อมูล", "ฝึกและประเมินโมเดล", "ทำนายข้อมูลใหม่"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ ตั้งค่าโมเดล")
k_value = st.sidebar.slider("จำนวนเพื่อนบ้าน (K)", min_value=1, max_value=25, value=5, step=1)
test_size = st.sidebar.slider("สัดส่วนข้อมูลทดสอบ (%)", 10, 40, 20, 5)
st.sidebar.caption("ใช้การแบ่งข้อมูลแบบ Stratified และ random_state = 42")
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Heart3.csv • Algorithm: KNN")

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
try:
    df = load_default_data()
except FileNotFoundError:
    st.error("ไม่พบไฟล์ Heart3.csv กรุณาวางไฟล์ CSV ไว้ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_cols:
    st.error(f"ไฟล์ข้อมูลขาดคอลัมน์ที่จำเป็น: {', '.join(missing_cols)}")
    st.stop()

df = df[FEATURES + [TARGET]].copy()
df = df.dropna()
X = df[FEATURES]
y = df[TARGET].astype(int)

if not set(y.unique()).issubset({0, 1}):
    st.error("คอลัมน์ HeartDisease ต้องมีค่าเป้าหมายเป็น 0 และ 1 เท่านั้น")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=k_value, weights="uniform", metric="minkowski"))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>💙 Heart Disease Prediction</h1>
  <p>K-Nearest Neighbors Classification · วิเคราะห์ข้อมูลและทดลองทำนายความเสี่ยงจากชุดข้อมูล Heart3</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
if page == "ภาพรวมข้อมูล":
    st.markdown('<div class="section-title">ภาพรวมชุดข้อมูล</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("จำนวนข้อมูล", f"{len(df):,}", "Records")
    with c2: metric_card("จำนวนคุณลักษณะ", f"{len(FEATURES)}", "Input Features")
    with c3: metric_card("คลาส 0", f"{(y == 0).sum():,}", f"{(y == 0).mean()*100:.1f}% ของข้อมูล")
    with c4: metric_card("คลาส 1", f"{(y == 1).sum():,}", f"{(y == 1).mean()*100:.1f}% ของข้อมูล")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### สัดส่วนผลลัพธ์ HeartDisease")
        counts = df[TARGET].value_counts().sort_index().rename(index={0:"0 · ไม่พบโรคในชุดข้อมูล", 1:"1 · พบโรคในชุดข้อมูล"}).reset_index()
        counts.columns = ["ผลลัพธ์", "จำนวน"]
        fig = px.pie(counts, names="ผลลัพธ์", values="จำนวน", hole=.62,
                     color_discrete_sequence=["#14b8a6", "#fb7185"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#d7e7f2", legend_title_text="")
        fig.update_traces(textposition="inside", textinfo="percent+value")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### สถิติข้อมูลตัวเลข")
        st.dataframe(df[FEATURES].describe().T.round(2), use_container_width=True, height=340)

    st.markdown("### ข้อมูลตัวอย่าง")
    st.dataframe(df.head(12), use_container_width=True)
    st.markdown('<div class="info-panel">หมายเหตุ: ค่า 0 ใน RestingBP หรือ Cholesterol ปรากฏอยู่ในไฟล์ต้นฉบับ ระบบจึงคงค่าตามข้อมูลเดิม ไม่ได้แทนค่าหรือลบออกโดยอัตโนมัติ</div>', unsafe_allow_html=True)

elif page == "สำรวจข้อมูล":
    st.markdown('<div class="section-title">Data Explorer</div>', unsafe_allow_html=True)
    st.write("เลือกคอลัมน์เพื่อดูการกระจายของข้อมูล")
    col = st.selectbox("เลือกตัวแปร", FEATURES + [TARGET], index=0)
    if df[col].nunique() <= 15:
        vc = df[col].value_counts().sort_index().reset_index()
        vc.columns = [col, "จำนวน"]
        fig = px.bar(vc, x=col, y="จำนวน", text="จำนวน", color="จำนวน",
                     color_continuous_scale=["#155e75", "#2dd4bf"])
    else:
        fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=["#2dd4bf"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#d7e7f2", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ความสัมพันธ์กับ HeartDisease")
    selected = st.multiselect("เลือกตัวแปรเพื่อเปรียบเทียบ", FEATURES, default=["Age", "Cholesterol", "MaxHR"])
    if selected:
        view = df[selected + [TARGET]].copy()
        st.dataframe(view.groupby(TARGET).mean(numeric_only=True).T.round(2), use_container_width=True)
    st.markdown("### ตรวจสอบคุณภาพข้อมูล")
    q1, q2, q3 = st.columns(3)
    with q1: metric_card("Missing values", int(df.isna().sum().sum()), "หลังเตรียมข้อมูล")
    with q2: metric_card("ข้อมูลซ้ำ", int(df.duplicated().sum()), "Duplicate rows")
    with q3: metric_card("จำนวนแถวที่ใช้", f"{len(df):,}", "หลังตัดแถวที่มีค่าว่าง")

elif page == "ฝึกและประเมินโมเดล":
    st.markdown('<div class="section-title">Model Training & Evaluation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-panel">โมเดลใช้ K = <b>{k_value}</b> · แบ่งข้อมูลฝึก {100-test_size}% และทดสอบ {test_size}% · StandardScaler + KNN</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    with a: metric_card("Accuracy", f"{accuracy*100:.2f}%", "ความถูกต้องโดยรวม")
    with b: metric_card("Precision", f"{precision*100:.2f}%", "ความแม่นยำของคลาส 1")
    with c: metric_card("Recall", f"{recall*100:.2f}%", "การตรวจพบคลาส 1")
    with d: metric_card("F1-score", f"{f1*100:.2f}%", "ค่าเฉลี่ย Precision/Recall")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        fig = px.imshow(cm, text_auto=True, x=["ทำนาย 0", "ทำนาย 1"], y=["จริง 0", "จริง 1"],
                        color_continuous_scale="Teal", aspect="auto")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#d7e7f2", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True, height=300)

    st.markdown("### ผลการทำนายชุดทดสอบ")
    result_df = X_test.copy()
    result_df["ค่าจริง"] = y_test
    result_df["ค่าทำนาย"] = y_pred
    result_df["ผล"] = np.where(y_test.to_numpy() == y_pred, "ถูกต้อง", "คลาดเคลื่อน")
    st.dataframe(result_df.head(30), use_container_width=True)
    st.download_button("⬇️ ดาวน์โหลดผลการทำนายชุดทดสอบ (CSV)",
                       result_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="knn_test_predictions.csv", mime="text/csv")

elif page == "ทำนายข้อมูลใหม่":
    st.markdown('<div class="section-title">ทำนายข้อมูลใหม่ด้วย KNN</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-panel">กรอกค่าคุณลักษณะทั้ง 11 รายการ แล้วกดปุ่มทำนาย ระบบจะใช้โมเดลที่ฝึกจาก Heart3.csv ประเมินคลาสที่ใกล้เคียงที่สุด</div>', unsafe_allow_html=True)
    ranges = {c: (float(df[c].min()), float(df[c].max())) for c in FEATURES}
    categories = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]

    with st.form("prediction_form"):
        st.markdown("### ข้อมูลพื้นฐาน")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input(DISPLAY["Age"], min_value=1, max_value=100, value=50, step=1)
        with c2:
            sex = st.selectbox(DISPLAY["Sex"], sorted(df["Sex"].unique().tolist()))
        with c3:
            chest = st.selectbox(DISPLAY["ChestPainType"], sorted(df["ChestPainType"].unique().tolist()))

        st.markdown("### ค่าตรวจสุขภาพ")
        c1, c2, c3 = st.columns(3)
        with c1:
            bp = st.number_input(DISPLAY["RestingBP"], min_value=0, max_value=300, value=120, step=1)
        with c2:
            chol = st.number_input(DISPLAY["Cholesterol"], min_value=0, max_value=1000, value=200, step=1)
        with c3:
            fasting = st.selectbox(DISPLAY["FastingBS"], sorted(df["FastingBS"].unique().tolist()))

        c1, c2, c3 = st.columns(3)
        with c1:
            ecg = st.selectbox(DISPLAY["RestingECG"], sorted(df["RestingECG"].unique().tolist()))
        with c2:
            maxhr = st.number_input(DISPLAY["MaxHR"], min_value=30, max_value=250, value=140, step=1)
        with c3:
            angina = st.selectbox(DISPLAY["ExerciseAngina"], sorted(df["ExerciseAngina"].unique().tolist()))

        c1, c2 = st.columns(2)
        with c1:
            oldpeak = st.number_input(DISPLAY["Oldpeak"], min_value=-5.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")
        with c2:
            slope = st.selectbox(DISPLAY["ST_Slope"], sorted(df["ST_Slope"].unique().tolist()))

        submitted = st.form_submit_button("🔍 ทำนายผลข้อมูลนี้", use_container_width=True)

    if submitted:
        new_data = pd.DataFrame([{
            "Age": age, "Sex": sex, "ChestPainType": chest, "RestingBP": bp,
            "Cholesterol": chol, "FastingBS": fasting, "RestingECG": ecg,
            "MaxHR": maxhr, "ExerciseAngina": angina, "Oldpeak": oldpeak, "ST_Slope": slope
        }], columns=FEATURES)

        prediction = int(model.predict(new_data)[0])
        probabilities = model.predict_proba(new_data)[0]
        class_index = list(model.named_steps["knn"].classes_).index(prediction)
        confidence = float(probabilities[class_index])

        st.markdown("---")
        st.markdown("### ผลการจำแนกจากโมเดล")
        if prediction == 1:
            st.error("ผลทำนาย: คลาส 1 — พบรูปแบบที่โมเดลจัดอยู่ในกลุ่ม HeartDisease")
        else:
            st.success("ผลทำนาย: คลาส 0 — พบรูปแบบที่โมเดลจัดอยู่ในกลุ่มไม่มี HeartDisease")
        p1, p2 = st.columns(2)
        with p1:
            metric_card("Predicted Class", str(prediction), "ผลจากโมเดล KNN")
        with p2:
            metric_card("สัดส่วนคะแนนของคลาสที่ทำนาย", f"{confidence*100:.1f}%", "สัดส่วนเพื่อนบ้านในคลาสที่ทำนาย ไม่ใช่ความน่าจะเป็นทางการแพทย์")
        prob_df = pd.DataFrame({"คลาส": ["0", "1"], "สัดส่วนเพื่อนบ้าน (%)": probabilities * 100})
        fig = px.bar(prob_df, x="คลาส", y="สัดส่วนเพื่อนบ้าน (%)", text_auto=".1f",
                     color="คลาส", color_discrete_map={"0":"#14b8a6", "1":"#fb7185"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#d7e7f2", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### ข้อมูลที่ส่งให้โมเดล")
        st.dataframe(new_data, use_container_width=True)
        st.warning("คำเตือน: ผลลัพธ์นี้เป็นการสาธิตการจำแนกด้วย Machine Learning จากชุดข้อมูล Heart3.csv เท่านั้น ไม่ใช่การวินิจฉัยโรคหรือคำแนะนำทางการแพทย์ โปรดปรึกษาแพทย์เพื่อประเมินสุขภาพจริง")

st.markdown("---")
st.caption("Heart KNN Studio · Educational Machine Learning Project · ผลการทำนายขึ้นอยู่กับข้อมูลและการตั้งค่าโมเดล")
