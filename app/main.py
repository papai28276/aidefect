"""Defect QC workbench: upload process/image data, train CPU models, review COPQ."""
from __future__ import annotations

import sys
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from defectlab.upload.pipeline import train_tabular, train_images, predict_images, permutation_importances
from defectlab.upload.loaders import read_tabular_upload, load_image_uploads
from defectlab.upload.fusion import train_real_pairing
from defectlab.upload.cost import cost_curve, compare_scenarios, sensitivity

st.set_page_config(page_title="Defect QC Workbench", page_icon=":material/precision_manufacturing:", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#10243a; --muted:#718096; --teal:#0f9f92; --line:#e5ebf2; --violet:#7666f5; --aqua:#38d8c4; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
[data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 90% 2%,rgba(118,102,245,.085),transparent 31%),radial-gradient(ellipse at 8% 38%,rgba(56,216,196,.06),transparent 29%),#f4f7fc; }
[data-testid="stHeader"] { background:rgba(244,247,252,.76); backdrop-filter:blur(16px); }
[data-testid="stSidebar"] { background:linear-gradient(165deg,#0c1a2b 0%,#132d43 62%,#154451 100%); border-right:1px solid rgba(255,255,255,.1); box-shadow:8px 0 30px rgba(14,31,50,.08); }
[data-testid="stSidebar"] h1 { color:#fff; letter-spacing:-.04em; }
[data-testid="stSidebar"] * { color:#e7f0f7; }
[data-testid="stSidebar"] [data-baseweb="select"] * { color:#10243a; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#9bb0c0; }
h1,h2,h3 { color:var(--ink); font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
h1 { font-size:2.2rem !important; }
[data-testid="stMetric"] { position:relative; overflow:hidden; background:linear-gradient(145deg,#fff 35%,#f4fbfa); border:1px solid rgba(217,230,237,.9); border-radius:19px; padding:18px 19px; box-shadow:0 8px 24px rgba(16,36,58,.055); transition:transform .2s ease,box-shadow .2s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-3px); box-shadow:0 14px 30px rgba(16,36,58,.10); }
[data-testid="stMetric"]:before { content:""; position:absolute; left:0; top:0; width:4px; height:100%; background:linear-gradient(180deg,var(--teal),var(--violet)); }
[data-testid="stMetricLabel"] { color:#718096; font-weight:600; }
[data-testid="stMetricValue"] { color:var(--ink); font-family:'Manrope',sans-serif; font-size:1.8rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:20px; border-color:rgba(223,232,241,.9); box-shadow:0 10px 30px rgba(16,36,58,.045); background:rgba(255,255,255,.92); transition:box-shadow .2s ease,transform .2s ease; }
div.stButton > button[kind="primary"] { border:0; border-radius:12px; background:linear-gradient(110deg,#078d83,#12b7a5 55%,#497fe8); font-weight:750; box-shadow:0 8px 20px rgba(15,159,146,.24); transition:transform .18s ease,box-shadow .18s ease; }
div.stButton > button[kind="primary"]:hover { transform:translateY(-2px); box-shadow:0 12px 26px rgba(15,159,146,.31); }
div.stButton > button { border-radius:12px; font-weight:650; border-color:#dce5ee; transition:transform .18s ease,border-color .18s ease; }
div.stButton > button:hover { transform:translateY(-1px); border-color:#7dcac1; }
[data-testid="stDataFrame"] { border-radius:15px; overflow:hidden; border:1px solid #e3eaf1; }
.block-container { padding-top:2.1rem; padding-bottom:3rem; max-width:1500px; }
[data-testid="stFileUploader"] section { background:linear-gradient(145deg,#fff,#f3fbfa); border:1.5px dashed #a9d2cc; border-radius:18px; transition:all .2s ease; }
[data-testid="stFileUploader"] section:hover { border-color:#0f9f92; background:#edfaf7; box-shadow:0 8px 24px rgba(15,159,146,.1); }
[data-testid="stFileUploader"] button { border-radius:10px !important; font-weight:700 !important; }
[data-testid="stFileUploader"] small { color:#718096; }
[data-testid="stTabs"] button { font-weight:650; }
div[data-testid="stAlert"] { border-radius:14px; }
.source-card { border-radius:20px; padding:20px 23px; background:linear-gradient(110deg,rgba(255,255,255,.98),rgba(236,249,247,.96) 72%,rgba(243,241,255,.92)); border:1px solid #e1e9f0; box-shadow:0 12px 28px rgba(21,53,79,.06); }
.upload-hero { position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between; gap:24px; border-radius:24px; padding:25px 30px; color:#fff; background:radial-gradient(ellipse at 92% 0%,rgba(56,216,196,.35),transparent 27%),linear-gradient(112deg,#10243a 0%,#17455a 57%,#176b6b 100%); box-shadow:0 18px 38px rgba(16,36,58,.14); }
.upload-hero-copy { position:relative; z-index:1; }
.upload-hero-kicker { color:#a9eee3; text-transform:uppercase; letter-spacing:.14em; font-size:.7rem; font-weight:800; margin-bottom:7px; }
.upload-hero-title { color:#fff; font:800 1.55rem/1.15 'Manrope',sans-serif; letter-spacing:-.04em; margin:0 0 7px; }
.upload-hero-text { color:#d4e4ec; font-size:.91rem; max-width:760px; }
.upload-chip { flex:none; border-radius:999px; padding:9px 13px; color:#eafffb; border:1px solid rgba(255,255,255,.24); background:rgba(255,255,255,.09); font-size:.76rem; font-weight:700; }
.upload-card-title { margin:0; color:#142a40; font:750 1.08rem/1.2 'Manrope',sans-serif; }
.upload-card-copy { color:#7b8999; font-size:.82rem; margin:4px 0 13px; }
.step-chip { display:inline-block; color:#078d83; background:#e8f8f5; border:1px solid #d4f0eb; border-radius:999px; font-size:.68rem; font-weight:800; letter-spacing:.08em; padding:5px 9px; text-transform:uppercase; margin-bottom:12px; }
.mini-kicker { color:#0f8f83; font-size:.72rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; }
.stMarkdown p { line-height:1.55; }
.hero { position:relative; overflow:hidden; padding:38px 42px; border-radius:26px; color:#f8fbff; background:radial-gradient(ellipse at 88% 12%,rgba(75,222,204,.32),transparent 31%),radial-gradient(ellipse at 72% 115%,rgba(118,102,245,.34),transparent 37%),linear-gradient(115deg,#0c1c31 0%,#173d55 54%,#117d79 100%); box-shadow:0 22px 50px rgba(16,38,62,.2); margin:4px 0 24px; }
.hero:after { content:""; pointer-events:none; position:absolute; width:230px; height:230px; border:1px solid rgba(255,255,255,.11); border-radius:50%; right:6%; top:-128px; box-shadow:0 0 0 22px rgba(255,255,255,.025),0 0 0 48px rgba(255,255,255,.02); }
.hero-kicker { color:#9de4d9; text-transform:uppercase; letter-spacing:.13em; font-size:.76rem; font-weight:700; margin-bottom:9px; }
.hero-title { position:relative; z-index:1; font-family:'Manrope',sans-serif; font-weight:800; letter-spacing:-.045em; font-size:2.55rem; line-height:1.08; color:#fff; margin:0 0 12px; text-shadow:0 2px 16px rgba(0,0,0,.12); }
.hero-copy { max-width:820px; color:#d1e1ed; font-size:1rem; line-height:1.55; }
.hero-tags { display:flex; flex-wrap:wrap; gap:9px; margin-top:19px; }
.hero-tag { border:1px solid rgba(255,255,255,.24); color:#e8fbf7; background:rgba(255,255,255,.1); backdrop-filter:blur(8px); border-radius:999px; padding:7px 12px; font-size:.78rem; font-weight:650; }
.section-eyebrow { color:#0f8f83; font-size:.74rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase; }
@media (max-width: 700px) { .hero {padding:25px;} .hero-title {font-size:1.8rem;} .upload-hero {align-items:flex-start; flex-direction:column; padding:22px;} .upload-chip {align-self:flex-start;} }
</style>
""", unsafe_allow_html=True)

def init_state():
    # Clear legacy built-in samples and fitted artifacts when upgrading an existing session.
    if st.session_state.get("data_mode_version") != 4:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.data_mode_version=4
    defaults={"frame":None,"images":[],"image_labels":[],"image_names":[],"target":None,
              "defect_value":None,"timestamp":None,"id_column":None,"tabular_result":None,
              "image_result":None,"fusion_result":None,"last_summary":None,"importance_cache":{}}
    for key,value in defaults.items(): st.session_state.setdefault(key,value)

def safe_run(label, fn):
    try:
        with st.spinner(label): return fn()
    except Exception as exc:
        st.error(f"{label} could not finish: {exc}")
        return None

def title(text, caption=None):
    st.title(text)
    if caption: st.caption(caption)

def render_home():
    frame=st.session_state.frame
    images=st.session_state.images
    image_labels=st.session_state.image_labels
    st.markdown("""
    <section class="hero">
      <div class="hero-kicker">Upload-first quality intelligence</div>
      <div class="hero-title">Analyze your production data<br>and inspection images.</div>
      <div class="hero-copy">Upload your process table, labeled inspection images, or both. The analysis uses only the files you provide in this session.</div>
      <div class="hero-tags"><span class="hero-tag">Your process data</span><span class="hero-tag">Your inspection images</span><span class="hero-tag">Session-only uploads</span></div>
    </section>
    """,unsafe_allow_html=True)
    if frame is None and not images:
        st.info("No data is loaded yet. Open **Data sources** and upload a process table, labeled images, or both.")
        return
    target=st.session_state.target
    sensor_count=0
    if frame is not None:
        defect_rate=float((frame[target]==st.session_state.defect_value).mean()*100) if target in frame else 0.
        pr_values=[float(value.get("pr_auc",np.nan)) for value in (st.session_state.tabular_result or {}).get("models",{}).values()]
        best_pr=max((value for value in pr_values if np.isfinite(value)),default=0.)
        sensor_count=frame.drop(columns=[target,st.session_state.timestamp,st.session_state.id_column],errors="ignore").select_dtypes(include=np.number).shape[1]
    with st.container(horizontal=True):
        if frame is not None:
            st.metric("Process records",f"{len(frame):,}","Uploaded dataset",border=True)
            st.metric("Numeric inputs",f"{sensor_count:,}","Available process features",border=True)
            st.metric("Observed fail rate",f"{defect_rate:.1f}%","From uploaded labels",border=True)
            st.metric("Best model PR-AUC",f"{best_pr:.3f}" if best_pr else "—","Current dataset" if best_pr else "Train models to calculate",border=True)
        if images:
            st.metric("Inspection images",f"{len(images):,}",f"{len(set(image_labels))} uploaded classes",border=True)
    st.space("medium")
    overview,workflow=st.columns([1.35,1],gap="large")
    with overview:
        st.subheader("Active datasets")
        with st.container(border=True):
            d1,d2=st.columns(2)
            with d1:
                st.markdown("**:material/settings: Production signals**")
                st.caption(f"Process table · {len(frame):,} rows · {sensor_count:,} numeric features" if frame is not None else "Upload a labeled process table to train process models.")
                st.badge("Uploaded" if frame is not None else "Waiting for upload",icon=":material/check_circle:" if frame is not None else ":material/pending:",color="green" if frame is not None else "orange")
            with d2:
                st.markdown("**:material/center_focus_strong: Surface inspection**")
                st.caption(f"Uploaded image set · {len(images)} images · {len(set(image_labels))} label groups" if images else "Upload labeled images to train the image model.")
                st.badge("Uploaded" if images else "Waiting for upload",icon=":material/check_circle:" if images else ":material/pending:",color="green" if images else "orange")
        if images:
            st.subheader("Uploaded image preview")
            preview_indices=np.linspace(0,len(images)-1,min(6,len(images)),dtype=int)
            tiles=st.columns(6,wrap=True)
            for column,index in zip(tiles,preview_indices):
                column.image(images[int(index)],caption=image_labels[int(index)],width="stretch")
        if frame is not None:
            with st.expander("Preview process records"):
                st.dataframe(frame.head(8),hide_index=True,width="stretch")
    with workflow:
        st.subheader("Workspace status")
        for label,ready,detail in [
            ("Process model",st.session_state.tabular_result is not None,"Metrics ready to explore" if st.session_state.tabular_result else "Upload process data and train"),
            ("Image model",st.session_state.image_result is not None,"Evaluation metrics ready" if st.session_state.image_result else "Upload labeled images and train"),
            ("Real paired fusion",st.session_state.fusion_result is not None,"Real row-to-image evaluation ready" if st.session_state.fusion_result else "Requires process/image pairs and a map"),
            ("Power BI exports",st.session_state.tabular_result is not None or st.session_state.image_result is not None,"Available model exports" if st.session_state.tabular_result or st.session_state.image_result else "Available after model training"),
        ]:
            with st.container(border=True,horizontal=True,vertical_alignment="center"):
                st.badge(label,icon=":material/check_circle:" if ready else ":material/pending:",color="green" if ready else "orange")
                st.caption(detail)
    st.caption("Only files uploaded in this session are analyzed. Uploaded files are not saved to project storage.")

def reset_derived_results(*, tabular=False, images=False):
    if tabular:
        st.session_state.tabular_result=None
        st.session_state.importance_cache={}
    if images: st.session_state.image_result=None
    st.session_state.fusion_result=None
    st.session_state.last_summary=None

def render_sources():
    title("Data sources", "Upload the dataset you want the models to analyze.")
    st.markdown('<div class="upload-hero"><div class="upload-hero-copy"><div class="upload-hero-kicker">Session-only data intake</div><div class="upload-hero-title">Your files drive every analysis.</div><div class="upload-hero-text">Choose files from your computer. Imports are held in this active app session and are not written to the project dataset folders.</div></div><div class="upload-chip">CSV · XLSX · JPG · ZIP</div></div>',unsafe_allow_html=True)
    st.space("medium")
    st.markdown('<div class="step-chip">Step 1 · Add files</div>',unsafe_allow_html=True)
    left,right=st.columns([1,1],gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="upload-card-title">Production process data</div><div class="upload-card-copy">Sensor readings, inspection results, or SECOM tables</div>',unsafe_allow_html=True)
            process_file=st.file_uploader("CSV, Excel, or raw SECOM sensor table",type=["csv","xlsx","xls","data","txt"],key="upload_process",help="Use CSV/Excel directly, or pair raw whitespace-separated SECOM data with its labels file.")
            labels_file=st.file_uploader("Optional SECOM labels file",type=["data","txt"],key="upload_labels",help="Use with raw whitespace-separated SECOM sensor data.")
        parsed=None
        if process_file is not None:
            try:
                process_buffer=io.BytesIO(process_file.getvalue()); process_buffer.name=process_file.name
                label_buffer=io.BytesIO(labels_file.getvalue()) if labels_file else None
                if label_buffer is not None: label_buffer.name=labels_file.name
                parsed=read_tabular_upload(process_buffer,label_buffer)
                st.caption(f"Detected {len(parsed):,} rows × {len(parsed.columns):,} columns")
            except Exception as exc: st.error(f"Could not read this process file: {exc}")
    with right:
        with st.container(border=True):
            st.markdown('<div class="upload-card-title">Surface inspection images</div><div class="upload-card-copy">Upload your images as labeled files or a class-folder ZIP</div>',unsafe_allow_html=True)
            image_files=st.file_uploader("Images or one ZIP of class folders",type=["png","jpg","jpeg","bmp","zip"],accept_multiple_files=True,key="upload_images",help="For training, organize images in class-named folders inside a ZIP.")
            st.caption("Tip: put labeled images in separate class folders, then upload one ZIP.")
        image_preview=None
        if image_files:
            try:
                image_preview=load_image_uploads(image_files)
                image_counts=pd.Series(image_preview[1],dtype="string").value_counts()
                st.caption(f"Detected {len(image_preview[0]):,} images · {len(image_counts)} label groups")
                if len(image_counts)<2:
                    st.warning("Only one image label was detected. Put each class in its own folder inside a ZIP (for example, `pass/` and `defect/`) to train an image model.")
                elif image_counts.min()<4:
                    st.warning("At least four images are needed in each class to train the image model.")
            except Exception as exc: st.error(f"Could not read these images: {exc}")
    target=timestamp=row_id=defect_value=None
    if parsed is not None:
        st.markdown('<div class="step-chip">Step 2 · Map your target</div>',unsafe_allow_html=True)
        columns=list(parsed.columns)
        target_names={"target","label","class","defect","fail","outcome","y"}
        target_default=next((i for i,name in enumerate(columns) if str(name).strip().lower() in target_names),len(columns)-1)
        a,b,c=st.columns(3)
        target=a.selectbox("Pass/fail target",columns,index=target_default,key="upload_target")
        timestamp=b.selectbox("Timestamp (optional)",["None",*columns],index=(columns.index("timestamp")+1 if "timestamp" in columns else 0),key="upload_timestamp")
        row_id=c.selectbox("Row ID (optional)",["None",*columns],index=(columns.index("row_id")+1 if "row_id" in columns else 0),key="upload_row_id")
        values=list(pd.Series(parsed[target].dropna().unique()).tolist())
        if len(values)>=2:
            default_fail=1 if 1 in values else (values[-1])
            defect_value=st.selectbox("Which target value means defect / fail?",values,index=values.index(default_fail),key="upload_defect_value")
        else:
            defect_value=values[0] if values else None
            st.warning("The selected target needs at least two distinct values for classification.")
    st.space("small")
    if st.button("Apply uploaded data",type="primary",icon=":material/bolt:"):
        if process_file is None and image_preview is None:
            st.info("Choose at least one file to apply.")
        elif process_file is not None and parsed is None:
            st.error("Fix the process file parsing issue above before applying.")
        else:
            if parsed is not None:
                st.session_state.frame=parsed.copy()
                st.session_state.target=target
                st.session_state.defect_value=defect_value
                st.session_state.timestamp=None if timestamp=="None" else timestamp
                st.session_state.id_column=None if row_id=="None" else row_id
                reset_derived_results(tabular=True)
            if image_preview is not None:
                st.session_state.images,st.session_state.image_labels,st.session_state.image_names=image_preview
                reset_derived_results(images=True)
            st.success("Session data updated. Train or score it from the relevant pages.")
            st.rerun()

def render_eda():
    title("Explore the data")
    frame=st.session_state.frame
    if frame is None:
        if st.session_state.image_labels:
            st.info("No process table is uploaded. Image class counts are shown below.")
            st.subheader("Uploaded image class distribution")
            st.bar_chart(pd.Series(st.session_state.image_labels).value_counts())
            st.caption(f"{len(st.session_state.images):,} uploaded images across {len(set(st.session_state.image_labels))} classes.")
        else:
            st.info("Upload a process dataset or labeled image collection on the Data sources page first.")
        return
    target=st.session_state.target
    l,r=st.columns(2)
    with l:
        st.subheader("Missingness")
        miss=frame.isna().mean().sort_values(ascending=False).head(30)
        st.bar_chart(miss)
        st.subheader("Class balance")
        if target in frame: st.bar_chart(frame[target].value_counts(dropna=False))
        st.subheader("Highest-variance numeric features")
        variance=frame.select_dtypes(include=np.number).drop(columns=[target],errors="ignore").var().sort_values(ascending=False).head(20)
        st.bar_chart(variance)
    with r:
        st.subheader("Feature correlation")
        numeric=frame.select_dtypes(include=np.number).drop(columns=[target],errors="ignore")
        if not numeric.empty:
            corr=numeric.iloc[:,:min(30,numeric.shape[1])].corr()
            fig,ax=plt.subplots(figsize=(8,6)); ax.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1); ax.set_title("First 30 numeric features"); st.pyplot(fig); plt.close(fig)
        timestamp=st.session_state.timestamp
        if timestamp and timestamp in frame:
            st.subheader("Target rate over time")
            series=frame[[timestamp,target]].copy(); series[timestamp]=pd.to_datetime(series[timestamp],errors="coerce")
            series=series.dropna().sort_values(timestamp).set_index(timestamp)
            if not series.empty:
                positive=(series[target]==st.session_state.defect_value).astype(int).resample("ME").mean()
                st.line_chart(positive)
    if st.session_state.image_labels:
        st.subheader("Image class distribution")
        st.bar_chart(pd.Series(st.session_state.image_labels).value_counts())
    st.caption("Drift and correlations are descriptive. They do not establish causal effects.")

@st.fragment
def render_tabular():
    title("Process model", "Train and compare tabular classifiers with stratified holdout evaluation.")
    frame=st.session_state.frame
    if frame is None: return st.info("Upload a process dataset on the Data sources page first.")
    a,b,c=st.columns(3)
    miss=a.number_input("Cost of missed defect",min_value=0.,value=1000.,step=100.)
    false=b.number_input("Cost of false alarm / rework",min_value=0.,value=100.,step=25.)
    size=c.slider("Test split",.15,.4,.25,.05)
    if st.button("Train tabular models",type="primary"):
        result=safe_run("Training two process models (one calibration fit and one final fit each)",lambda:train_tabular(frame,st.session_state.target,st.session_state.defect_value,st.session_state.timestamp,st.session_state.id_column,test_size=size,missed_cost=miss,false_alarm_cost=false,quick=True))
        if result:
            st.session_state.tabular_result=result
            st.session_state.importance_cache={}
    result=st.session_state.tabular_result
    if not result: return st.info("Train models to see held-out metrics.")
    rows=[]
    for name,m in result["models"].items(): rows.append({"model":name,**{k:round(m[k],4) for k in ["precision","recall","f1","pr_auc","roc_auc","threshold"]}})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    name=st.selectbox("Inspect model",list(result["models"]))
    m=result["models"][name]
    x,y=st.columns(2)
    with x:
        st.subheader("Confusion matrix [pass, fail]")
        cm=pd.DataFrame(m["confusion_matrix"],index=["Actual pass","Actual fail"],columns=["Pred pass","Pred fail"])
        st.dataframe(cm,width="stretch")
        fpr,tpr,_=m["roc_curve"]
        fig,ax=plt.subplots(); ax.plot(fpr,tpr,label=f"ROC AUC {m['roc_auc']:.3f}"); ax.plot([0,1],[0,1],"--",color="gray"); ax.set(xlabel="False positive rate",ylabel="True positive rate"); ax.legend(); st.pyplot(fig); plt.close(fig)
    with y:
        st.subheader("Precision-recall curve")
        precision,recall,_=m["pr_curve"]
        fig,ax=plt.subplots(); ax.plot(recall,precision,label=f"PR AUC {m['pr_auc']:.3f}"); ax.set(xlabel="Recall",ylabel="Precision"); ax.legend(); st.pyplot(fig); plt.close(fig)
        st.json(m["report"])
    st.session_state.last_summary={"y_test":result["y_test"],"predictions":result["predictions"]}

def render_images():
    title("Image inspection", "Train on uploaded labeled images or score an uploaded image.")
    if not st.session_state.images: return st.info("No images are uploaded. Add labeled images on the Data sources page.")
    class_counts=pd.Series(st.session_state.image_labels,dtype="string").value_counts()
    st.caption(f"Uploaded image set · {len(st.session_state.images):,} images · {len(class_counts)} label groups · HOG + calibrated SVM")
    if len(class_counts)<2:
        st.warning("Only one image label was detected. Upload a ZIP with one class-named folder per label (for example, `pass/` and `defect/`) so the model can compare classes.")
        return
    if class_counts.min()<4:
        st.warning("The image model needs at least four images in every class. Current counts: "+", ".join(f"{name}: {count}" for name,count in class_counts.items()))
        return
    if st.button("Train image baseline",icon=":material/refresh:"):
        result=safe_run("Training HOG + SVM on uploaded images",lambda:train_images(st.session_state.images,st.session_state.image_labels))
        if result:
            st.session_state.image_result=result
    result=st.session_state.image_result
    if result:
        report=result["report"]
        macro=report.get("macro avg",{})
        with st.container(horizontal=True):
            st.metric("Sample holdout accuracy",f"{report.get('accuracy',0):.1%}",border=True)
            st.metric("Macro F1",f"{macro.get('f1-score',0):.3f}",border=True)
            st.metric("Defect classes",len(result["classes"]),"Uploaded labels",border=True)
        left,right=st.columns([1.15,1],gap="large")
        with left:
            st.subheader("Class-level performance")
            report_frame=pd.DataFrame({key:value for key,value in report.items() if isinstance(value,dict)}).T
            st.dataframe(report_frame,width="stretch")
            st.subheader("Confusion matrix")
            st.dataframe(pd.DataFrame(result["confusion_matrix"],index=result["classes"],columns=result["classes"]),width="stretch")
        with right:
            st.subheader("Score an image")
            image_index=st.slider("Sample image",0,len(st.session_state.images)-1,0)
            st.image(st.session_state.images[image_index],caption=f"Reference class: {st.session_state.image_labels[image_index]}",width="stretch")
            prediction=predict_images(result["model"],[st.session_state.images[image_index]],[st.session_state.image_names[image_index]])
            st.success(f"Predicted {prediction.predicted_class.iloc[0]} · {prediction.confidence.iloc[0]:.0%} confidence")
        st.caption("Holdout scores describe only this labeled dataset and split. They may not generalize to other cameras, materials, or lighting conditions.")
    else:
        st.info("No image model is trained in this session yet. Upload at least four labeled images per class, then train the baseline.")

def render_fusion():
    title("Fusion lab", "Evaluate combined signals only when you have real process-to-image pairs.")
    frame=st.session_state.frame; images=st.session_state.images
    st.markdown("Upload a pairing map that links each process row to the image captured for that same item. The app will reject missing or unmatched pairs. It will not create synthetic matches.")
    st.download_button("Download pairing-map template",pd.DataFrame(columns=["row_id","image_filename"]).to_csv(index=False),"pairing_map_template.csv","text/csv",icon=":material/download:")
    if frame is None or not images:
        st.info("Upload a process table and labeled images on the Data sources page first.")
        return
    if len(set(st.session_state.image_names))!=len(st.session_state.image_names):
        st.error("Some images have duplicate file names. Rename them so each image has a unique filename before creating the pairing map.")
        return
    pair_file=st.file_uploader("Real process-row / image pairing map",type=["csv"],key="real_pair_map",help="CSV must contain row_id and image_filename columns.")
    id_col=st.session_state.id_column
    if id_col and id_col in frame:
        st.caption(f"Use values from the process table’s `{id_col}` column in `row_id`. For index-based data, use the zero-based row number.")
    st.caption("`image_filename` must exactly match one uploaded image filename. Each row and image can appear only once.")
    if st.button("Evaluate real paired fusion",type="primary",icon=":material/analytics:"):
        if pair_file is None:
            st.error("Choose a pairing-map CSV first.")
        else:
            try:
                pair_map=pd.read_csv(pair_file,dtype={"row_id":"string","image_filename":"string"})
                result=safe_run("Evaluating real process/image pairs",lambda:train_real_pairing(
                    frame,st.session_state.target,images,st.session_state.image_names,pair_map,
                    id_column=id_col,defect_value=st.session_state.defect_value,
                    timestamp=st.session_state.timestamp))
                if result:
                    st.session_state.fusion_result=result
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not read the pairing-map CSV: {exc}")
    result=st.session_state.fusion_result
    if result:
        st.subheader(f"Evaluation results · {len(result['row_ids']):,} real matched items")
        metrics=pd.DataFrame(result["metrics"]).T
        numeric=metrics.select_dtypes(include=np.number).columns
        st.dataframe(metrics.style.format({col:"{:.3f}" for col in numeric}),width="stretch")
        st.caption("Metrics use a stratified holdout split of the explicitly paired records. Validate the split against production lot/time groups before operational use.")
        st.download_button("Download matched evaluation pairs",result["mapping"].to_csv(index=False),"matched_pairs.csv","text/csv")

def render_root_cause():
    title("Root cause analysis", "Model attribution ranks associations; SECOM sensors are anonymized.")
    result=st.session_state.tabular_result
    if not result: return st.info("Train a process model first.")
    model_names=list(result["trained"])
    default_model=model_names.index("Random forest") if "Random forest" in model_names else 0
    model=st.selectbox("Importance model",model_names,index=default_model)
    cache=st.session_state.importance_cache
    if model not in cache:
        if st.button("Calculate held-out feature importance",icon=":material/analytics:"):
            cache[model]=safe_run("Computing held-out permutation importance",lambda:permutation_importances(result,model))
        else:
            st.info("Feature ranking can take longer for wide tables. Calculate it when you’re ready to inspect the results.")
            return
    importance=cache.get(model)
    if importance is None: return
    st.bar_chart(importance.head(20).set_index("feature")["importance_mean"])
    st.dataframe(importance.head(50),hide_index=True,width="stretch")
    frame=st.session_state.frame; target=st.session_state.target; value=st.session_state.defect_value
    top=importance.feature.head(5).tolist()
    st.subheader("Pass and fail distributions")
    for feature in top:
        st.write(feature)
        grouped=frame[[target,feature]].copy()
        grouped["_group"]=np.where(grouped[target]==value,"Fail","Pass")
        st.dataframe(grouped.groupby("_group")[feature].quantile([.1,.5,.9]).unstack().rename(columns={.1:"p10",.5:"median",.9:"p90"}),width="stretch")
    st.warning("Correlation is not causation. Sensor names are anonymized; ranges are descriptive 10th–90th percentiles, not setpoints.")
    st.subheader("Domain knowledge (not model output)")
    st.dataframe(pd.DataFrame([
        ["crazing","Overheating, insufficient cleaning"],["inclusion","Entrained debris, roller contamination"],["patches","Adhered filings or scale"],["pitted_surface","Cooling water / cooling control"],["rolled-in_scale","Oxide and rough work-roll contact"],["scratches","Handling or roller contact"]],columns=["defect class","likely process causes (domain knowledge)"]),hide_index=True,width="stretch")

def render_cost():
    title("Cost and savings", "All monetary inputs are assumptions; adjust them to match your line economics.")
    result=st.session_state.tabular_result
    if not result: return st.info("Train the process model first.")
    a,b,c,d=st.columns(4)
    miss=a.number_input("Escape / missed defect cost",min_value=0.,value=1000.,step=100.,key="cost_miss")
    false=b.number_input("False alarm / rework cost",min_value=0.,value=100.,step=25.,key="cost_fp")
    inspect=c.number_input("Inspection cost per part",min_value=0.,value=2.,step=1.)
    volume=d.number_input("Production volume",min_value=1,value=10000,step=1000)
    model_names=list(result["predictions"])
    if st.session_state.fusion_result and "predictions" in st.session_state.fusion_result:
        model_names += [f"Fusion: {n}" for n in st.session_state.fusion_result["predictions"]]
    name=st.selectbox("Model probabilities",model_names)
    if name.startswith("Fusion: "):
        fusion=st.session_state.fusion_result; fusion_name=name.removeprefix("Fusion: ")
        y=np.asarray(fusion["y_test"]); p=np.asarray(fusion["predictions"][fusion_name]); training_threshold=.5
    else:
        pred=result["predictions"][name]; y=pred.actual.to_numpy(); p=pred.probability.to_numpy()
        training_threshold=float(result["models"][name]["threshold"])
    curve=cost_curve(y,p,missed_cost=miss,false_alarm_cost=false,inspection_cost=inspect)
    best=curve.loc[curve.copq.idxmin()]
    threshold=st.slider("Live decision threshold",0.,1.,training_threshold,.01)
    selected=curve.loc[(curve.threshold-threshold).abs().idxmin()]
    with st.container(horizontal=True):
        st.metric("Training-selected threshold",f"{training_threshold:.2f}",border=True)
        st.metric("Selected threshold COPQ",f"${selected.copq*volume/len(y):,.0f}",border=True)
        st.metric("Holdout-min threshold (diagnostic)",f"{best.threshold:.2f}",border=True)
        baseline=inspect*len(y)
        savings=(baseline-selected.copq)*volume/len(y)
        st.metric("Net savings vs 100% manual inspection",f"${savings:,.0f}",border=True)
        st.metric("ROI",f"{(baseline-selected.copq)/max(selected.copq,1e-9)*100:.1f}%",border=True)
    st.line_chart(curve.set_index("threshold")["copq"])
    st.caption("Baseline assumes every part receives manual inspection at the declared unit cost. Model COPQ counts false negatives, false alarms and inspection cost for predicted defects. Estimates are scaled from holdout data.")
    st.subheader("Cost ratio sensitivity")
    oof=result["calibration_predictions"].get(name) if not name.startswith("Fusion: ") else None
    if oof is not None: sens=sensitivity(oof.actual.to_numpy(),oof.probability.to_numpy())
    else: sens=sensitivity(y,p)
    heat=sens.pivot(index="missed_defect_cost",columns="false_alarm_cost",values="minimum_cost")
    fig,ax=plt.subplots(figsize=(7,4)); im=ax.imshow(heat.to_numpy(),aspect="auto",cmap="YlOrRd"); ax.set(xticks=range(len(heat.columns)),xticklabels=heat.columns,yticks=range(len(heat.index)),yticklabels=heat.index,xlabel="False alarm cost",ylabel="Missed defect cost"); fig.colorbar(im,ax=ax,label="Minimum holdout COPQ"); st.pyplot(fig); plt.close(fig)
    st.dataframe(heat,width="stretch")

def render_predict():
    title("Predictions", "Review predictions from models trained on your uploaded data.")
    result=st.session_state.tabular_result
    if not result: return st.info("Upload a process dataset and train a model first.")
    a,b=st.columns([1,2])
    with a: name=st.selectbox("Model trained on uploaded data",list(result["predictions"]))
    predictions=result["predictions"][name].copy()
    default=float(result["models"][name]["threshold"])
    with b: threshold=st.slider("Decision threshold",0.,1.,default,.01)
    predictions["predicted"]= (predictions.probability>=threshold).astype(int)
    predictions=predictions.rename(columns={"row_index":"sample row","actual":"actual label","probability":"defect probability","predicted":"predicted defect"})
    predictions["risk band"]=pd.cut(predictions["defect probability"],[-.001,.25,.6,1.001],labels=["Low","Watch","High"])
    fail_count=int(predictions["predicted defect"].sum())
    with st.container(horizontal=True):
        st.metric("Rows scored",f"{len(predictions):,}","Held-out uploaded records",border=True)
        st.metric("Flagged at threshold",f"{fail_count:,}",f"{fail_count/max(1,len(predictions)):.1%} of sample",border=True)
        st.metric("Model threshold",f"{default:.3f}","Selected on training folds",border=True)
    st.dataframe(predictions.sort_values("defect probability",ascending=False),hide_index=True,width="stretch",height=440,
                 column_config={"defect probability":st.column_config.ProgressColumn("Defect probability",min_value=0,max_value=1,format="percent",color="auto")})
    st.download_button("Download predictions",predictions.to_csv(index=False).encode(),f"{name.lower().replace(' ','_')}_predictions.csv","text/csv",icon=":material/download:")
    st.caption("These predictions come from the dataset uploaded in this session.")

def render_export():
    title("Power BI exports", "Review your trained results, then download clean tables for Power BI.")
    result=st.session_state.tabular_result
    if not result:
        image_result=st.session_state.image_result
        if image_result is None:
            st.info("Train a process or image model first. Its session results will appear here for download.")
            return
        image_metrics=pd.DataFrame(image_result["report"]).T.rename_axis("class_or_metric").reset_index()
        image_predictions=image_result["predictions"].copy()
        image_predictions.insert(0,"filename",[st.session_state.image_names[i] for i in image_result["test_indices"]])
        for j,label in enumerate(image_result["classes"]):
            image_predictions[f"probability_{label}"]=image_result["test_probabilities"][:,j]
        image_confusion=pd.DataFrame(image_result["confusion_matrix"],index=image_result["classes"],columns=image_result["classes"])
        st.subheader("Uploaded image model exports")
        st.metric("Image holdout accuracy",f"{image_result['report'].get('accuracy',float('nan')):.1%}")
        st.dataframe(image_metrics,hide_index=True,width="stretch")
        st.dataframe(image_predictions.head(100),hide_index=True,width="stretch")
        st.dataframe(image_confusion,width="stretch")
        for label,frame,name in [
            ("Image metrics",image_metrics,"image_metrics_by_class.csv"),
            ("Image predictions",image_predictions,"image_predictions.csv"),
            ("Image confusion matrix",image_confusion,"image_confusion_matrix.csv"),
        ]:
            st.download_button(f"Download {label}",frame.to_csv(index=True if name.endswith("confusion_matrix.csv") else False).encode(),name,"text/csv",key=f"image_export_{name}")
        return
    metrics=pd.DataFrame([{"model":name,**{k:v for k,v in m.items() if isinstance(v,(int,float,np.number))}} for name,m in result["models"].items()])
    predictions=pd.concat([df.assign(model=name) for name,df in result["predictions"].items()],ignore_index=True)
    importance=st.session_state.importance_cache.get("Random forest")
    if importance is None and st.button("Calculate feature importance for this upload",icon=":material/analytics:"):
        importance=safe_run("Calculating held-out feature importance",lambda:permutation_importances(result))
        if importance is not None:
            st.session_state.importance_cache["Random forest"]=importance
            st.rerun()
    if importance is None:
        st.info("Feature importance is available on demand. Calculate it after reviewing the other exports; this can take longer on wide process tables.")
    cost_df=pd.DataFrame()
    curves=[]
    if len(predictions):
        cost_df=compare_scenarios(result["y_test"],result["predictions"],{"missed_cost":1000,"false_alarm_cost":100,"inspection_cost":2},10000)
        curve=cost_curve(result["predictions"]["Random forest"].actual,result["predictions"]["Random forest"].probability,1000,100,2)
        curve["training_selected_threshold"]=result["models"]["Random forest"]["threshold"]
        curves.append(curve)
        if st.session_state.fusion_result and "predictions" in st.session_state.fusion_result:
            fused=st.session_state.fusion_result
            fusion_predictions={f"Fusion: {name}":pd.DataFrame({"actual":fused["y_test"],"probability":prob,"threshold":.5}) for name,prob in fused["predictions"].items()}
            cost_df=pd.concat([cost_df,compare_scenarios(fused["y_test"],fusion_predictions,{"missed_cost":1000,"false_alarm_cost":100,"inspection_cost":2},10000)],ignore_index=True)
        cost_df=pd.concat([pd.DataFrame([{"scenario":"100% manual inspection","threshold":np.nan,"copq":20000.,"net_savings_vs_manual":0.,"roi":0.,"missed_defects":0,"false_alarms":0}]),cost_df],ignore_index=True)
    oof=result["calibration_predictions"].get("Random forest")
    cost_sensitivity=sensitivity(oof.actual,oof.probability) if oof is not None else pd.DataFrame()
    frame=st.session_state.frame
    if st.session_state.timestamp and st.session_state.timestamp in frame:
        drift=frame[[st.session_state.timestamp,st.session_state.target]].copy()
        drift[st.session_state.timestamp]=pd.to_datetime(drift[st.session_state.timestamp],errors="coerce")
        drift=drift.dropna().sort_values(st.session_state.timestamp).set_index(st.session_state.timestamp)
        drift=drift.assign(defect_rate=(drift[st.session_state.target]==st.session_state.defect_value).astype(float)).resample("ME").agg(defect_rate=("defect_rate","mean"),parts=(st.session_state.target,"size")).reset_index()
        drift=drift.rename(columns={st.session_state.timestamp:"timestamp"})
    else: drift=pd.DataFrame(columns=["timestamp","defect_rate","parts"])
    class_summary=pd.Series(st.session_state.image_labels or [],dtype="string").value_counts().rename_axis("defect_class").reset_index(name="image_count")
    items={"metrics_by_model.csv":metrics,"predictions.csv":predictions,"feature_importance.csv":importance,
           "copq_scenarios.csv":cost_df,"threshold_vs_cost.csv":pd.concat(curves,ignore_index=True) if curves else pd.DataFrame(),
           "cost_sensitivity.csv":cost_sensitivity,"process_drift.csv":drift,"defect_class_summary.csv":class_summary}
    if st.session_state.image_result:
        img=st.session_state.image_result
        items["image_metrics_by_class.csv"]=pd.DataFrame(img["report"]).T.rename_axis("class_or_metric").reset_index()
        image_pred=img["predictions"].copy(); image_pred.insert(0,"filename",[st.session_state.image_names[i] for i in img["test_indices"]])
        for j,label in enumerate(img["classes"]): image_pred[f"probability_{label}"]=img["test_probabilities"][:,j]
        items["image_predictions.csv"]=image_pred
        items["image_confusion_matrix.csv"]=pd.DataFrame(img["confusion_matrix"],index=img["classes"],columns=img["classes"])
    if st.session_state.fusion_result:
        fused=st.session_state.fusion_result
        items["fusion_metrics.csv"]=pd.DataFrame(fused["metrics"]).T.rename_axis("model").reset_index()
        items["fusion_pairing_map.csv"]=fused["mapping"]
    fusion_note="Fusion results use the uploaded real pairing map." if st.session_state.fusion_result else "Fusion results were not calculated because no real process-to-image pairing map was supplied."
    summary="# Defect QC analysis summary\n\n```text\n"+metrics.to_string(index=False)+"\n```\n\n"+fusion_note+" Costs use declared assumptions; validate cost inputs against your production line.\n"
    image_result=st.session_state.image_result
    with st.container(horizontal=True):
        st.metric("Process models",f"{len(metrics)}","Trained and evaluated",border=True)
        st.metric("Process test rows",f"{len(predictions):,}","Across model predictions",border=True)
        pr_scores=pd.to_numeric(metrics["pr_auc"],errors="coerce") if "pr_auc" in metrics else pd.Series(dtype=float)
        if pr_scores.notna().any():
            best_row=metrics.loc[pr_scores.idxmax()]
            st.metric("Best process PR-AUC",f"{best_row.pr_auc:.3f}",str(best_row.model),border=True)
        if image_result:
            image_macro=image_result["report"].get("macro avg",{}).get("f1-score",float("nan"))
            st.metric("Image macro-F1",f"{image_macro:.3f}",f"{len(st.session_state.images):,} images loaded",border=True)
    st.caption("This page previews the results and prepares Power BI-ready CSV files. In Power BI Desktop, import a file with Home → Get data → Text/CSV.")

    overview_tab,cost_tab,prediction_tab,download_tab=st.tabs(["Model overview","Cost & trend","Prediction tables","Downloads"])
    with overview_tab:
        left,right=st.columns([1.1,1])
        with left:
            st.subheader("Process model comparison")
            display_metrics=metrics.copy()
            score_cols=[col for col in ("pr_auc","roc_auc","precision","recall","f1","accuracy") if col in display_metrics]
            if score_cols:
                st.bar_chart(display_metrics.set_index("model")[score_cols[:4]])
            st.dataframe(display_metrics,hide_index=True,width="stretch")
        with right:
            st.subheader("Image model")
            if image_result:
                image_metrics=items.get("image_metrics_by_class.csv",pd.DataFrame())
                st.metric("Holdout accuracy",f"{image_result['report'].get('accuracy',float('nan')):.1%}")
                image_classes=image_metrics[~image_metrics.class_or_metric.astype(str).isin(["accuracy","macro avg","weighted avg"])].copy() if not image_metrics.empty else pd.DataFrame()
                if not image_classes.empty and "f1-score" in image_classes:
                    st.bar_chart(image_classes.set_index("class_or_metric")[["f1-score"]])
                st.dataframe(image_metrics,hide_index=True,width="stretch")
            else:
                st.info("No image model result is loaded. Add labeled images on Data sources and train on Image inspection.")
    with cost_tab:
        st.subheader("Cost scenarios")
        if not cost_df.empty:
            cost_display=cost_df.copy()
            for column in ("copq","net_savings_vs_manual"):
                if column in cost_display:
                    cost_display[column]=cost_display[column].map(lambda value:f"${value:,.0f}" if pd.notna(value) else "—")
            if "roi" in cost_display:
                cost_display["roi"]=cost_display.roi.map(lambda value:f"{value:.1%}" if pd.notna(value) else "—")
            st.dataframe(cost_display,hide_index=True,width="stretch")
        else:
            st.info("Cost scenarios appear after process predictions are available.")
        if curves:
            curve_frame=pd.concat(curves,ignore_index=True)
            st.subheader("Threshold versus cost")
            st.line_chart(curve_frame.set_index("threshold")[["copq"]])
            st.dataframe(curve_frame.nsmallest(10,"copq"),hide_index=True,width="stretch")
        if not drift.empty:
            st.subheader("Monthly defect trend")
            st.line_chart(drift.set_index("timestamp")[["defect_rate"]])
            st.dataframe(drift,hide_index=True,width="stretch")
        else:
            st.info("No usable timestamp column is available for a monthly trend.")
    with prediction_tab:
        st.subheader("Held-out process predictions")
        model_choice=st.selectbox("Prediction model",list(result["predictions"]),key="powerbi_prediction_model")
        prediction_preview=result["predictions"][model_choice].copy()
        st.dataframe(prediction_preview.head(100),hide_index=True,width="stretch",height=360)
        st.caption(f"Showing up to 100 of {len(prediction_preview):,} held-out rows for {model_choice}.")
        if "image_predictions.csv" in items:
            st.subheader("Held-out image predictions")
            st.dataframe(items["image_predictions.csv"].head(100),hide_index=True,width="stretch",height=300)
    with download_tab:
        st.subheader("Power BI-ready files")
        st.write("Import these CSVs into Power BI Desktop using **Home → Get data → Text/CSV**. Each table below is also previewed in the other tabs.")
        download_cols=st.columns(2)
        for index,(filename,df) in enumerate(items.items()):
            if df is not None:
                with download_cols[index%2]:
                    st.download_button(f"Download {filename}",df.to_csv(index=False).encode(),filename,"text/csv",key=f"powerbi_{filename}",width="stretch")
        st.download_button("Download Markdown summary",summary,"summary.md","text/markdown",key="powerbi_summary",width="stretch")

pages=["Data sources","Overview","Data explorer","Process models","Image inspection","Fusion lab","Root cause","Cost & savings","Predictions","Power BI exports"]
page_renderers={"Overview":render_home,"Data sources":render_sources,"Data explorer":render_eda,"Process models":render_tabular,"Image inspection":render_images,"Fusion lab":render_fusion,"Root cause":render_root_cause,"Cost & savings":render_cost,"Predictions":render_predict,"Power BI exports":render_export}

init_state()
with st.sidebar:
    st.title("Defect QC")
    page=st.radio("Navigate",pages,key="navigation_page")
    st.caption("CPU-first quality analytics · fixed seed 42")
    if st.session_state.frame is not None: st.caption(f"Process rows: {len(st.session_state.frame):,}")
    if st.session_state.images: st.caption(f"Images: {len(st.session_state.images):,}")
try:
    page_renderers[page]()
except Exception as exc:
    st.error(f"This page encountered a problem: {exc}")
