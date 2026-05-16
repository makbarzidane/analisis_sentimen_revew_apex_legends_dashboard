import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from sklearn.utils import resample
from nltk.stem import PorterStemmer


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Apex Legends",
    page_icon="🎮",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #777;
        font-size: 16px;
        margin-top: 0px;
    }
    .small-note {
        color: #777;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# FUNGSI PREPROCESSING FINAL
# =========================================================
stemmer = PorterStemmer()

def case_folding(text):
    if pd.isna(text):
        return ""
    return str(text).lower()

def cleansing(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[@#]\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalization(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # De-elongation sederhana, contoh: goooooood -> good
    text = re.sub(r"(.)\1+", r"\1\1", text)

    # Penanganan frasa negasi agar tidak dibaca sebagai kata tunggal good/bad
    text = re.sub(r"\bnot\s+good\b", "not_good", text)
    text = re.sub(r"\bnot\s+bad\b", "not_bad", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenizing(text):
    if pd.isna(text):
        return []
    return str(text).split()

def filtering_optimized(tokens, stop_words):
    return [word for word in tokens if word not in stop_words]

def stem_token(word):
    if word in ["not_good", "not_bad"]:
        return word
    return stemmer.stem(word)

def stemming(tokens):
    return [stem_token(word) for word in tokens]

def rejoining(tokens):
    return " ".join(tokens)

def preprocess_text(text, stop_words):
    step_case = case_folding(text)
    step_clean = cleansing(step_case)
    step_norm = normalization(step_clean)
    tokens = tokenizing(step_norm)
    filtered = filtering_optimized(tokens, stop_words)
    stemmed = stemming(filtered)
    final_text = rejoining(stemmed)

    return {
        "case_folding": step_case,
        "cleansing": step_clean,
        "normalization": step_norm,
        "tokenizing": tokens,
        "filtering": filtered,
        "stemming": stemmed,
        "final_text": final_text
    }


# =========================================================
# LOAD MODEL FINAL
# Dashboard tidak melakukan training ulang agar model tetap stabil.
# =========================================================
@st.cache_resource
def load_model_package():
    model_path = "apex_sentiment_model_final.pkl"

    if not os.path.exists(model_path):
        st.error(
            "File `apex_sentiment_model_final.pkl` belum ditemukan. "
            "Simpan model final dari Colab terlebih dahulu, lalu letakkan file tersebut "
            "di folder yang sama dengan dashboard ini."
        )
        st.stop()

    package = joblib.load(model_path)

    required_keys = ["model", "vectorizer", "stop_words", "label_mapping"]
    missing_keys = [key for key in required_keys if key not in package]

    if missing_keys:
        st.error(f"File model final belum lengkap. Key yang belum ada: {missing_keys}")
        st.stop()

    package["stop_words"] = set(package["stop_words"])
    package["label_mapping"] = {
        int(k): v for k, v in package.get("label_mapping", {0: "Negative", 1: "Positive"}).items()
    }

    return package


# =========================================================
# LOAD DATASET
# =========================================================
@st.cache_data
def load_dataset(stop_words_tuple):
    stop_words = set(stop_words_tuple)

    if os.path.exists("dataset_final_rejoined.csv"):
        df = pd.read_csv("dataset_final_rejoined.csv")
        dataset_source = "dataset_final_rejoined.csv"
    elif os.path.exists("scraping_data_2026.csv"):
        df = pd.read_csv("scraping_data_2026.csv")
        dataset_source = "scraping_data_2026.csv"
        st.warning(
            "File `dataset_final_rejoined.csv` belum ditemukan. "
            "Dashboard memakai `scraping_data_2026.csv` dan membuat final_text otomatis."
        )
    else:
        return None, None

    if "label_encoded" not in df.columns and "label" in df.columns:
        df["label_encoded"] = df["label"].map({"Positive": 1, "Negative": 0})

    if "final_text" not in df.columns and "review" in df.columns:
        df["final_text"] = df["review"].fillna("").apply(
            lambda x: preprocess_text(x, stop_words)["final_text"]
        )

    if "final_text" in df.columns:
        df["final_text"] = df["final_text"].fillna("")

    return df, dataset_source


def make_balanced_data(df):
    df = df.dropna(subset=["label_encoded"]).copy()
    df["label_encoded"] = df["label_encoded"].astype(int)

    df_pos = df[df["label_encoded"] == 1]
    df_neg = df[df["label_encoded"] == 0]

    if len(df_pos) == 0 or len(df_neg) == 0:
        return df

    # Mengikuti notebook final:
    # kelas mayoritas di-undersampling mengikuti jumlah kelas minoritas.
    if len(df_pos) >= len(df_neg):
        df_pos_downsampled = resample(
            df_pos,
            replace=False,
            n_samples=len(df_neg),
            random_state=1
        )
        df_balanced = pd.concat([df_pos_downsampled, df_neg])
    else:
        df_neg_downsampled = resample(
            df_neg,
            replace=False,
            n_samples=len(df_pos),
            random_state=1
        )
        df_balanced = pd.concat([df_pos, df_neg_downsampled])

    return df_balanced


# =========================================================
# HELPER MODEL DAN FILTER
# =========================================================
def apply_dashboard_filters(base_df, selected_labels, keyword):
    filtered_df = base_df.copy()

    if selected_labels:
        filtered_df = filtered_df[filtered_df["label"].isin(selected_labels)]

    if keyword:
        keyword = keyword.strip()
        text_cols = []

        if "review" in filtered_df.columns:
            text_cols.append("review")
        if "final_text" in filtered_df.columns:
            text_cols.append("final_text")

        if text_cols:
            mask = pd.Series(False, index=filtered_df.index)
            for col in text_cols:
                mask = mask | filtered_df[col].astype(str).str.contains(keyword, case=False, na=False)
            filtered_df = filtered_df[mask]

    return filtered_df


def predict_sentiment(text, package):
    model = package["model"]
    vectorizer = package["vectorizer"]
    stop_words = package["stop_words"]
    label_mapping = package["label_mapping"]

    processed = preprocess_text(text, stop_words)
    vectorized_text = vectorizer.transform([processed["final_text"]])

    pred = model.predict(vectorized_text)[0]
    proba = model.predict_proba(vectorized_text)[0]

    class_list = list(model.classes_)
    pred_index = class_list.index(pred)

    label_result = label_mapping.get(int(pred), str(pred))
    confidence = proba[pred_index] * 100

    prob_rows = []
    for i, cls in enumerate(class_list):
        prob_rows.append({
            "Sentimen": label_mapping.get(int(cls), str(cls)),
            "Probabilitas": proba[i]
        })

    return label_result, confidence, pd.DataFrame(prob_rows), processed


def get_top_features(vectorizer, data_df, top_n=20):
    if data_df.empty:
        return pd.DataFrame(columns=["Fitur", "Skor_Bobot"])

    X_tfidf = vectorizer.transform(data_df["final_text"].fillna(""))
    feature_names = vectorizer.get_feature_names_out()
    sum_tfidf = X_tfidf.sum(axis=0).A1

    words_importance = pd.DataFrame({
        "Fitur": feature_names,
        "Skor_Bobot": sum_tfidf
    })

    return words_importance.sort_values(
        by="Skor_Bobot",
        ascending=False
    ).head(top_n)


def get_prediction_samples(package, data_df, n=10):
    if data_df.empty:
        return pd.DataFrame(columns=["Teks_Bersih", "Label_Asli", "Prediksi_Model"])

    model = package["model"]
    vectorizer = package["vectorizer"]
    label_mapping = package["label_mapping"]

    sample_df = data_df.sample(n=min(n, len(data_df)), random_state=1).copy()
    X_sample = vectorizer.transform(sample_df["final_text"].fillna(""))
    preds = model.predict(X_sample)

    result = pd.DataFrame({
        "Teks_Bersih": sample_df["final_text"].values,
        "Label_Asli": sample_df["label_encoded"].map(label_mapping).values,
        "Prediksi_Model": [label_mapping.get(int(p), str(p)) for p in preds]
    })

    return result


def create_wordcloud_figure(data_df):
    if data_df.empty or "final_text" not in data_df.columns:
        return None

    text = " ".join(data_df["final_text"].dropna().astype(str).tolist()).strip()

    if text == "":
        return None

    wc = WordCloud(
        width=1000,
        height=450,
        background_color="white",
        collocations=False,
        max_words=120
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def build_ngram_table(data_df, ngram_n=2, top_n=15):
    if data_df.empty or "final_text" not in data_df.columns:
        return pd.DataFrame(columns=["Fitur", "Jumlah"])

    from collections import Counter

    counter = Counter()

    for text in data_df["final_text"].dropna().astype(str):
        tokens = text.split()
        if len(tokens) < ngram_n:
            continue

        grams = zip(*[tokens[i:] for i in range(ngram_n)])
        counter.update([" ".join(g) for g in grams])

    rows = counter.most_common(top_n)
    return pd.DataFrame(rows, columns=["Fitur", "Jumlah"])


# =========================================================
# MAIN APP
# =========================================================
st.markdown('<p class="main-title">Dashboard Analisis Sentimen Ulasan Apex Legends</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Multinomial Naive Bayes dengan TF-IDF Bigram</p>',
    unsafe_allow_html=True
)

package = load_model_package()
model = package["model"]
vectorizer = package["vectorizer"]
stop_words = package["stop_words"]
label_mapping = package["label_mapping"]

df, dataset_source = load_dataset(tuple(sorted(stop_words)))

if df is None:
    st.error(
        "Dataset belum ditemukan. Pastikan file `dataset_final_rejoined.csv` "
        "atau `scraping_data_2026.csv` berada di folder yang sama dengan dashboard."
    )
    st.stop()

required_cols = {"label", "label_encoded", "final_text"}
missing_cols = required_cols - set(df.columns)

if missing_cols:
    st.error(f"Kolom berikut belum tersedia di dataset: {missing_cols}")
    st.info("Jalankan notebook sampai tahap Rejoining & Label Encoding terlebih dahulu.")
    st.stop()

df_balanced = make_balanced_data(df)

accuracy = package.get("accuracy", None)
report = package.get("classification_report", None)
cm = package.get("confusion_matrix", None)

if report is not None:
    report_df = pd.DataFrame(report).T
else:
    report_df = pd.DataFrame()

if cm is not None:
    cm = np.array(cm)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Pengaturan Dashboard")
    st.caption("Model dimuat dari `apex_sentiment_model_final.pkl`")
    selected_data = st.radio(
        "Data yang ditampilkan",
        ["Dataset Asli", "Dataset Setelah Balancing"],
        index=0,
        help="Mengatur sumber data yang dipakai oleh grafik ringkasan, WordCloud, Top Fitur, dan tabel dataset."
    )

    label_options = sorted(df["label"].dropna().unique().tolist())

    label_filter = st.multiselect(
        "Filter label",
        options=label_options,
        default=label_options,
        help="Memilih sentimen yang ingin ditampilkan. Jika hanya memilih Positive, grafik dan tabel hanya menampilkan data Positive."
    )

    keyword = st.text_input(
        "Cari kata pada review/final_text",
        help="Mencari kata pada kolom review dan final_text. Contoh: server, cheater, not_good, bug."
    )

    st.markdown("---")
    st.write("**Dataset aktif:**")
    st.code(dataset_source if dataset_source else "Tidak ditemukan")

    st.write("**Model aktif:**")
    st.code("apex_sentiment_model_final.pkl")


# =========================================================
# DATA AKTIF SESUAI SIDEBAR
# =========================================================
base_df = df if selected_data == "Dataset Asli" else df_balanced
display_df = apply_dashboard_filters(base_df, label_filter, keyword)

top_20_words = get_top_features(vectorizer, display_df, top_n=20)
test_results = get_prediction_samples(package, display_df, n=10)
bigram_df = build_ngram_table(display_df, ngram_n=2, top_n=15)

# =========================================================
# METRIC CARDS
# =========================================================
total_data = len(display_df)
positive_count = int((display_df["label"] == "Positive").sum()) if not display_df.empty else 0
negative_count = int((display_df["label"] == "Negative").sum()) if not display_df.empty else 0
balanced_count = len(df_balanced)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Data Ditampilkan", f"{total_data:,}")
col2.metric("Positive", f"{positive_count:,}")
col3.metric("Negative", f"{negative_count:,}")
col4.metric("Total Balanced", f"{balanced_count:,}")

if accuracy is not None:
    col5.metric("Akurasi Model", f"{accuracy * 100:.2f}%")
else:
    col5.metric("Akurasi Model", "Tidak tersedia")


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Ringkasan Data",
    "Evaluasi Model",
    "Top Fitur TF-IDF",
    "WordCloud",
    "Prediksi Ulasan Baru",
    "Tabel Dataset"
])


with tab1:
    st.subheader("Ringkasan Data Berdasarkan Filter Sidebar")

    if display_df.empty:
        st.warning("Tidak ada data yang sesuai dengan filter sidebar.")
    else:
        counts_active = display_df["label"].value_counts().reset_index()
        counts_active.columns = ["Sentimen", "Jumlah"]

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("Distribusi Sentimen Data Aktif")
            fig_active = px.bar(
                counts_active,
                x="Sentimen",
                y="Jumlah",
                text="Jumlah",
                title=f"Distribusi Sentimen - {selected_data}"
            )
            fig_active.update_traces(textposition="outside")
            fig_active.update_layout(yaxis_title="Jumlah Ulasan")
            st.plotly_chart(fig_active, use_container_width=True)

        with col_b:
            st.write("Persentase Sentimen")
            fig_pie = px.pie(
                counts_active,
                names="Sentimen",
                values="Jumlah",
                title="Proporsi Sentimen Data Aktif",
                hole=0.35
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### Perbandingan Dataset Asli dan Dataset Balanced")

        counts_original = df["label"].value_counts().reset_index()
        counts_original.columns = ["Sentimen", "Jumlah"]
        counts_original["Dataset"] = "Dataset Asli"

        counts_balanced = df_balanced["label"].value_counts().reset_index()
        counts_balanced.columns = ["Sentimen", "Jumlah"]
        counts_balanced["Dataset"] = "Dataset Setelah Balancing"

        compare_df = pd.concat([counts_original, counts_balanced], ignore_index=True)

        fig_compare = px.bar(
            compare_df,
            x="Sentimen",
            y="Jumlah",
            color="Dataset",
            barmode="group",
            text="Jumlah",
            title="Perbandingan Distribusi Sebelum dan Sesudah Balancing"
        )
        fig_compare.update_traces(textposition="outside")
        st.plotly_chart(fig_compare, use_container_width=True)

        if "date" in display_df.columns:
            st.subheader("Jumlah Ulasan Berdasarkan Tanggal")
            df_date = display_df.copy()
            df_date["date"] = pd.to_datetime(df_date["date"], errors="coerce")
            date_count = df_date.dropna(subset=["date"]).groupby(
                [pd.Grouper(key="date", freq="D"), "label"]
            ).size().reset_index(name="Jumlah")

            if not date_count.empty:
                fig_date = px.line(
                    date_count,
                    x="date",
                    y="Jumlah",
                    color="label",
                    markers=True,
                    title="Tren Ulasan Harian Berdasarkan Sentimen"
                )
                st.plotly_chart(fig_date, use_container_width=True)


with tab2:
    st.subheader("Evaluasi Performa Model Final")

    st.info(
        "Bagian evaluasi ini membaca hasil evaluasi yang tersimpan di `apex_sentiment_model_final.pkl`. "
    )

    if accuracy is not None:
        st.success(f"Akurasi model final: {accuracy * 100:.2f}%")
    else:
        st.warning("Akurasi tidak tersedia di file model final.")

    if not report_df.empty:
        st.write("Classification Report")
        st.dataframe(report_df, use_container_width=True)

        if {"Negative", "Positive"}.issubset(report_df.index):
            metric_plot_df = report_df.loc[["Negative", "Positive"], ["precision", "recall", "f1-score"]].reset_index()
            metric_plot_df = metric_plot_df.rename(columns={"index": "Sentimen"})
            metric_plot_long = metric_plot_df.melt(
                id_vars="Sentimen",
                value_vars=["precision", "recall", "f1-score"],
                var_name="Metrik",
                value_name="Skor"
            )

            fig_metric = px.bar(
                metric_plot_long,
                x="Sentimen",
                y="Skor",
                color="Metrik",
                barmode="group",
                text="Skor",
                title="Perbandingan Precision, Recall, dan F1-Score"
            )
            fig_metric.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_metric.update_yaxes(range=[0, 1.2])
            st.plotly_chart(fig_metric, use_container_width=True)
    else:
        st.info("Classification report belum tersedia di file model final.")

    col_cm, col_pred = st.columns([1, 1])

    with col_cm:
        st.write("Confusion Matrix")

        if cm is not None:
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"]
            )
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Confusion matrix belum tersedia di file model final.")

    with col_pred:
        st.write("Contoh Hasil Prediksi Data Aktif")
        st.caption("Contoh ini mengikuti filter sidebar, tetapi bukan perhitungan ulang akurasi.")
        st.dataframe(test_results, use_container_width=True)


with tab3:
    st.subheader("Top Fitur Berdasarkan TF-IDF pada Data Aktif")

    if top_20_words.empty:
        st.warning("Tidak ada fitur yang dapat ditampilkan karena data aktif kosong.")
    else:
        fig_features = px.bar(
            top_20_words.sort_values("Skor_Bobot"),
            x="Skor_Bobot",
            y="Fitur",
            orientation="h",
            text="Skor_Bobot",
            title="Top 20 Fitur TF-IDF"
        )
        fig_features.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig_features, use_container_width=True)

        st.dataframe(top_20_words, use_container_width=True)

        st.markdown("### Top 15 Bigram Paling Sering Muncul")
        if not bigram_df.empty:
            fig_bigram = px.bar(
                bigram_df.sort_values("Jumlah"),
                x="Jumlah",
                y="Fitur",
                orientation="h",
                text="Jumlah",
                title="Top 15 Bigram pada Data Aktif"
            )
            fig_bigram.update_traces(textposition="outside")
            st.plotly_chart(fig_bigram, use_container_width=True)
            st.dataframe(bigram_df, use_container_width=True)
        else:
            st.info("Bigram belum tersedia untuk filter data aktif.")


with tab4:
    st.subheader("WordCloud Ulasan Apex Legends")
    st.info("WordCloud berada di tab ini dan mengikuti filter sidebar.")

    st.caption(
        "WordCloud mengikuti filter sidebar. Jika kamu memilih label Negative, WordCloud hanya dibuat dari data Negative."
    )

    wc_fig = create_wordcloud_figure(display_df)

    if wc_fig is None:
        st.warning("WordCloud tidak dapat dibuat karena data aktif kosong atau final_text kosong.")
    else:
        st.pyplot(wc_fig)

    st.markdown("### WordCloud Per Label")

    col_wc_neg, col_wc_pos = st.columns(2)

    with col_wc_neg:
        st.write("Negative")
        neg_df = display_df[display_df["label"] == "Negative"]
        neg_fig = create_wordcloud_figure(neg_df)
        if neg_fig is not None:
            st.pyplot(neg_fig)
        else:
            st.info("Tidak ada data Negative pada filter aktif.")

    with col_wc_pos:
        st.write("Positive")
        pos_df = display_df[display_df["label"] == "Positive"]
        pos_fig = create_wordcloud_figure(pos_df)
        if pos_fig is not None:
            st.pyplot(pos_fig)
        else:
            st.info("Tidak ada data Positive pada filter aktif.")


with tab5:
    st.subheader("Prediksi Sentimen Ulasan Baru")

    user_review = st.text_area(
        "Masukkan ulasan berbahasa Inggris",
        value="not playable so many bug and cheater",
        height=140
    )

    if st.button("Analisis Sentimen", type="primary"):
        label_result, confidence, prob_df, processed = predict_sentiment(user_review, package)

        if label_result == "Positive":
            st.success(f"Hasil Prediksi: {label_result} ({confidence:.2f}% yakin)")
        else:
            st.error(f"Hasil Prediksi: {label_result} ({confidence:.2f}% yakin)")

        st.write("Probabilitas Model")
        fig_prob = px.bar(
            prob_df,
            x="Sentimen",
            y="Probabilitas",
            text="Probabilitas",
            title="Probabilitas Prediksi"
        )
        fig_prob.update_traces(texttemplate="%{text:.2%}", textposition="outside")
        fig_prob.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_prob, use_container_width=True)

        with st.expander("Lihat hasil preprocessing"):
            st.write("Case Folding")
            st.code(processed["case_folding"])
            st.write("Cleansing")
            st.code(processed["cleansing"])
            st.write("Normalization")
            st.code(processed["normalization"])
            st.write("Tokenizing")
            st.write(processed["tokenizing"])
            st.write("Filtering")
            st.write(processed["filtering"])
            st.write("Stemming")
            st.write(processed["stemming"])
            st.write("Final Text")
            st.code(processed["final_text"])


with tab6:
    st.subheader("Tabel Dataset")
    st.caption(f"Menampilkan {len(display_df):,} baris berdasarkan filter sidebar.")

    columns_to_show = [
        col for col in ["date", "review", "final_text", "label", "label_encoded"]
        if col in display_df.columns
    ]

    st.dataframe(
        display_df[columns_to_show],
        use_container_width=True,
        height=520
    )

    csv = display_df[columns_to_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download data hasil filter sebagai CSV",
        data=csv,
        file_name="hasil_filter_dashboard_apex.csv",
        mime="text/csv"
    )
