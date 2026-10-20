"""
Đánh giá chất lượng rượu vang dựa trên thành phần hóa lý
Bootstrap - Bagging - Random Forest - Extra Trees (lỗi OOB)

PHIÊN BẢN 2 - bám sát bộ tiêu chí chấm của học phần:
  BT1: từ điển dữ liệu, thống kê, EDA, định lượng & xử lý vấn đề dữ liệu, chia tập đúng cách
  BT2: baseline, Pipeline (không rò rỉ), cross-validation (TB ± SD), test chỉ dùng đánh giá cuối,
       ma trận nhầm lẫn, phân tích lỗi, so sánh train/validation (quá khớp)
  BT3: ablation (đóng góp từng thay đổi), đặc trưng mới, RandomizedSearchCV bằng CV,
       đường cong học tập, độ ổn định qua nhiều lần chạy, so sánh trước/sau cải tiến
  Kết thúc: chạy lại được, cố định seed, lưu kết quả và mô hình

Cách chạy:
    pip install -r requirements.txt
    python wine_quality_rf_v2.py            # gộp vang đỏ + trắng (mặc định)
    python wine_quality_rf_v2.py red        # chỉ vang đỏ   (tùy chọn)
    python wine_quality_rf_v2.py white      # chỉ vang trắng (tùy chọn)
    FAST=1 python wine_quality_rf_v2.py     # chạy nhanh (ít cây, ít fold) để kiểm tra lỗi

LƯU Ý PHƯƠNG PHÁP:
  * Chia train/test TRƯỚC mọi phép biến đổi phụ thuộc dữ liệu. Mọi phép biến đổi nằm trong Pipeline
    nên chỉ được fit trên tập huấn luyện (trong từng fold khi cross-validation).
  * Việc CHỌN mô hình / siêu tham số chỉ dựa trên cross-validation trên tập huấn luyện.
    Tập test chỉ dùng để báo cáo kết quả cuối (các đồ thị khảo sát siêu tham số có vẽ thêm
    lỗi test chỉ để THAM KHẢO theo yêu cầu đề tài, không dùng để chọn).
"""

import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # chỉ lưu ảnh, không cần mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, learning_curve,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CẤU HÌNH
# ----------------------------------------------------------------------------
FAST = os.environ.get("FAST") == "1"
SEED = 42
TEST_SIZE = 0.2
USE_3_LEVELS = True                  # True: gộp nhãn 3 mức (thấp/TB/cao); False: giữ điểm gốc
N_TREES = 30 if FAST else 300
CV_FOLDS = 3 if FAST else 5
N_ITER_SEARCH = 3 if FAST else 15
PERM_REPEATS = 3 if FAST else 20
STAB_SEEDS = [1, 2, 3] if FAST else [1, 2, 3, 4, 5]

SCOPE = sys.argv[1] if len(sys.argv) > 1 else "all"   # all | red | white
if SCOPE not in ("all", "red", "white"):
    raise SystemExit("Tham số phải là: all, red hoặc white")
OUT_DIR = "ket_qua" if SCOPE == "all" else f"ket_qua_{SCOPE}"
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
FILES = {"red": "winequality-red.csv", "white": "winequality-white.csv"}

os.makedirs(OUT_DIR, exist_ok=True)
CV = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)


def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  -> đã lưu {path}")


def save_csv(df, name, index=True):
    df.to_csv(os.path.join(OUT_DIR, name), index=index, encoding="utf-8-sig")


# ----------------------------------------------------------------------------
# 1. ĐỌC DỮ LIỆU + TỪ ĐIỂN DỮ LIỆU
# ----------------------------------------------------------------------------
DATA_DICT = [
    ("fixed acidity", "số thực", "g/dm³ (tartaric)", "Độ axit cố định (axit không bay hơi)"),
    ("volatile acidity", "số thực", "g/dm³ (acetic)", "Độ axit bay hơi; cao gây mùi giấm"),
    ("citric acid", "số thực", "g/dm³", "Axit citric, tạo vị tươi"),
    ("residual sugar", "số thực", "g/dm³", "Đường còn lại sau lên men"),
    ("chlorides", "số thực", "g/dm³ (NaCl)", "Hàm lượng muối"),
    ("free sulfur dioxide", "số thực", "mg/dm³", "SO2 tự do, chống oxy hóa/vi khuẩn"),
    ("total sulfur dioxide", "số thực", "mg/dm³", "Tổng SO2 (tự do + liên kết)"),
    ("density", "số thực", "g/cm³", "Khối lượng riêng, phụ thuộc cồn và đường"),
    ("pH", "số thực", "-", "Độ chua/kiềm"),
    ("sulphates", "số thực", "g/dm³ (K2SO4)", "Muối sunfat, phụ gia bảo quản"),
    ("alcohol", "số thực", "% thể tích", "Nồng độ cồn"),
    ("type", "nhị phân (mã hóa)", "0 = đỏ, 1 = trắng", "Loại rượu (chỉ có khi gộp hai tập)"),
    ("quality", "số nguyên 0-10", "điểm", "Điểm cảm quan do chuyên gia chấm (nhãn gốc)"),
    ("label", "nhãn lớp", "0/1/2", "Nhãn huấn luyện: 0 = thấp (<=5), 1 = TB (6), 2 = cao (>=7)"),
]


def load_data():
    frames = []
    for wine_type, fname in FILES.items():
        if SCOPE != "all" and wine_type != SCOPE:
            continue
        src = fname if os.path.exists(fname) else BASE_URL + fname
        print(f"Đọc {src}")
        d = pd.read_csv(src, sep=";")
        if SCOPE == "all":
            d["type"] = 0 if wine_type == "red" else 1  # 0 = đỏ, 1 = trắng
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------
# 2. KHẢO SÁT + ĐỊNH LƯỢNG VẤN ĐỀ DỮ LIỆU + XỬ LÝ
# ----------------------------------------------------------------------------
def to_level(q):
    if q <= 5:
        return 0  # thấp
    if q == 6:
        return 1  # trung bình
    return 2      # cao


def feature_cols(df):
    return [c for c in df.columns if c not in ("quality", "label", "type")]


def outlier_counts(df, feats):
    out = {}
    for c in feats:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        out[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    return pd.Series(out, name="Số ngoại lai (quy tắc IQR 1.5)")


def explore_and_clean(raw):
    print("\n=== KHẢO SÁT VÀ XỬ LÝ DỮ LIỆU ===")
    feats = feature_cols(raw)
    print("Kích thước ban đầu:", raw.shape)
    print("\nKiểu dữ liệu:\n", raw.dtypes.to_string())

    # Từ điển dữ liệu + thống kê mô tả
    dd = pd.DataFrame(DATA_DICT, columns=["Thuộc tính", "Kiểu", "Đơn vị", "Ý nghĩa"])
    dd = dd[dd["Thuộc tính"].isin(list(raw.columns) + ["label"])]
    save_csv(dd, "tu_dien_du_lieu.csv", index=False)
    save_csv(raw.describe().T, "thong_ke_mo_ta_truoc_xu_ly.csv")
    print("\nThống kê mô tả:\n", raw.describe().T.round(3).to_string())

    # Định lượng vấn đề dữ liệu
    n_missing = int(raw.isna().sum().sum())
    n_dup = int(raw.duplicated().sum())
    out_cnt = outlier_counts(raw, feats)
    cls = raw["quality"].value_counts().sort_index()
    imb = cls.max() / cls.min()
    issues = pd.DataFrame([
        ["Giá trị thiếu", f"{n_missing} ô", "Không cần xử lý" if n_missing == 0 else "Loại dòng thiếu",
         "Không có/ít nên loại bỏ không mất thông tin đáng kể"],
        ["Bản ghi trùng lặp", f"{n_dup} dòng ({n_dup / len(raw):.1%})", "Loại bỏ (drop_duplicates) TRƯỚC khi chia tập",
         "Tránh cùng một mẫu nằm ở cả train và test -> rò rỉ, điểm test cao giả"],
        ["Ngoại lai", f"{int(out_cnt.sum())} giá trị; nhiều nhất: " +
         ", ".join(f"{k} ({v})" for k, v in out_cnt.sort_values(ascending=False).head(3).items()),
         "Giữ nguyên", "Mô hình cây chia theo ngưỡng nên không nhạy với giá trị cực đoan; đây là giá trị đo thật"],
        ["Mất cân bằng lớp", f"điểm {cls.idxmax()} có {cls.max()} mẫu, điểm {cls.idxmin()} chỉ {cls.min()} mẫu "
                             f"(tỉ lệ {imb:.0f}:1)",
         "Gộp nhãn 3 mức + dùng F1 macro + thử class_weight" if USE_3_LEVELS else "Dùng F1 macro + class_weight",
         "Các điểm 3, 4, 8, 9 quá ít mẫu để học; F1 macro không bị lớp đông lấn át"],
        ["Rò rỉ dữ liệu", "Cột quality là nguồn của nhãn", "Loại quality và label khỏi X; chia tập trước khi biến đổi",
         "Nếu giữ quality trong X thì mô hình 'biết' đáp án"],
    ], columns=["Vấn đề", "Định lượng", "Xử lý", "Lý do"])
    save_csv(issues, "van_de_du_lieu.csv", index=False)
    save_csv(out_cnt.to_frame(), "ngoai_lai_theo_dac_trung.csv")
    print("\nBảng vấn đề dữ liệu:\n", issues.to_string(index=False))

    # Xử lý: loại thiếu + trùng
    df = raw.dropna().drop_duplicates().reset_index(drop=True)

    # So sánh trước - sau
    cmp = pd.DataFrame({"Trước xử lý": raw[feats].mean(), "Sau xử lý": df[feats].mean()})
    cmp.loc["(số mẫu)"] = [len(raw), len(df)]
    save_csv(cmp.round(4), "so_sanh_truoc_sau_xu_ly.csv")
    print(f"\nSố mẫu: {len(raw)} -> {len(df)} (loại {len(raw) - len(df)})")

    df["label"] = df["quality"].apply(to_level) if USE_3_LEVELS else df["quality"]
    class_names = (["thấp (<=5)", "trung bình (6)", "cao (>=7)"] if USE_3_LEVELS
                   else [str(c) for c in sorted(df["label"].unique())])
    print("\nPhân phối điểm gốc sau xử lý:\n", df["quality"].value_counts().sort_index().to_string())
    print("\nPhân phối nhãn huấn luyện:\n", df["label"].value_counts().sort_index().to_string())

    # ---- EDA: hình + số liệu hỗ trợ viết nhận xét ----
    ax = pd.concat([raw["quality"].value_counts().sort_index().rename("Trước xử lý"),
                    df["quality"].value_counts().sort_index().rename("Sau xử lý")], axis=1).fillna(0).plot(
        kind="bar", figsize=(8, 5), rot=0)
    ax.set_xlabel("Điểm chất lượng (quality)")
    ax.set_ylabel("Số mẫu")
    ax.set_title("Phân phối điểm chất lượng trước và sau khi loại bản ghi trùng")
    save_fig("01_phan_phoi_nhan.png")

    axes = df[feats].hist(bins=30, figsize=(14, 9), layout=(3, 4))
    for a in axes.ravel():
        a.set_xlabel("Giá trị")
        a.set_ylabel("Số mẫu")
    plt.suptitle("Phân phối các đặc trưng hóa lý")
    save_fig("02_phan_phoi_dac_trung.png")

    corr = df[feats + ["quality"]].corr()
    plt.figure(figsize=(9, 7))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Hệ số tương quan Pearson")
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Ma trận tương quan giữa các đặc trưng và quality")
    save_fig("03_tuong_quan.png")

    cq = corr["quality"].drop("quality").sort_values(key=abs, ascending=False)
    print("\nTương quan với quality (giảm dần theo |r|):\n", cq.round(3).to_string())
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:] if abs(corr.loc[a, b]) >= 0.6]
    print("\nCặp đặc trưng tương quan mạnh (|r| >= 0.6):",
          [(a, b, round(r, 2)) for a, b, r in pairs] or "không có")
    top4 = list(cq.index[:4])
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    for a, f in zip(axs, top4):
        df.boxplot(column=f, by="label", ax=a)
        a.set_title(f)
        a.set_xlabel("Nhãn (0 thấp, 1 TB, 2 cao)")
        a.set_ylabel("Giá trị")
    plt.suptitle("Phân phối 4 đặc trưng tương quan mạnh nhất với chất lượng, theo nhãn")
    save_fig("04_boxplot_theo_nhan.png")
    print("\nSkewness (độ lệch) lớn nhất:\n", df[feats].skew().sort_values(key=abs, ascending=False).head(4).round(2).to_string())

    X = df.drop(columns=["quality", "label"])
    y = df["label"]
    return X, y, class_names


# ----------------------------------------------------------------------------
# 3. PIPELINE + THÍ NGHIỆM
# ----------------------------------------------------------------------------
def add_features(X):
    """Đặc trưng mới tính trên từng dòng nên không gây rò rỉ dữ liệu."""
    X = X.copy()
    X["so2_ratio"] = X["free sulfur dioxide"] / (X["total sulfur dioxide"] + 1e-9)
    X["total_acidity"] = X["fixed acidity"] + X["volatile acidity"] + X["citric acid"]
    X["sugar_per_alcohol"] = X["residual sugar"] / (X["alcohol"] + 1e-9)
    return X


def build_pipeline(clf, scale=False, fe=False):
    steps = []
    if fe:
        steps.append(("fe", FunctionTransformer(add_features)))
    if scale:
        steps.append(("scale", StandardScaler()))   # chỉ fit trên train (trong pipeline/fold)
    steps.append(("clf", clf))
    return Pipeline(steps)


def rf(**kw):
    return RandomForestClassifier(n_estimators=N_TREES, oob_score=True, n_jobs=-1, random_state=SEED, **kw)


def base_experiments():
    return {
        "Baseline: lớp đa số": build_pipeline(DummyClassifier(strategy="most_frequent")),
        "Baseline: Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=2000, random_state=SEED), scale=True),
        "Decision Tree": build_pipeline(DecisionTreeClassifier(random_state=SEED)),
        "Bagging": build_pipeline(BaggingClassifier(
            DecisionTreeClassifier(random_state=SEED), n_estimators=N_TREES, oob_score=True,
            n_jobs=-1, random_state=SEED)),
        "Random Forest": build_pipeline(rf()),
        "Extra Trees": build_pipeline(ExtraTreesClassifier(
            n_estimators=N_TREES, bootstrap=True, oob_score=True, n_jobs=-1, random_state=SEED)),
        "RF + class_weight": build_pipeline(rf(class_weight="balanced_subsample")),
        "RF + class_weight + đặc trưng mới": build_pipeline(rf(class_weight="balanced_subsample"), fe=True),
    }


def run_experiments(exps, Xtr, Xte, ytr, yte):
    """CV trên train (TB ± SD, có điểm train để nhận diện quá khớp), rồi đánh giá test MỘT lần."""
    rows, fitted = [], {}
    for name, pipe in exps.items():
        r = cross_validate(pipe, Xtr, ytr, cv=CV, return_train_score=True,
                           scoring={"acc": "accuracy", "f1": "f1_macro"})
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        clf = pipe.named_steps["clf"]
        rows.append({
            "Mô hình": name,
            "CV Acc TB": r["test_acc"].mean(), "CV Acc SD": r["test_acc"].std(),
            "CV F1 TB": r["test_f1"].mean(), "CV F1 SD": r["test_f1"].std(),
            "Train F1 (CV)": r["train_f1"].mean(),
            "Chênh lệch train-val": r["train_f1"].mean() - r["test_f1"].mean(),
            "Test Acc": accuracy_score(yte, pred),
            "Test F1 macro": f1_score(yte, pred, average="macro"),
            "Lỗi OOB": (1 - clf.oob_score_) if hasattr(clf, "oob_score_") else np.nan,
        })
        fitted[name] = pipe
        print(f"  xong: {name}")
    return pd.DataFrame(rows).set_index("Mô hình"), fitted


def plot_model_comparison(res):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(res))
    ax.bar(x, res["CV F1 TB"], yerr=res["CV F1 SD"], capsize=4, color="#4c72b0", label="CV F1 macro (TB ± SD, trên train)")
    ax.plot(x, res["Test F1 macro"], "D", color="#dd8452", label="F1 macro trên tập test")
    ax.set_xticks(x)
    ax.set_xticklabels(res.index, rotation=30, ha="right")
    ax.set_ylabel("F1 macro")
    ax.set_xlabel("Mô hình")
    ax.set_ylim(0, 1)
    ax.set_title("So sánh các mô hình (baseline -> ensemble -> cải tiến)")
    ax.legend()
    save_fig("05_so_sanh_mo_hinh.png")


# ----------------------------------------------------------------------------
# 4. KHẢO SÁT SIÊU THAM SỐ (đường cong kiểm định bằng OOB)
# ----------------------------------------------------------------------------
def sweep(label, values, make, Xtr, Xte, ytr, yte, fname):
    rows = []
    for v in values:
        m = make(v)
        m.fit(Xtr, ytr)
        rows.append({label: str(v),
                     "Lỗi train": 1 - m.score(Xtr, ytr),
                     "Lỗi OOB": 1 - m.oob_score_,
                     "Lỗi test (chỉ tham khảo)": 1 - m.score(Xte, yte)})
    d = pd.DataFrame(rows)
    print(f"\nKhảo sát {label}:\n", d.round(4).to_string(index=False))
    save_csv(d.round(4), fname + ".csv", index=False)
    x = np.arange(len(d))
    plt.figure(figsize=(7, 5))
    plt.plot(x, d["Lỗi train"], "^:", label="Lỗi train")
    plt.plot(x, d["Lỗi OOB"], "o-", label="Lỗi OOB")
    plt.plot(x, d["Lỗi test (chỉ tham khảo)"], "s--", label="Lỗi test (chỉ tham khảo)")
    plt.xticks(x, d[label])
    plt.xlabel(label)
    plt.ylabel("Tỉ lệ lỗi")
    plt.title(f"Ảnh hưởng của {label}")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig(fname + ".png")


def hyperparam_study(Xtr, Xte, ytr, yte):
    print("\n=== KHẢO SÁT ẢNH HƯỞNG SIÊU THAM SỐ ===")
    # Mô hình cây không cần biến đổi đặc trưng nên dùng trực tiếp bộ phân loại ở bước này.
    n_list = [10, 25, 50, 100, 200] if FAST else [10, 25, 50, 100, 200, 300, 500]
    sweep("n_estimators (Random Forest)", n_list,
          lambda n: RandomForestClassifier(n_estimators=n, oob_score=True, n_jobs=-1, random_state=SEED),
          Xtr, Xte, ytr, yte, "06_n_estimators_rf")
    sweep("n_estimators (Bagging)", n_list,
          lambda n: BaggingClassifier(DecisionTreeClassifier(random_state=SEED), n_estimators=n,
                                      oob_score=True, n_jobs=-1, random_state=SEED),
          Xtr, Xte, ytr, yte, "07_n_estimators_bagging")
    n_feat = Xtr.shape[1]
    mf = [1, 2, 3, "sqrt", n_feat // 2, n_feat]
    sweep("max_features", mf,
          lambda m: RandomForestClassifier(n_estimators=N_TREES, max_features=m, oob_score=True,
                                           n_jobs=-1, random_state=SEED),
          Xtr, Xte, ytr, yte, "08_max_features")
    md = [2, 3, 5, 8, 12, 20, None]
    sweep("max_depth", md,
          lambda m: RandomForestClassifier(n_estimators=N_TREES, max_depth=m, oob_score=True,
                                           n_jobs=-1, random_state=SEED),
          Xtr, Xte, ytr, yte, "09_max_depth")


# ----------------------------------------------------------------------------
# 5. TỐI ƯU SIÊU THAM SỐ (CV trên train)
# ----------------------------------------------------------------------------
def tune(Xtr, ytr):
    print("\n=== TỐI ƯU SIÊU THAM SỐ (RandomizedSearchCV trên cross-validation, tập train) ===")
    pipe = build_pipeline(rf(class_weight="balanced_subsample"), fe=True)
    space = {
        "clf__n_estimators": [100, 200] if FAST else [200, 300, 500],
        "clf__max_features": ["sqrt", 0.3, 0.5],
        "clf__max_depth": [None, 12, 16, 20],
        "clf__min_samples_leaf": [1, 2, 3, 5],
    }
    s = RandomizedSearchCV(pipe, space, n_iter=N_ITER_SEARCH, scoring="f1_macro", cv=CV,
                           random_state=SEED, n_jobs=1, refit=True)
    s.fit(Xtr, ytr)
    res = pd.DataFrame(s.cv_results_).sort_values("rank_test_score")
    cols = ["rank_test_score", "mean_test_score", "std_test_score"] + [c for c in res.columns if c.startswith("param_")]
    save_csv(res[cols].round(4), "ket_qua_tim_kiem_sieu_tham_so.csv", index=False)
    print("Top 5 cấu hình:\n", res[cols].head(5).round(4).to_string(index=False))
    print("Tốt nhất:", s.best_params_, f"| CV F1 macro = {s.best_score_:.4f}")
    return s.best_estimator_


# ----------------------------------------------------------------------------
# 6. TỔNG QUÁT HÓA, ĐƯỜNG CONG HỌC TẬP, PHÂN TÍCH LỖI, ĐỘ QUAN TRỌNG
# ----------------------------------------------------------------------------
def stability(final, Xtr, Xte, ytr, yte):
    print("\n=== ĐỘ ỔN ĐỊNH QUA CÁC LẦN CHẠY (đổi random_state của mô hình, cùng một lần chia) ===")
    accs, f1s = [], []
    for sd in STAB_SEEDS:
        m = clone(final).set_params(clf__random_state=sd)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        accs.append(accuracy_score(yte, p))
        f1s.append(f1_score(yte, p, average="macro"))
    print(f"Accuracy test: {np.mean(accs):.4f} ± {np.std(accs):.4f} | F1 macro test: {np.mean(f1s):.4f} ± {np.std(f1s):.4f} "
          f"(n = {len(STAB_SEEDS)} lần)")


def plot_learning_curve(final, Xtr, ytr):
    sizes, tr, va = learning_curve(final, Xtr, ytr, cv=CV, scoring="f1_macro",
                                   train_sizes=np.linspace(0.2, 1.0, 4 if FAST else 6), random_state=SEED)
    plt.figure(figsize=(7, 5))
    for arr, lab in [(tr, "Train"), (va, "Validation (CV)")]:
        plt.plot(sizes, arr.mean(1), "o-", label=lab)
        plt.fill_between(sizes, arr.mean(1) - arr.std(1), arr.mean(1) + arr.std(1), alpha=0.2)
    plt.xlabel("Số mẫu huấn luyện")
    plt.ylabel("F1 macro")
    plt.title("Đường cong học tập của mô hình cuối")
    plt.legend()
    plt.grid(alpha=0.3)
    save_fig("10_duong_cong_hoc_tap.png")


def error_analysis(final, Xte, yte, class_names):
    print("\n=== PHÂN TÍCH LỖI TRÊN TẬP TEST (mô hình cuối) ===")
    pred = final.predict(Xte)
    print(classification_report(yte, pred, target_names=class_names, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=class_names, cmap="Blues", xticks_rotation=15)
    plt.title("Ma trận nhầm lẫn - mô hình cuối")
    save_fig("11_confusion_matrix.png")

    err = pred != yte.values
    adj = (np.abs(pred - yte.values) == 1)[err].mean() if err.any() else float("nan")
    print(f"Tỉ lệ lỗi: {err.mean():.4f}. Trong các mẫu sai, {adj:.1%} chỉ lệch 1 mức (nhầm sang lớp kề bên).")
    d = Xte.copy()
    d["sai"] = err
    d["thật"] = yte.values
    out = {}
    out["Theo lớp thật"] = d.groupby("thật")["sai"].mean().round(4)
    if "type" in d.columns:
        out["Theo loại rượu (0 đỏ, 1 trắng)"] = d.groupby("type")["sai"].mean().round(4)
    out["Theo nhóm cồn (tứ phân vị)"] = d.groupby(pd.qcut(d["alcohol"], 4, duplicates="drop"))["sai"].mean().round(4)
    for k, v in out.items():
        print(f"\nTỉ lệ sai {k}:\n", v.to_string())
    pd.concat({k: v.reset_index(drop=True) for k, v in out.items()}, axis=0).to_csv(
        os.path.join(OUT_DIR, "phan_tich_loi.csv"), encoding="utf-8-sig")


def feature_importance(pipe, Xte, yte):
    print("\n=== ĐỘ QUAN TRỌNG ĐẶC TRƯNG (Random Forest trên đặc trưng gốc) ===")
    names = Xte.columns
    mdi = pd.Series(pipe.named_steps["clf"].feature_importances_, index=names)
    perm = permutation_importance(pipe, Xte, yte, n_repeats=PERM_REPEATS, scoring="f1_macro",
                                  random_state=SEED, n_jobs=-1)
    pm = pd.Series(perm.importances_mean, index=names).clip(lower=0)
    tbl = pd.DataFrame({"Impurity (MDI)": mdi / mdi.sum(), "Permutation": pm / max(pm.sum(), 1e-12)})
    tbl["Hạng MDI"] = tbl["Impurity (MDI)"].rank(ascending=False).astype(int)
    tbl["Hạng Perm"] = tbl["Permutation"].rank(ascending=False).astype(int)
    tbl = tbl.sort_values("Impurity (MDI)", ascending=False)
    print(tbl.round(4).to_string())
    save_csv(tbl.round(4), "do_quan_trong_dac_trung.csv")
    idx, w = np.arange(len(tbl)), 0.4
    plt.figure(figsize=(10, 6))
    plt.barh(idx - w / 2, tbl["Impurity (MDI)"], w, label="Impurity (MDI)")
    plt.barh(idx + w / 2, tbl["Permutation"], w, label="Permutation")
    plt.yticks(idx, tbl.index)
    plt.gca().invert_yaxis()
    plt.xlabel("Độ quan trọng (đã chuẩn hóa, tổng = 1)")
    plt.ylabel("Đặc trưng")
    plt.title("So sánh độ quan trọng đặc trưng: impurity và permutation")
    plt.legend()
    save_fig("12_do_quan_trong.png")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    if FAST:
        print(">>> CHẾ ĐỘ FAST: chỉ để kiểm tra chạy được, KHÔNG dùng số liệu này cho báo cáo <<<")
    raw = load_data()
    X, y, class_names = explore_and_clean(raw)

    # CHIA TẬP TRƯỚC khi có bất kỳ phép biến đổi nào (phân tầng theo nhãn)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"\nTrain: {Xtr.shape}, Test: {Xte.shape}")
    print("Tỉ lệ nhãn train:", ytr.value_counts(normalize=True).sort_index().round(3).to_dict())
    print("Tỉ lệ nhãn test: ", yte.value_counts(normalize=True).sort_index().round(3).to_dict())

    print("\n=== THÍ NGHIỆM: BASELINE, ENSEMBLE, ABLATION (CV trên train, test đánh giá một lần) ===")
    res, fitted = run_experiments(base_experiments(), Xtr, Xte, ytr, yte)

    hyperparam_study(Xtr, Xte, ytr, yte)

    best_pipe = tune(Xtr, ytr)
    res2, fitted2 = run_experiments({"RF + class_weight + đặc trưng mới + tinh chỉnh": best_pipe}, Xtr, Xte, ytr, yte)
    res = pd.concat([res, res2])
    fitted.update(fitted2)

    print("\n=== BẢNG TỔNG HỢP (cũng là bảng ablation: đi từ trên xuống dưới) ===")
    print(res.round(4).to_string())
    save_csv(res.round(4), "bang_tong_hop_va_ablation.csv")
    plot_model_comparison(res)

    # Chọn mô hình cuối theo CV (KHÔNG theo test)
    final_name = res["CV F1 TB"].idxmax()
    final = fitted[final_name]
    print(f"\nMô hình cuối (chọn theo CV F1 macro cao nhất): {final_name}")
    print(f"  CV F1 = {res.loc[final_name, 'CV F1 TB']:.4f} ± {res.loc[final_name, 'CV F1 SD']:.4f}; "
          f"Test F1 = {res.loc[final_name, 'Test F1 macro']:.4f}")

    stability(final, Xtr, Xte, ytr, yte)
    plot_learning_curve(final, Xtr, ytr)
    error_analysis(final, Xte, yte, class_names)
    feature_importance(fitted["Random Forest"], Xte, yte)

    joblib.dump(final, os.path.join(OUT_DIR, "mo_hinh_cuoi.joblib"))
    print("\nDemo dự đoán 3 mẫu test:", final.predict(Xte.head(3)).tolist(), "| nhãn thật:", yte.head(3).tolist())
    print(f"\nHoàn tất. Xem kết quả trong thư mục '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()
