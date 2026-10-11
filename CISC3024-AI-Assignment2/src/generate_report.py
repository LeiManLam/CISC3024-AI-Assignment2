"""Generate PDF and DOCX reports for CISC3024 AI Assignment #2."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
REPORT = ROOT / "report"
METRICS = json.loads((RES / "metrics.json").read_text())
HISTORY = json.loads((RES / "history.json").read_text())
SOURCE_URL = "https://github.com/LeiManLam/CISC3024-AI-Assignment2"


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontSize=22,
            spaceAfter=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1a1a2e"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSub",
            parent=styles["Normal"],
            fontSize=14,
            spaceAfter=6,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SecHead",
            parent=styles["Heading1"],
            fontSize=14,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor("#16213e"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJust",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCN",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            fontName="Helvetica",  # Traditional Chinese may need fallback; we also put CN in docx
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["Normal"],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=12,
            spaceBefore=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Code"],
            fontSize=8,
            leading=11,
            backColor=colors.HexColor("#f5f5f5"),
            leftIndent=6,
            rightIndent=6,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    return styles


def add_figure(story, styles, path: Path, caption: str, width=5.5 * inch):
    if path.exists():
        story.append(Image(str(path), width=width, height=width * 0.72))
        story.append(Paragraph(caption, styles["Caption"]))


def build_pdf():
    REPORT.mkdir(parents=True, exist_ok=True)
    out = REPORT / "AI_Assignment2_CISC3024.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="AI Assignment #2 — CISC3024 Pattern Recognition",
        author="coco_lei73",
    )
    styles = build_styles()
    story = []

    # ---- Cover ----
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("AI Assignment #2", styles["CoverTitle"]))
    story.append(Paragraph("CISC3024 Pattern Recognition", styles["CoverSub"]))
    story.append(Paragraph("University of Macau", styles["CoverSub"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph(
            "<b>Monte Carlo Dropout as a Bayesian Approximation<br/>for Fashion-MNIST Image Classification</b>",
            styles["CoverSub"],
        )
    )
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("October 2026", styles["CoverSub"]))
    story.append(
        Paragraph(
            "Algorithm: Gal &amp; Ghahramani, ICML 2016 (arXiv:1506.02142)",
            styles["CoverSub"],
        )
    )
    story.append(
        Paragraph(
            f'Source code: <link href="{SOURCE_URL}">{SOURCE_URL}</link>',
            styles["CoverSub"],
        )
    )
    story.append(PageBreak())

    # ---- Section 1 ----
    story.append(Paragraph("1. How the AI Finds the Algorithm", styles["SecHead"]))
    story.append(
        Paragraph(
            "The assignment requires a <b>recent</b> classification/recognition method based on "
            "<b>probability or Bayes theory</b> for computer vision / pattern recognition, with the AI "
            "performing algorithm search, programming, and report writing.",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "<b>Search process.</b> Queries used included: "
            "“Bayesian neural network computer vision classification 2020 2021 2022 2023 MC Dropout variational inference”, "
            "“Monte Carlo Dropout Bayesian approximation Gal Ghahramani image classification”, and "
            "“Mobiny Monte Carlo DropConnect Bayesian deep networks”. Sources consulted were arXiv, "
            "NeurIPS proceedings, and Scientific Reports.",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "<b>Candidates considered:</b>",
            styles["BodyJust"],
        )
    )
    story.append(
        ListFlowable(
            [
                ListItem(
                    Paragraph(
                        "<b>MC Dropout</b> (Gal &amp; Ghahramani, ICML 2016, arXiv:1506.02142) — "
                        "dropout as approximate variational inference in deep Gaussian processes; "
                        "still the standard Bayesian DL baseline in 2020+ CV / medical imaging benchmarks.",
                        styles["BodyJust"],
                    )
                ),
                ListItem(
                    Paragraph(
                        "<b>MC-DropConnect</b> (Mobiny et al., Scientific Reports, 2021; arXiv:1906.04569) — "
                        "Bernoulli variational approximation on weights; validated on classification and segmentation.",
                        styles["BodyJust"],
                    )
                ),
                ListItem(
                    Paragraph(
                        "<b>Variational Structured Dropout (VSD)</b> (NeurIPS 2021) — structured variational "
                        "noise; stronger theory but heavier to implement on CPU.",
                        styles["BodyJust"],
                    )
                ),
                ListItem(
                    Paragraph(
                        "<b>Function-space VI / MC-DropBlock</b> (NeurIPS 2022 / 2021) — excellent uncertainty "
                        "work but more complex architectures and compute.",
                        styles["BodyJust"],
                    )
                ),
            ],
            bulletType="bullet",
        )
    )
    story.append(
        Paragraph(
            "<b>Selection.</b> We chose <b>Monte Carlo Dropout</b> because (1) it has a clear Bayes / "
            "variational-inference foundation (approximate posterior over weights via Bernoulli dropout masks), "
            "(2) it is directly applicable to CNN image classification, (3) it remains widely used and cited "
            "in post-2020 Bayesian deep learning for vision (e.g., NeurIPS 2021 diabetic-retinopathy benchmarks), "
            "and (4) it is implementable on CPU with modest resources (Fashion-MNIST). "
            "MC-DropConnect (2021) was a strong recent alternative; MC Dropout was preferred for simpler "
            "reproducibility while preserving the same Bayesian interpretation.",
            styles["BodyJust"],
        )
    )

    # ---- Section 2 ----
    story.append(Paragraph("2. Algorithm Description", styles["SecHead"]))
    story.append(
        Paragraph(
            "<b>Reference:</b> Yarin Gal and Zoubin Ghahramani. "
            "“Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.” "
            "<i>Proceedings of the 33rd International Conference on Machine Learning (ICML)</i>, "
            "pp. 1050–1059, 2016. arXiv:1506.02142.",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "<b>Core idea.</b> Training a neural network with dropout (and L2 weight decay) can be "
            "interpreted as performing approximate <b>variational inference</b> for a deep Gaussian process. "
            "The variational distribution q(ω) places independent Bernoulli masks on network units "
            "(or equivalently induces a distribution over weights). Minimising the usual dropout training "
            "objective approximates minimising KL[q(ω) || p(ω | data)].",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "<b>Predictive inference (MC Dropout).</b> At test time, instead of turning dropout off, "
            "we keep it active and draw T stochastic forward passes with independent dropout masks "
            "ω̂<sub>t</sub> ~ q(ω). The approximate predictive distribution is the Monte Carlo average:",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "q(y* | x*) ≈ (1/T) Σ<sub>t=1</sub><sup>T</sup> Softmax(f<sub>ω̂<sub>t</sub></sub>(x*))",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "From the averaged probabilities we obtain the predicted class, negative log-likelihood (NLL), "
            "and uncertainty summaries such as <b>predictive entropy</b>. Because the method yields a "
            "distribution over predictions rather than a single point estimate, it is a probability / "
            "Bayes-theory approach suitable for computer vision classification and for detecting "
            "when the model is unsure.",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "<b>Why it fits this course.</b> Pattern recognition historically uses Bayes classifiers and "
            "probabilistic models. MC Dropout brings that Bayesian predictive view to modern CNNs: "
            "the network still learns visual features, but predictions come from an approximate posterior "
            "rather than a single MAP weight vector.",
            styles["BodyJust"],
        )
    )

    # ---- Section 3 ----
    story.append(Paragraph("3. How the AI Implements the Algorithm", styles["SecHead"]))
    story.append(
        Paragraph(
            "All code was written by the AI agent (Python + PyTorch). The student did not write source code. "
            "Implementation highlights:",
            styles["BodyJust"],
        )
    )
    story.append(
        ListFlowable(
            [
                ListItem(
                    Paragraph(
                        "<b>model.py</b> — <font face='Courier'>BayesianCNN</font>: two conv blocks "
                        "(32→64 channels), max-pooling, fully connected 128→10, with "
                        "<font face='Courier'>Dropout2d</font> / <font face='Courier'>Dropout</font> "
                        "(p=0.25). Method <font face='Courier'>mc_predict(x, T)</font> forces dropout "
                        "layers into train mode and averages T softmax outputs.",
                        styles["BodyJust"],
                    )
                ),
                ListItem(
                    Paragraph(
                        "<b>train_eval.py</b> — downloads Fashion-MNIST, trains with Adam "
                        "(lr=1e-3, weight decay=1e-4, 8 epochs, batch size 128), evaluates both "
                        "deterministic (dropout off) and MC Dropout (T=30) modes.",
                        styles["BodyJust"],
                    )
                ),
                ListItem(
                    Paragraph(
                        "<b>metrics.py</b> — accuracy, mean NLL, Expected Calibration Error (ECE, 15 bins), "
                        "confusion matrix.",
                        styles["BodyJust"],
                    )
                ),
                ListItem(
                    Paragraph(
                        "Figures: training curves, confusion matrix, reliability diagram, and "
                        "predictive-entropy histograms for correct vs incorrect predictions.",
                        styles["BodyJust"],
                    )
                ),
            ],
            bulletType="bullet",
        )
    )
    story.append(
        Paragraph(
            f"Repository layout and instructions are in the project README. "
            f"Public source link: <link href='{SOURCE_URL}'>{SOURCE_URL}</link>.",
            styles["BodyJust"],
        )
    )

    # ---- Section 4 ----
    story.append(Paragraph("4. Experiment Settings and Results", styles["SecHead"]))
    story.append(
        Paragraph(
            "<b>Dataset.</b> Fashion-MNIST (Zalando Research): 60,000 training and 10,000 test "
            "grayscale 28×28 images of clothing items in 10 classes "
            "(T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).",
            styles["BodyJust"],
        )
    )

    settings = [
        ["Setting", "Value"],
        ["Algorithm", "MC Dropout (Gal & Ghahramani, 2016)"],
        ["Model", "CNN (~422K parameters)"],
        ["Dropout probability p", str(METRICS["dropout_p"])],
        ["MC samples T", str(METRICS["mc_samples_t"])],
        ["Epochs", str(METRICS["epochs"])],
        ["Batch size", str(METRICS["batch_size"])],
        ["Optimizer", f"Adam, lr={METRICS['lr']}, wd={METRICS['weight_decay']}"],
        ["Device", METRICS["device"]],
        ["Seed", str(METRICS["seed"])],
    ]
    t = Table(settings, colWidths=[2.4 * inch, 4.0 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))

    metrics_rows = [
        ["Metric", "Value"],
        ["Deterministic test accuracy (dropout off)", f"{METRICS['deterministic_test_accuracy']*100:.2f}%"],
        ["MC Dropout test accuracy (T=30)", f"{METRICS['mc_dropout_test_accuracy']*100:.2f}%"],
        ["MC Dropout mean NLL", f"{METRICS['mc_dropout_nll']:.4f}"],
        ["Expected Calibration Error (ECE)", f"{METRICS['mc_dropout_ece']:.4f}"],
        ["Mean predictive entropy (all)", f"{METRICS['mean_predictive_entropy']:.4f}"],
        ["Mean entropy (correct predictions)", f"{METRICS['mean_entropy_correct']:.4f}"],
        ["Mean entropy (incorrect predictions)", f"{METRICS['mean_entropy_incorrect']:.4f}"],
    ]
    t2 = Table(metrics_rows, colWidths=[3.4 * inch, 3.0 * inch])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eef5ff")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "<b>Discussion.</b> MC Dropout accuracy (~91.07%) matches the deterministic CNN (~91.11%), "
            "so Bayesian averaging does not hurt recognition performance. ECE ≈ 0.020 indicates reasonably "
            "calibrated probabilities. Incorrect predictions have much higher predictive entropy "
            f"({METRICS['mean_entropy_incorrect']:.3f}) than correct ones "
            f"({METRICS['mean_entropy_correct']:.3f}), showing that the Bayesian approximation flags "
            "uncertain cases — a key advantage over point-estimate networks for pattern recognition.",
            styles["BodyJust"],
        )
    )

    add_figure(story, styles, FIG / "training_curves.png", "Figure 1. Training / validation loss and accuracy.")
    add_figure(story, styles, FIG / "confusion_matrix.png", "Figure 2. Confusion matrix under MC Dropout (T=30).")
    add_figure(
        story, styles, FIG / "reliability_diagram.png", "Figure 3. Reliability diagram (calibration)."
    )
    add_figure(
        story,
        styles,
        FIG / "entropy_correct_vs_incorrect.png",
        "Figure 4. Predictive entropy for correct vs incorrect predictions.",
    )

    # ---- Section 5 (Traditional Chinese) ----
    story.append(Paragraph("5. What the Student Has Learned from This AI Assignment", styles["SecHead"]))
    story.append(
        Paragraph(
            "(This section is written in Traditional Chinese from the student’s first-person perspective. "
            "If glyphs render incompletely in PDF Helvetica, the DOCX version carries the full Chinese text.)",
            styles["Caption"],
        )
    )
    # Note: ReportLab Helvetica can't render CJK. We'll embed Chinese as Unicode with a note,
    # and put proper Chinese in DOCX. For PDF, try to use a system CJK font if available.
    cn_paragraphs = [
        "透過這次 AI 作業，我學到如何把課堂上的貝氏／機率概念連接到現代深度學習。以前接觸貝氏分類器時，"
        "多半是較簡單的生成模型；這次用 Monte Carlo Dropout，才理解「對權重做近似後驗推論」也可以套在 CNN 上，"
        "讓影像分類不只給出類別，還能估計模型有多不確定。",
        "我也體會到「查論文」與「能實作」之間的差別。AI 幫我搜尋 2020 年後仍常用的貝氏深度學習方法，"
        "比較 MC Dropout、MC-DropConnect、結構化變分 Dropout 等候選後，選了理論清楚、資源需求適中的方案，"
        "並在 Fashion-MNIST 上跑出真實的準確率、NLL、ECE 與混淆矩陣，而不是虛構數字。",
        "最重要的是，作業規定學生不可自行寫程式，整段實作與報告由 AI 完成。我學到如何清楚下達需求、"
        "檢查實驗設定是否合理、解讀不確定度指標（例如錯誤樣本的預測熵明顯較高），以及把原始碼整理成可公開的專案。"
        "這讓我更明白在模式識別中，機率思維不只是公式，而是可以落地成可重現實驗的工程流程。",
    ]
    # Will replace with CJK font if found
    for p in cn_paragraphs:
        story.append(Paragraph(p, styles["BodyJust"]))

    # ---- Section 6 ----
    story.append(Paragraph("6. Webpage Link to the Source Codes", styles["SecHead"]))
    story.append(
        Paragraph(
            f"All source code, experiment scripts, figures, and this report are published at:",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            f'<b><link href="{SOURCE_URL}" color="blue">{SOURCE_URL}</link></b>',
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "The repository includes <font face='Courier'>src/model.py</font>, "
            "<font face='Courier'>src/train_eval.py</font>, <font face='Courier'>src/metrics.py</font>, "
            "<font face='Courier'>requirements.txt</font>, <font face='Courier'>README.md</font>, "
            "trained results under <font face='Courier'>results/</font>, and figures under "
            "<font face='Courier'>figures/</font>.",
            styles["BodyJust"],
        )
    )

    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "<b>References</b>",
            styles["SecHead"],
        )
    )
    story.append(
        Paragraph(
            "[1] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian Approximation: Representing Model "
            "Uncertainty in Deep Learning,” in <i>ICML</i>, 2016. https://arxiv.org/abs/1506.02142",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "[2] A. Mobiny et al., “DropConnect is effective in modeling uncertainty of Bayesian deep "
            "networks,” <i>Scientific Reports</i>, 2021. https://arxiv.org/abs/1906.04569",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "[3] H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-MNIST: a Novel Image Dataset for "
            "Benchmarking Machine Learning Algorithms,” arXiv:1708.07747, 2017.",
            styles["BodyJust"],
        )
    )

    # Register CJK font if available for better Chinese rendering
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import glob

    cjk_candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    # Also search
    cjk_candidates += glob.glob("/usr/share/fonts/**/*Noto*CJK*", recursive=True)
    cjk_candidates += glob.glob("/usr/share/fonts/**/*wqy*", recursive=True)
    cjk_candidates += glob.glob("/usr/share/fonts/**/*SourceHan*", recursive=True)

    cjk_font = None
    for fp in cjk_candidates:
        p = Path(fp)
        if p.exists():
            try:
                # TTC needs subfontIndex
                if p.suffix.lower() == ".ttc":
                    pdfmetrics.registerFont(TTFont("CJK", str(p), subfontIndex=0))
                else:
                    pdfmetrics.registerFont(TTFont("CJK", str(p)))
                cjk_font = "CJK"
                print(f"Registered CJK font: {p}")
                break
            except Exception as e:
                print(f"Skip font {p}: {e}")
                continue

    if cjk_font:
        styles["BodyJust"].fontName = cjk_font
        # Rebuild Chinese section with CJK — actually already added with Helvetica.
        # Re-build story is complex; instead regenerate Chinese paragraphs by
        # rebuilding the whole document with CJK as default body for CN parts.
        # Simpler approach: rebuild PDF with CJK-aware styles from the start.
        return build_pdf_with_cjk(cjk_font)

    doc.build(story)
    print(f"Wrote {out}")
    return out


def build_pdf_with_cjk(cjk_font: str):
    """Full PDF build using a registered CJK font for body text that includes Chinese."""
    REPORT.mkdir(parents=True, exist_ok=True)
    out = REPORT / "AI_Assignment2_CISC3024.pdf"

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle", parent=styles["Title"], fontSize=22, spaceAfter=12,
            alignment=TA_CENTER, textColor=colors.HexColor("#1a1a2e"), fontName=cjk_font,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSub", parent=styles["Normal"], fontSize=13, spaceAfter=6,
            alignment=TA_CENTER, textColor=colors.HexColor("#333333"), fontName=cjk_font,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SecHead", parent=styles["Heading1"], fontSize=13, spaceBefore=14,
            spaceAfter=7, textColor=colors.HexColor("#16213e"), fontName=cjk_font,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJust", parent=styles["Normal"], fontSize=10, leading=14,
            alignment=TA_JUSTIFY, spaceAfter=7, fontName=cjk_font,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption", parent=styles["Normal"], fontSize=8.5, alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"), spaceAfter=10, spaceBefore=3, fontName=cjk_font,
        )
    )

    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="AI Assignment #2 — CISC3024 Pattern Recognition",
    )
    story = []
    story.append(Spacer(1, 1.4 * inch))
    story.append(Paragraph("AI Assignment #2", styles["CoverTitle"]))
    story.append(Paragraph("CISC3024 Pattern Recognition", styles["CoverSub"]))
    story.append(Paragraph("University of Macau", styles["CoverSub"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph(
        "<b>Monte Carlo Dropout as a Bayesian Approximation<br/>for Fashion-MNIST Image Classification</b>",
        styles["CoverSub"],
    ))
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("October 2026", styles["CoverSub"]))
    story.append(Paragraph(
        "Algorithm: Gal &amp; Ghahramani, ICML 2016 (arXiv:1506.02142)", styles["CoverSub"]
    ))
    story.append(Paragraph(
        f'Source code: <link href="{SOURCE_URL}">{SOURCE_URL}</link>', styles["CoverSub"]
    ))
    story.append(PageBreak())

    # Section 1
    story.append(Paragraph("1. How the AI Finds the Algorithm", styles["SecHead"]))
    story.append(Paragraph(
        "The assignment requires a <b>recent</b> classification/recognition method based on "
        "<b>probability or Bayes theory</b> for computer vision / pattern recognition, with the AI "
        "performing algorithm search, programming, and report writing.",
        styles["BodyJust"],
    ))
    story.append(Paragraph(
        "<b>Search process.</b> Queries used included: "
        "“Bayesian neural network computer vision classification 2020–2023 MC Dropout variational inference”, "
        "“Monte Carlo Dropout Bayesian approximation Gal Ghahramani image classification”, and "
        "“Mobiny Monte Carlo DropConnect Bayesian deep networks”. Sources: arXiv, NeurIPS proceedings, "
        "and Scientific Reports.",
        styles["BodyJust"],
    ))
    story.append(Paragraph("<b>Candidates considered:</b>", styles["BodyJust"]))
    story.append(ListFlowable([
        ListItem(Paragraph(
            "<b>MC Dropout</b> (Gal &amp; Ghahramani, ICML 2016, arXiv:1506.02142) — dropout as approximate "
            "variational inference in deep Gaussian processes; still a standard Bayesian DL baseline in "
            "2020+ vision and medical-imaging benchmarks.", styles["BodyJust"])),
        ListItem(Paragraph(
            "<b>MC-DropConnect</b> (Mobiny et al., Scientific Reports, 2021; arXiv:1906.04569) — Bernoulli "
            "variational approximation on weights; validated on classification and segmentation.",
            styles["BodyJust"])),
        ListItem(Paragraph(
            "<b>Variational Structured Dropout (VSD)</b> (NeurIPS 2021) — structured variational noise; "
            "stronger theory but heavier to implement on CPU.", styles["BodyJust"])),
        ListItem(Paragraph(
            "<b>Function-space VI / MC-DropBlock</b> (NeurIPS 2022 / 2021) — excellent uncertainty work but "
            "more complex architectures and compute.", styles["BodyJust"])),
    ], bulletType="bullet"))
    story.append(Paragraph(
        "<b>Selection.</b> We chose <b>Monte Carlo Dropout</b> because (1) it has a clear Bayes / "
        "variational-inference foundation (approximate posterior over weights via Bernoulli dropout masks), "
        "(2) it applies directly to CNN image classification, (3) it remains widely used after 2020 "
        "(e.g. NeurIPS 2021 Bayesian DL benchmarks), and (4) it runs on CPU with Fashion-MNIST. "
        "MC-DropConnect (2021) was a strong recent alternative; MC Dropout was preferred for simpler "
        "reproducibility while keeping the same Bayesian interpretation.",
        styles["BodyJust"],
    ))

    # Section 2
    story.append(Paragraph("2. Algorithm Description", styles["SecHead"]))
    story.append(Paragraph(
        "<b>Reference:</b> Yarin Gal and Zoubin Ghahramani. "
        "“Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.” "
        "<i>ICML</i>, 2016, pp. 1050–1059. arXiv:1506.02142.",
        styles["BodyJust"],
    ))
    story.append(Paragraph(
        "<b>Core idea.</b> Training a neural network with dropout (and L2 weight decay) can be interpreted "
        "as approximate <b>variational inference</b> for a deep Gaussian process. The variational distribution "
        "q(ω) places Bernoulli masks on network units. Minimising the dropout objective approximates "
        "minimising KL[q(ω) || p(ω | data)].",
        styles["BodyJust"],
    ))
    story.append(Paragraph(
        "<b>Predictive inference (MC Dropout).</b> At test time, dropout stays active. We draw T stochastic "
        "forward passes with independent masks ω̂_t ~ q(ω) and average the softmax probabilities:",
        styles["BodyJust"],
    ))
    story.append(Paragraph(
        "q(y*|x*) ≈ (1/T) Σ_t Softmax(f_ω̂_t (x*))",
        styles["BodyJust"],
    ))
    story.append(Paragraph(
        "From the averaged probabilities we obtain the predicted class, negative log-likelihood (NLL), "
        "and uncertainty summaries such as <b>predictive entropy</b>. This is a probability / Bayes-theory "
        "approach for computer vision classification that also indicates when the model is unsure.",
        styles["BodyJust"],
    ))
    story.append(Paragraph(
        "<b>Why it fits this course.</b> Pattern recognition historically uses Bayes classifiers. "
        "MC Dropout brings that Bayesian predictive view to modern CNNs: visual features are learned as usual, "
        "but predictions come from an approximate posterior rather than a single point-estimate weight vector.",
        styles["BodyJust"],
    ))

    # Section 3
    story.append(Paragraph("3. How the AI Implements the Algorithm", styles["SecHead"]))
    story.append(Paragraph(
        "All code was written by the AI agent (Python + PyTorch). The student did not write source code.",
        styles["BodyJust"],
    ))
    story.append(ListFlowable([
        ListItem(Paragraph(
            "<b>model.py — BayesianCNN:</b> two conv blocks (32→64), max-pooling, FC 128→10, Dropout2d/Dropout "
            "(p=0.25). Method mc_predict(x, T) keeps dropout stochastic and averages T softmax outputs.",
            styles["BodyJust"])),
        ListItem(Paragraph(
            "<b>train_eval.py:</b> Fashion-MNIST, Adam (lr=1e-3, weight decay=1e-4), 8 epochs, batch 128; "
            "evaluates deterministic (dropout off) and MC Dropout (T=30).",
            styles["BodyJust"])),
        ListItem(Paragraph(
            "<b>metrics.py:</b> accuracy, mean NLL, Expected Calibration Error (ECE, 15 bins), confusion matrix.",
            styles["BodyJust"])),
        ListItem(Paragraph(
            "Figures: training curves, confusion matrix, reliability diagram, entropy histograms "
            "(correct vs incorrect).",
            styles["BodyJust"])),
    ], bulletType="bullet"))
    story.append(Paragraph(
        f"Public source: <link href='{SOURCE_URL}'>{SOURCE_URL}</link>.",
        styles["BodyJust"],
    ))

    # Section 4
    story.append(Paragraph("4. Experiment Settings and Results", styles["SecHead"]))
    story.append(Paragraph(
        "<b>Dataset.</b> Fashion-MNIST: 60,000 train + 10,000 test grayscale 28×28 images, 10 clothing classes.",
        styles["BodyJust"],
    ))
    settings = [
        ["Setting", "Value"],
        ["Algorithm", "MC Dropout (Gal & Ghahramani, 2016)"],
        ["Model", "CNN (~422K parameters)"],
        ["Dropout p", str(METRICS["dropout_p"])],
        ["MC samples T", str(METRICS["mc_samples_t"])],
        ["Epochs / batch", f"{METRICS['epochs']} / {METRICS['batch_size']}"],
        ["Optimizer", f"Adam, lr={METRICS['lr']}, wd={METRICS['weight_decay']}"],
        ["Device / seed", f"{METRICS['device']} / {METRICS['seed']}"],
    ]
    t = Table(settings, colWidths=[2.3 * inch, 4.1 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), cjk_font),
        ("FONTNAME", (0, 1), (-1, -1), cjk_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    metrics_rows = [
        ["Metric", "Value"],
        ["Deterministic test accuracy", f"{METRICS['deterministic_test_accuracy']*100:.2f}%"],
        ["MC Dropout test accuracy (T=30)", f"{METRICS['mc_dropout_test_accuracy']*100:.2f}%"],
        ["MC Dropout mean NLL", f"{METRICS['mc_dropout_nll']:.4f}"],
        ["Expected Calibration Error (ECE)", f"{METRICS['mc_dropout_ece']:.4f}"],
        ["Mean predictive entropy (all)", f"{METRICS['mean_predictive_entropy']:.4f}"],
        ["Mean entropy (correct)", f"{METRICS['mean_entropy_correct']:.4f}"],
        ["Mean entropy (incorrect)", f"{METRICS['mean_entropy_incorrect']:.4f}"],
    ]
    t2 = Table(metrics_rows, colWidths=[3.3 * inch, 3.1 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), cjk_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eef5ff")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Discussion.</b> MC Dropout accuracy (~91.07%) matches the deterministic CNN (~91.11%). "
        "ECE ≈ 0.020 indicates reasonably calibrated probabilities. Incorrect predictions have much higher "
        f"predictive entropy ({METRICS['mean_entropy_incorrect']:.3f}) than correct ones "
        f"({METRICS['mean_entropy_correct']:.3f}), so the Bayesian approximation flags uncertain cases.",
        styles["BodyJust"],
    ))
    add_figure(story, styles, FIG / "training_curves.png", "Figure 1. Training / validation loss and accuracy.")
    add_figure(story, styles, FIG / "confusion_matrix.png", "Figure 2. Confusion matrix (MC Dropout, T=30).")
    add_figure(story, styles, FIG / "reliability_diagram.png", "Figure 3. Reliability diagram (calibration).")
    add_figure(
        story, styles, FIG / "entropy_correct_vs_incorrect.png",
        "Figure 4. Predictive entropy: correct vs incorrect predictions.",
    )

    # Section 5 Traditional Chinese
    story.append(Paragraph(
        "5. What the Student Has Learned from This AI Assignment", styles["SecHead"]
    ))
    story.append(Paragraph(
        "（本節以學生第一人稱、繁體中文撰寫。）",
        styles["Caption"],
    ))
    for p in [
        "透過這次 AI 作業，我學到如何把課堂上的貝氏／機率概念連接到現代深度學習。以前接觸貝氏分類器時，"
        "多半是較簡單的生成模型；這次用 Monte Carlo Dropout，才理解「對權重做近似後驗推論」也可以套在 CNN 上，"
        "讓影像分類不只給出類別，還能估計模型有多不確定。",
        "我也體會到「查論文」與「能實作」之間的差別。AI 幫我搜尋 2020 年後仍常用的貝氏深度學習方法，"
        "比較 MC Dropout、MC-DropConnect、結構化變分 Dropout 等候選後，選了理論清楚、資源需求適中的方案，"
        "並在 Fashion-MNIST 上跑出真實的準確率、NLL、ECE 與混淆矩陣，而不是虛構數字。",
        "最重要的是，作業規定學生不可自行寫程式，整段實作與報告由 AI 完成。我學到如何清楚下達需求、"
        "檢查實驗設定是否合理、解讀不確定度指標（例如錯誤樣本的預測熵明顯較高），以及把原始碼整理成可公開的專案。"
        "這讓我更明白在模式識別中，機率思維不只是公式，而是可以落地成可重現實驗的工程流程。",
    ]:
        story.append(Paragraph(p, styles["BodyJust"]))

    # Section 6
    story.append(Paragraph("6. Webpage Link to the Source Codes", styles["SecHead"]))
    story.append(Paragraph(
        "All source code, experiment scripts, figures, and this report are published at:",
        styles["BodyJust"],
    ))
    story.append(Paragraph(
        f'<b><link href="{SOURCE_URL}" color="blue">{SOURCE_URL}</link></b>',
        styles["BodyJust"],
    ))
    story.append(Paragraph(
        "The repository includes src/model.py, src/train_eval.py, src/metrics.py, requirements.txt, "
        "README.md, results/, and figures/.",
        styles["BodyJust"],
    ))

    story.append(Paragraph("References", styles["SecHead"]))
    story.append(Paragraph(
        "[1] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian Approximation: Representing Model "
        "Uncertainty in Deep Learning,” ICML, 2016. https://arxiv.org/abs/1506.02142",
        styles["BodyJust"],
    ))
    story.append(Paragraph(
        "[2] A. Mobiny et al., “DropConnect is effective in modeling uncertainty of Bayesian deep "
        "networks,” Scientific Reports, 2021. https://arxiv.org/abs/1906.04569",
        styles["BodyJust"],
    ))
    story.append(Paragraph(
        "[3] H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-MNIST: a Novel Image Dataset for "
        "Benchmarking Machine Learning Algorithms,” arXiv:1708.07747, 2017.",
        styles["BodyJust"],
    ))

    doc.build(story)
    print(f"Wrote {out}")
    return out


def set_run_font(run, font_name="Microsoft YaHei", size=11, bold=False):
    run.font.size = Pt(size)
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
    run.bold = bold


def add_heading_docx(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        set_run_font(run, size=14 if level == 1 else 12, bold=True)
    return h


def add_para_docx(doc, text, size=11, bold=False, center=False):
    p = doc.add_paragraph()
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)
    p.paragraph_format.space_after = Pt(8)
    return p


def build_docx():
    REPORT.mkdir(parents=True, exist_ok=True)
    out = REPORT / "AI_Assignment2_CISC3024.docx"
    doc = Document()

    # Cover
    add_para_docx(doc, "AI Assignment #2", size=22, bold=True, center=True)
    add_para_docx(doc, "CISC3024 Pattern Recognition", size=14, center=True)
    add_para_docx(doc, "University of Macau", size=12, center=True)
    add_para_docx(
        doc,
        "Monte Carlo Dropout as a Bayesian Approximation for Fashion-MNIST Image Classification",
        size=12,
        bold=True,
        center=True,
    )
    add_para_docx(doc, "October 2026", size=12, center=True)
    add_para_docx(doc, "Algorithm: Gal & Ghahramani, ICML 2016 (arXiv:1506.02142)", size=11, center=True)
    add_para_docx(doc, f"Source code: {SOURCE_URL}", size=11, center=True)
    doc.add_page_break()

    add_heading_docx(doc, "1. How the AI Finds the Algorithm")
    add_para_docx(
        doc,
        "The assignment requires a recent classification/recognition method based on probability or "
        "Bayes theory for computer vision / pattern recognition, with the AI performing algorithm search, "
        "programming, and report writing.",
    )
    add_para_docx(
        doc,
        "Search process. Queries used included: “Bayesian neural network computer vision classification "
        "2020–2023 MC Dropout variational inference”, “Monte Carlo Dropout Bayesian approximation Gal "
        "Ghahramani image classification”, and “Mobiny Monte Carlo DropConnect Bayesian deep networks”. "
        "Sources consulted: arXiv, NeurIPS proceedings, and Scientific Reports.",
    )
    add_para_docx(
        doc,
        "Candidates: (1) MC Dropout (Gal & Ghahramani, ICML 2016, arXiv:1506.02142); "
        "(2) MC-DropConnect (Mobiny et al., Scientific Reports 2021, arXiv:1906.04569); "
        "(3) Variational Structured Dropout (NeurIPS 2021); "
        "(4) Function-space VI / MC-DropBlock (NeurIPS 2022 / 2021).",
    )
    add_para_docx(
        doc,
        "Selection. We chose Monte Carlo Dropout because it has a clear Bayes/variational foundation, "
        "applies to CNN image classification, remains widely used after 2020, and is implementable on CPU "
        "with Fashion-MNIST. MC-DropConnect (2021) was a strong recent alternative.",
    )

    add_heading_docx(doc, "2. Algorithm Description")
    add_para_docx(
        doc,
        "Reference: Yarin Gal and Zoubin Ghahramani. “Dropout as a Bayesian Approximation: Representing "
        "Model Uncertainty in Deep Learning.” ICML 2016, pp. 1050–1059. arXiv:1506.02142.",
    )
    add_para_docx(
        doc,
        "Core idea. Training with dropout (and L2 weight decay) is approximate variational inference for "
        "a deep Gaussian process. The variational distribution q(ω) uses Bernoulli masks on units. "
        "Minimising the dropout objective approximates KL[q(ω) || p(ω | data)].",
    )
    add_para_docx(
        doc,
        "Predictive inference. At test time, keep dropout on, draw T stochastic forward passes, and average "
        "softmax outputs: q(y*|x*) ≈ (1/T) Σ Softmax(f_ω̂_t(x*)). From this we get class predictions, NLL, "
        "and predictive entropy. This is a probability/Bayes method for CV classification with uncertainty.",
    )

    add_heading_docx(doc, "3. How the AI Implements the Algorithm")
    add_para_docx(
        doc,
        "All code was written by the AI (Python + PyTorch). The student did not write source code. "
        "model.py defines BayesianCNN with Dropout and mc_predict(x, T). train_eval.py trains on "
        "Fashion-MNIST (Adam, 8 epochs) and evaluates deterministic vs MC Dropout (T=30). metrics.py "
        "computes accuracy, NLL, and ECE. Figures include training curves, confusion matrix, reliability "
        f"diagram, and entropy histograms. Source: {SOURCE_URL}",
    )

    add_heading_docx(doc, "4. Experiment Settings and Results")
    add_para_docx(
        doc,
        f"Dataset: Fashion-MNIST (60k train / 10k test). Settings: dropout p={METRICS['dropout_p']}, "
        f"T={METRICS['mc_samples_t']}, epochs={METRICS['epochs']}, batch={METRICS['batch_size']}, "
        f"Adam lr={METRICS['lr']}, weight_decay={METRICS['weight_decay']}, device={METRICS['device']}.",
    )
    add_para_docx(
        doc,
        f"Results: Deterministic accuracy {METRICS['deterministic_test_accuracy']*100:.2f}%; "
        f"MC Dropout accuracy {METRICS['mc_dropout_test_accuracy']*100:.2f}%; "
        f"NLL {METRICS['mc_dropout_nll']:.4f}; ECE {METRICS['mc_dropout_ece']:.4f}; "
        f"mean entropy correct {METRICS['mean_entropy_correct']:.4f} vs incorrect "
        f"{METRICS['mean_entropy_incorrect']:.4f}.",
    )
    add_para_docx(
        doc,
        "Discussion: MC Dropout matches point-estimate accuracy while providing calibrated probabilities "
        "and higher entropy on mistakes — useful uncertainty for pattern recognition.",
    )
    for name, caption in [
        ("training_curves.png", "Figure 1. Training curves"),
        ("confusion_matrix.png", "Figure 2. Confusion matrix"),
        ("reliability_diagram.png", "Figure 3. Reliability diagram"),
        ("entropy_correct_vs_incorrect.png", "Figure 4. Predictive entropy"),
    ]:
        fp = FIG / name
        if fp.exists():
            doc.add_picture(str(fp), width=Inches(5.5))
            add_para_docx(doc, caption, size=9, center=True)

    add_heading_docx(doc, "5. What the Student Has Learned from This AI Assignment")
    add_para_docx(doc, "（本節以學生第一人稱、繁體中文撰寫。）", size=10)
    for p in [
        "透過這次 AI 作業，我學到如何把課堂上的貝氏／機率概念連接到現代深度學習。以前接觸貝氏分類器時，"
        "多半是較簡單的生成模型；這次用 Monte Carlo Dropout，才理解「對權重做近似後驗推論」也可以套在 CNN 上，"
        "讓影像分類不只給出類別，還能估計模型有多不確定。",
        "我也體會到「查論文」與「能實作」之間的差別。AI 幫我搜尋 2020 年後仍常用的貝氏深度學習方法，"
        "比較 MC Dropout、MC-DropConnect、結構化變分 Dropout 等候選後，選了理論清楚、資源需求適中的方案，"
        "並在 Fashion-MNIST 上跑出真實的準確率、NLL、ECE 與混淆矩陣，而不是虛構數字。",
        "最重要的是，作業規定學生不可自行寫程式，整段實作與報告由 AI 完成。我學到如何清楚下達需求、"
        "檢查實驗設定是否合理、解讀不確定度指標（例如錯誤樣本的預測熵明顯較高），以及把原始碼整理成可公開的專案。"
        "這讓我更明白在模式識別中，機率思維不只是公式，而是可以落地成可重現實驗的工程流程。",
    ]:
        add_para_docx(doc, p)

    add_heading_docx(doc, "6. Webpage Link to the Source Codes")
    add_para_docx(doc, "All source code for this assignment is available at:")
    add_para_docx(doc, SOURCE_URL, bold=True)
    add_para_docx(
        doc,
        "The repository includes src/, requirements.txt, README.md, results/, figures/, and report/.",
    )

    add_heading_docx(doc, "References")
    add_para_docx(
        doc,
        "[1] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian Approximation…,” ICML, 2016. "
        "https://arxiv.org/abs/1506.02142",
    )
    add_para_docx(
        doc,
        "[2] A. Mobiny et al., “DropConnect is effective in modeling uncertainty of Bayesian deep "
        "networks,” Scientific Reports, 2021. https://arxiv.org/abs/1906.04569",
    )
    add_para_docx(
        doc,
        "[3] H. Xiao et al., “Fashion-MNIST…,” arXiv:1708.07747, 2017.",
    )

    doc.save(str(out))
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    # Prefer CJK path
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import glob

    cjk_candidates = []
    cjk_candidates += glob.glob("/usr/share/fonts/**/*NotoSans*CJK*", recursive=True)
    cjk_candidates += glob.glob("/usr/share/fonts/**/*NotoSansSC*", recursive=True)
    cjk_candidates += glob.glob("/usr/share/fonts/**/*NotoSansTC*", recursive=True)
    cjk_candidates += glob.glob("/usr/share/fonts/**/*wqy*", recursive=True)
    cjk_candidates += glob.glob("/usr/share/fonts/**/*uming*", recursive=True)
    cjk_candidates += glob.glob("/usr/share/fonts/**/*DroidSansFallback*", recursive=True)
    cjk = None
    for fp in cjk_candidates:
        p = Path(fp)
        if not p.is_file():
            continue
        try:
            if p.suffix.lower() == ".ttc":
                pdfmetrics.registerFont(TTFont("CJK", str(p), subfontIndex=0))
            else:
                pdfmetrics.registerFont(TTFont("CJK", str(p)))
            cjk = "CJK"
            print("Using font", p)
            break
        except Exception as e:
            print("font fail", p, e)
    if cjk:
        build_pdf_with_cjk(cjk)
    else:
        print("WARNING: No CJK font; PDF Chinese may not render. Installing font...")
        build_pdf()
    build_docx()
