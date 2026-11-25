import base64
import math
from io import BytesIO
from pathlib import Path
import os, tempfile

# Ensure Matplotlib can write its cache even on read‑only user profiles
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from flask import Flask, make_response, render_template, request
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Configuration and constants
# ------------------------------------------------------------------
APP_DIRECTORY = Path(__file__).resolve().parent
DATASET_PATH = APP_DIRECTORY / 'breast-cancer-data.csv'

FEATURES = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'smoothness_mean',
    'compactness_mean',
]

FEATURE_DETAILS = [
    {
        'name': 'radius_mean',
        'label': 'Mean radius',
        'unit': 'mm',
        'description': 'Average distance from the cell center to its edge.',
        'decimals': 2,
    },
    {
        'name': 'texture_mean',
        'label': 'Mean texture',
        'unit': 'score',
        'description': 'Average variation in grayscale values in the image.',
        'decimals': 2,
    },
    {
        'name': 'perimeter_mean',
        'label': 'Mean perimeter',
        'unit': 'mm',
        'description': 'Average perimeter of the cell outline.',
        'decimals': 2,
    },
    {
        'name': 'smoothness_mean',
        'label': 'Mean smoothness',
        'unit': 'score',
        'description': 'Average local variation in the cell outline radius.',
        'decimals': 4,
    },
    {
        'name': 'compactness_mean',
        'label': 'Mean compactness',
        'unit': 'score',
        'description': 'A measure describing how compact the cell outline is.',
        'decimals': 4,
    },
]

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _positive_class_shap_values(values):
    """Return SHAP values for the positive (malignant) class."""
    values = np.asarray(values)
    if values.ndim == 3:        # multi‑class output
        return values[:, :, 1]
    if values.ndim == 2:        # binary output
        return values
    raise ValueError(f'Unexpected SHAP output shape: {values.shape}')

def _plot_data_uri(draw_func):
    """Create a matplotlib figure using *draw_func* and return a data URI."""
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    try:
        draw_func(fig, ax)
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=130, facecolor='#ffffff')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
    finally:
        plt.close(fig)

# ------------------------------------------------------------------
# Data loading & model training
# ------------------------------------------------------------------

def load_project():
    """Load the CSV, train models, and compute all artefacts used by the UI."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f'Dataset not found at {DATASET_PATH}. Keep "breast-cancer-data.csv" beside app.py.'
        )

    df = pd.read_csv(DATASET_PATH)
    required = FEATURES + ['diagnosis']
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f'Dataset is missing required columns: {", ".join(missing)}')

    df = df[required].dropna()
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})
    df = df.dropna(subset=['diagnosis'])
    if df.empty or df['diagnosis'].nunique() != 2:
        raise ValueError('Dataset must contain both B (benign) and M (malignant) examples.')

    X = df[FEATURES]
    y = df['diagnosis'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Base model used for predictions on the UI
    base_model = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    base_model.fit(X_train, y_train)

    # Candidate models for the comparison table
    candidates = {
        'Random Forest': base_model,
        'Logistic Regression': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42)
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
    }

    comparison = []
    roc_data = []
    for name, mdl in candidates.items():
        if mdl is not base_model:
            mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)
        prob = mdl.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        metrics = {
            'name': name,
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, prob),
        }
        comparison.append(metrics)
        roc_data.append((name, fpr, tpr, metrics['roc_auc']))

    # Overall metrics for the UI (based on the Random Forest)
    test_pred = base_model.predict(X_test)
    test_prob = base_model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    overall_metrics = {
        'accuracy': accuracy_score(y_test, test_pred),
        'precision': precision_score(y_test, test_pred, zero_division=0),
        'recall': recall_score(y_test, test_pred, zero_division=0),
        'f1': f1_score(y_test, test_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, test_prob),
        'test_size': len(y_test),
        'true_benign_predicted_benign': int(cm[0, 0]),
        'true_benign_predicted_malignant': int(cm[0, 1]),
        'true_malignant_predicted_benign': int(cm[1, 0]),
        'true_malignant_predicted_malignant': int(cm[1, 1]),
    }

    counts = {
        'total': len(df),
        'benign': int((y == 0).sum()),
        'malignant': int((y == 1).sum()),
        'training': len(X_train),
        'test': len(X_test),
    }

    feature_ranges = {
        f: (float(X[f].min()), float(X[f].max())) for f in FEATURES
    }

    # Representative examples (closest to median per class)
    examples = {}
    for label, diag in (('benign', 0), ('malignant', 1)):
        class_rows = X.loc[y == diag]
        center = class_rows.median()
        dists = ((class_rows - center) / X.std()).pow(2).sum(axis=1)
        rep = class_rows.loc[dists.idxmin()]
        examples[label] = {f: float(rep[f]) for f in FEATURES}

    # Feature importances from the Random Forest
    importances = sorted(
        zip(FEATURES, base_model.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )

    correlation = X.corr()

    # Rows for the dataset explorer (JSON‑encoded for the front‑end)
    explorer_rows = [
        {
            'diagnosis': 'Malignant' if int(lbl) == 1 else 'Benign',
            **{f: float(row[f]) for f in FEATURES},
        }
        for (_, row), lbl in zip(X.iterrows(), y)
    ]

    # Global SHAP values for the Random Forest
    explainer = shap.TreeExplainer(base_model)
    shap_vals = _positive_class_shap_values(explainer(X_test).values)
    global_shap = sorted(
        zip(FEATURES, np.abs(shap_vals).mean(axis=0).tolist()),
        key=lambda kv: kv[1],
        reverse=True,
    )

    return (
        base_model,
        overall_metrics,
        counts,
        importances,
        feature_ranges,
        examples,
        comparison,
        roc_data,
        correlation,
        global_shap,
        explainer,
        explorer_rows,
    )

# Load everything once at start‑up
(
    model,
    metrics,
    dataset_counts,
    feature_importances,
    feature_ranges,
    examples,
    model_comparison,
    roc_data,
    feature_correlation,
    global_shap,
    shap_explainer,
    explorer_rows,
) = load_project()

# ------------------------------------------------------------------
# Plotting helpers (pre‑rendered static images)
# ------------------------------------------------------------------

def _roc_plot():
    palette = ['#7653a6', '#60856c', '#b46b88']

    def draw(fig, ax):
        for (name, fpr, tpr, auc), color in zip(roc_data, palette):
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{name} (AUC {auc:.3f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='#b9afc2', linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1.02),
               xlabel='False positive rate', ylabel='True positive rate')
        ax.grid(alpha=0.18)
        ax.legend(loc='lower right', frameon=False, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8, colors='#71677c')

    return _plot_data_uri(draw)

def _correlation_plot():
    values = feature_correlation.to_numpy()
    labels = [f.replace('_mean', '').replace('_', ' ').title() for f in FEATURES]

    def draw(fig, ax):
        im = ax.imshow(values, cmap='Purples', vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=35,
                       ha='right', fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f'{values[i, j]:.2f}',
                        ha='center', va='center',
                        fontsize=8, color='#30243c')
        fig.colorbar(im, ax=ax, shrink=.8,
                     label='Pearson correlation')
        ax.tick_params(length=0)

    return _plot_data_uri(draw)

# Render static images once (they do not depend on user input)
roc_plot = _roc_plot()
correlation_plot = _correlation_plot()
global_shap_chart = _plot_data_uri(
    lambda fig, ax: (
        ax.barh(
            [name.replace('_', ' ') for name, _ in reversed(global_shap)],
            [value for _, value in reversed(global_shap)],
            color='#7653a6',
        ),
        ax.set_xlabel('Mean absolute SHAP value (model output scale)', fontsize=8),
        ax.tick_params(labelsize=8, colors='#71677c'),
        ax.spines[['top', 'right', 'left']].set_visible(False),
        ax.grid(axis='x', alpha=.18),
        ax.set_axisbelow(True),
    )
)

# ------------------------------------------------------------------
# Flask web app
# ------------------------------------------------------------------

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route('/', methods=['GET', 'POST'])
def home():
    values = {f: '' for f in FEATURES}
    error = None
    result = None

    if request.method == 'POST':
        values = {f: request.form.get(f, '').strip() for f in FEATURES}
        try:
            numbers = {}
            for f in FEATURES:
                if not values[f]:
                    raise ValueError('Please enter a value for every feature.')
                num = float(values[f])
                if not math.isfinite(num) or num < 0:
                    raise ValueError("Enter valid, non-negative numbers for all features.")
                numbers[f] = num

            df_input = pd.DataFrame([numbers], columns=FEATURES)
            pred_class = int(model.predict(df_input)[0])
            votes = model.predict_proba(df_input)[0]

            # Local SHAP contribution for this single sample
            local_shap = _positive_class_shap_values(
                shap_explainer(df_input).values
            )[0]

            shap_rows = sorted(
                [
                    {
                        'name': f,
                        'label': f.replace('_', ' '),
                        'value': numbers[f],
                        'impact': float(imp),
                    }
                    for f, imp in zip(FEATURES, local_shap)
                ],
                key=lambda item: abs(item['impact']),
                reverse=True,
            )

            out_of_range = [
                f for f in FEATURES
                if numbers[f] < feature_ranges[f][0] or numbers[f] > feature_ranges[f][1]
            ]

            result = {
                'label': 'Malignant-like' if pred_class == 1 else 'Benign-like',
                'description': (
                    'The model assigned this example to the malignant class.'
                    if pred_class == 1 else
                    'The model assigned this example to the benign class.'
                ),
                'benign_vote': float(votes[0]),
                'malignant_vote': float(votes[1]),
                'predicted_vote': float(votes[pred_class]),
                'is_malignant': pred_class == 1,
                'out_of_range': out_of_range,
                'shap_rows': shap_rows,
            }
        except ValueError as exc:
            error = str(exc)

    return render_template(
        'home.html',
        features=FEATURE_DETAILS,
        values=values,
        error=error,
        result=result,
        metrics=metrics,
        dataset_counts=dataset_counts,
        feature_importances=feature_importances,
        feature_ranges=feature_ranges,
        examples=examples,
        model_comparison=model_comparison,
        roc_plot=roc_plot,
        correlation_plot=correlation_plot,
        global_shap=global_shap,
        global_shap_chart=global_shap_chart,
        explorer_rows=explorer_rows,
    )

@app.route('/report')
def download_report():
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title='CellScope Educational Model Report',
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name='PurpleHeading',
            parent=styles['Heading1'],
            textColor=colors.HexColor('#5e3d8e'),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name='SmallNote',
            parent=styles['BodyText'],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#71677c'),
        )
    )
    story = [
        Paragraph('CellScope | Breast Cancer Classification', styles['PurpleHeading']),
        Paragraph(
            'Educational project report · Wisconsin diagnostic dataset · 80/20 stratified holdout split',
            styles['BodyText'],
        ),
        Spacer(1, 12),
        Paragraph('Model comparison', styles['Heading2']),
    ]

    # Comparison table
    table_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC‑AUC']]
    for entry in model_comparison:
        table_data.append(
            [
                entry['name'],
                *[f'{entry[k] * 100:.1f}%' for k in ('accuracy', 'precision', 'recall', 'f1')],
                f'{entry["roc_auc"]:.3f}',
            ]
        )
    tbl = Table(table_data, repeatRows=1, hAlign='LEFT')
    tbl.setStyle(
        TableStyle(
            [
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7653a6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f1f8')]),
                ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#e8e1ef')),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('PADDING', (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([tbl, Spacer(1, 15), Paragraph('Random Forest evaluation', styles['Heading2'])])

    # Metrics paragraph
    story.append(
        Paragraph(
            f'Accuracy: {metrics["accuracy"]:.1%} · Precision: {metrics["precision"]:.1%} · '
            f'Recall: {metrics["recall"]:.1%} · F1: {metrics["f1"]:.1%} · '
            f'ROC‑AUC: {metrics["roc_auc"]:.3f}',
            styles['BodyText'],
        )
    )
    story.append(
        Paragraph(
            f'Dataset: {dataset_counts["total"]} records '
            f'({dataset_counts["benign"]} benign, {dataset_counts["malignant"]} malignant); '
            f'{dataset_counts["training"]} training and {dataset_counts["test"]} test records.',
            styles['BodyText'],
        )
    )
    story.append(Spacer(1, 14))
    story.append(Paragraph('Top global SHAP features', styles['Heading2']))

    # SHAP table
    shap_tbl = [['Feature', 'Mean absolute SHAP']]
    shap_tbl.extend([[name, f'{val:.5f}'] for name, val in global_shap])
    story.append(
        Table(
            shap_tbl,
            hAlign='LEFT',
            style=TableStyle(
                [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3eff8')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#5e3d8e')),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#e8e1ef')),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('PADDING', (0, 0), (-1, -1), 6),
                ]
            ),
        )
    )
    story.append(Spacer(1, 15))
    story.append(
        Paragraph(
            'For education only. This model is not a medical device and must not be used for '
            'screening, diagnosis, treatment decisions, or personal health assessment. '
            'Test‑set metrics are not evidence of clinical safety or real‑world performance.',
            styles['SmallNote'],
        )
    )

    doc.build(story)
    resp = make_response(buffer.getvalue())
    resp.headers['Content-Type'] = 'application/pdf'
    resp.headers['Content-Disposition'] = 'attachment; filename="cellscope-model-report.pdf"'
    return resp

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8001, debug=False)