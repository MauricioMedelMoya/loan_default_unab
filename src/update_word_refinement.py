from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCX_PATH = ROOT / "Documentos" / "acif104_s6_MMedel.docx"
TMP_PATH = DOCX_PATH.with_suffix(".tmp.docx")
ASSETS_DIR = ROOT / ".generated_assets"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
ET.register_namespace("w", W)


def qn(name: str) -> str:
    return f"{{{W}}}{name}"


def paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.iter(qn("t")))


def set_paragraph_text(p: ET.Element, text: str) -> None:
    runs = list(p.findall("w:r", NS))
    if not runs:
        runs = [ET.SubElement(p, qn("r"))]
    first_run = runs[0]
    for run in runs[1:]:
        p.remove(run)
    for child in list(first_run):
        if child.tag != qn("rPr"):
            first_run.remove(child)
    t = ET.SubElement(first_run, qn("t"))
    t.set(qn("space"), "preserve")
    t.text = text


def make_paragraph(text: str, bold: bool = False) -> ET.Element:
    p = ET.Element(qn("p"))
    r = ET.SubElement(p, qn("r"))
    if bold:
        rpr = ET.SubElement(r, qn("rPr"))
        ET.SubElement(rpr, qn("b"))
    t = ET.SubElement(r, qn("t"))
    t.set(qn("space"), "preserve")
    t.text = text
    return p


def make_cell(text: str, bold: bool = False) -> ET.Element:
    tc = ET.Element(qn("tc"))
    tc_pr = ET.SubElement(tc, qn("tcPr"))
    width = ET.SubElement(tc_pr, qn("tcW"))
    width.set(qn("w"), "2400")
    width.set(qn("type"), "dxa")
    tc.append(make_paragraph(str(text), bold=bold))
    return tc


def make_table(headers: list[str], rows: list[list[str]]) -> ET.Element:
    tbl = ET.Element(qn("tbl"))
    tbl_pr = ET.SubElement(tbl, qn("tblPr"))
    borders = ET.SubElement(tbl_pr, qn("tblBorders"))
    for border_name in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        border = ET.SubElement(borders, qn(border_name))
        border.set(qn("val"), "single")
        border.set(qn("sz"), "4")
        border.set(qn("space"), "0")
        border.set(qn("color"), "BFBFBF")

    header_row = ET.SubElement(tbl, qn("tr"))
    for header in headers:
        header_row.append(make_cell(header, bold=True))

    for row in rows:
        tr = ET.SubElement(tbl, qn("tr"))
        for value in row:
            tr.append(make_cell(value))
    return tbl


def body_children(root: ET.Element) -> list[ET.Element]:
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("No se encontró el cuerpo del documento.")
    return list(body)


def find_child_index(children: list[ET.Element], text: str, exact: bool = False) -> int:
    for index, child in enumerate(children):
        if child.tag != qn("p"):
            continue
        current = paragraph_text(child).strip()
        if (exact and current == text) or (not exact and text in current):
            return index
    raise ValueError(f"No se encontró el texto: {text}")


def insert_after(body: ET.Element, children: list[ET.Element], anchor_text: str, new_items: list[ET.Element], exact: bool = False) -> None:
    index = find_child_index(children, anchor_text, exact=exact)
    for offset, item in enumerate(new_items, start=1):
        body.insert(index + offset, item)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    comparison = pd.read_csv(ASSETS_DIR / "comparacion_modelos_ml_dl.csv")
    balance = pd.read_csv(ASSETS_DIR / "comparacion_balanceo.csv")
    final_metrics = pd.read_csv(ASSETS_DIR / "metricas_modelo_final.csv").iloc[0]

    with zipfile.ZipFile(DOCX_PATH, "r") as zin:
        document_xml = zin.read("word/document.xml")
        all_files = {name: zin.read(name) for name in zin.namelist() if name != "word/document.xml"}

    root = ET.fromstring(document_xml)

    for text_node in root.iter(qn("t")):
        if not text_node.text:
            continue
        text_node.text = (
            text_node.text.replace("CONCLUCIÓN", "CONCLUSIÓN")
            .replace("CONCLUCIÓn", "CONCLUSIÓN")
            .replace("A Como parte", "Como parte")
            .replace("incumplimieto", "incumplimiento")
        )

    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("No se encontró el cuerpo del documento.")

    children = body_children(root)

    insert_after(
        body,
        children,
        "METODOLOGÍA",
        [
            make_paragraph("2.0 Refinamiento iterativo y mejora del proyecto", bold=True),
            make_paragraph(
                "En esta segunda versión se mantiene la línea técnica del proyecto original, pero se amplía el análisis experimental para responder a las observaciones de la evaluación anterior. El refinamiento incorpora nuevos modelos de comparación, tres configuraciones MLP, técnicas adicionales de balanceo, ajuste de threshold y una exportación más ordenada de métricas y gráficos. Con esto se fortalece la trazabilidad del trabajo sin cambiar la arquitectura central ni reemplazar la selección final de Random Forest."
            ),
        ],
        exact=True,
    )

    children = body_children(root)
    insert_after(
        body,
        children,
        "El proyecto será desarrollado de manera individual",
        [
            make_paragraph("Tabla 1 actualizada: Plan de trabajo detallado.", bold=True),
            make_table(
                ["Fase", "Actividad", "Entregable", "Requisito asociado", "Métrica", "Responsable", "Plazo"],
                [
                    ["EDA", "Revisar variables, nulos, outliers y distribución objetivo", "Hallazgos y gráficos EDA", "Comprensión de datos", "Completitud y patrones", "Mauricio Medel", "4-6 h"],
                    ["Preprocesamiento", "Definir ColumnTransformer con imputación, scaling y One-Hot Encoding", "Pipeline reproducible", "Confiabilidad y reproducibilidad", "Sin fuga de datos", "Mauricio Medel", "3-4 h"],
                    ["Modelado", "Comparar LR, árbol, RF, Gradient Boosting y MLP", "Tabla ML vs MLP", "Selección de técnicas", "F1 y ROC-AUC", "Mauricio Medel", "5-7 h"],
                    ["Balanceo", "Evaluar sin balanceo, class_weight, undersampling y SMOTE", "Tabla de balanceo", "Tratamiento de desbalance", "Precision, recall y F1", "Mauricio Medel", "3-4 h"],
                    ["Threshold", "Probar thresholds y seleccionar el mejor por F1", "Curva y umbral óptimo", "Evaluación del modelo", "F1-score", "Mauricio Medel", "2-3 h"],
                    ["Interpretabilidad", "Actualizar SHAP global y local", "Gráficos SHAP", "Explicabilidad", "Importancia media SHAP", "Mauricio Medel", "2-3 h"],
                    ["Aplicación", "Conectar Dash con pipeline y métricas reales", "App funcional actualizada", "Frontend/backend", "Predicción y visualización", "Mauricio Medel", "3-5 h"],
                    ["Documentación", "Actualizar README e informe", "Repositorio e informe final", "GitHub y formato", "Instrucciones reproducibles", "Mauricio Medel", "3-5 h"],
                ],
            ),
        ],
    )

    children = body_children(root)
    idx = find_child_index(children, "En este sentido, si bien existen modelos más complejos")
    set_paragraph_text(
        children[idx],
        "En esta segunda iteración, los modelos MLP sí fueron incorporados como comparación experimental de deep learning mediante scikit-learn. No se seleccionan como arquitectura final porque, para este dataset tabular, Random Forest mantiene un mejor equilibrio entre desempeño, costo computacional e interpretabilidad.",
    )

    children = body_children(root)
    insert_after(
        body,
        children,
        "Modelos seleccionados",
        [
            make_paragraph("Ampliación de modelos en la segunda iteración", bold=True),
            make_paragraph(
                "Además de Regresión Logística, Árbol de Decisión y Random Forest, se incorporó Gradient Boosting como técnica ensemble adicional. También se evaluaron tres arquitecturas MLP: una red simple, una red profunda y una red regularizada. Estas configuraciones permiten evidenciar una comparación ML vs DL sin introducir TensorFlow ni PyTorch, manteniendo el alcance técnico del proyecto."
            ),
        ],
    )

    model_rows = [
        [
            row["Modelo"],
            fmt(row["Accuracy"]),
            fmt(row["Precision"]),
            fmt(row["Recall"]),
            fmt(row["F1"]),
            fmt(row["ROC_AUC"]),
            fmt(row["Tiempo_entrenamiento_seg"]),
        ]
        for _, row in comparison.iterrows()
    ]

    children = body_children(root)
    insert_after(
        body,
        children,
        "Conclusión de la comparación",
        [
            make_paragraph("Comparación ampliada ML vs DL", bold=True),
            make_paragraph(
                "La comparación ampliada confirma que Random Forest mantiene el mejor equilibrio general. Gradient Boosting obtiene resultados competitivos, mientras que las tres configuraciones MLP presentan un desempeño razonable, aunque no superan a Random Forest en F1-score ni entregan el mismo nivel de interpretabilidad para un contexto financiero."
            ),
            make_table(
                ["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Tiempo (s)"],
                model_rows,
            ),
        ],
    )

    balance_rows = [
        [
            row["Tecnica_balanceo"],
            fmt(row["Precision"]),
            fmt(row["Recall"]),
            fmt(row["F1"]),
            fmt(row["ROC_AUC"]),
        ]
        for _, row in balance.iterrows()
    ]

    children = body_children(root)
    insert_after(
        body,
        children,
        "Dado que el análisis exploratorio evidenció un desbalance",
        [
            make_paragraph("Comparación de balanceo y threshold tuning", bold=True),
            make_paragraph(
                "Además de SMOTE, se compararon class_weight y RandomUnderSampler. Los resultados muestran que el balanceo modifica principalmente el trade-off entre precision y recall. RandomUnderSampler aumenta la detección de incumplimientos, pero reduce la precision; class_weight y SMOTE mantienen un comportamiento más estable. Finalmente, se evaluaron múltiples thresholds y se seleccionó un umbral de 0.55 por maximizar el F1-score del Random Forest final."
            ),
            make_table(["Técnica", "Precision", "Recall", "F1", "ROC-AUC"], balance_rows),
        ],
    )

    frontend_texts = [
        (
            "Como parte del proyecto, se mantuvo una interfaz simple en Dash con fines demostrativos, pero se actualizó su integración para utilizar el pipeline completo exportado desde el notebook. De esta manera, la aplicación recibe variables crudas del solicitante y no depende de transformaciones manuales fuera del modelo."
        ),
        (
            "El flujo de inferencia queda definido como: usuario ingresa datos, la app valida campos, el pipeline aplica imputación, escalamiento y One-Hot Encoding, Random Forest estima la probabilidad de default y la interfaz muestra la clasificación junto con la probabilidad asociada."
        ),
        (
            "El backend de esta versión corresponde al pipeline serializado en app/model.pkl, acompañado por métricas reales leídas desde los archivos generados por el notebook. Esto mejora la coherencia entre entrenamiento, evaluación y uso de la app."
        ),
        (
            "La implementación sigue siendo una demo académica. No incluye una API desacoplada, monitoreo persistente, gestión avanzada de estados ni detección automática de drift. Estas limitaciones son consistentes con el alcance del proyecto y quedan identificadas como oportunidades futuras."
        ),
    ]

    children = body_children(root)
    start = find_child_index(children, "Como parte del proyecto, se desarrolló una interfaz simple")
    for offset, text in enumerate(frontend_texts):
        set_paragraph_text(children[start + offset], text)

    children = body_children(root)
    insert_after(
        body,
        children,
        "El código del proyecto se encuentra disponible en el siguiente repositorio de GitHub",
        [
            make_paragraph(
                "Para reforzar la reproducibilidad, el repositorio incorpora instrucciones de instalación y ejecución, un script de refinamiento para recalcular métricas y gráficos, el pipeline final serializado y una carpeta .generated_assets ignorada por Git para almacenar resultados generados automáticamente. Esto permite revisar el código fuente, ejecutar el notebook, regenerar evidencia experimental y levantar la app Dash localmente."
            )
        ],
    )

    children = body_children(root)
    insert_after(
        body,
        children,
        "CONCLUSIÓN GENERAL",
        [
            make_paragraph(
                f"Como cierre de esta segunda iteración, el proyecto evidencia una mejora metodológica concreta: se amplió la comparación de modelos, se evaluaron técnicas de balanceo, se ajustó el threshold final y se reforzó la reproducibilidad del pipeline. Random Forest se mantiene como modelo final con F1-score {final_metrics['F1']:.4f} y ROC-AUC {final_metrics['ROC_AUC']:.4f}, no por continuidad automática, sino porque conserva el mejor equilibrio entre desempeño, interpretabilidad y uso práctico sobre datos tabulares."
            )
        ],
    )

    updated_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    with zipfile.ZipFile(TMP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, content in all_files.items():
            zout.writestr(name, content)
        zout.writestr("word/document.xml", updated_xml)

    shutil.move(TMP_PATH, DOCX_PATH)
    print(f"Documento actualizado: {DOCX_PATH}")


if __name__ == "__main__":
    main()
