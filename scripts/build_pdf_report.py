"""
Génération du rapport synthétique professionnel PDF (report.pdf)
Palmer Penguins - Laboratoire de Recherche Écologique
Conforme au cahier des charges : 2 à 4 pages
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Palette de couleurs professionnelle
PRIMARY = colors.HexColor("#1A365D")    # Bleu marine profond
SECONDARY = colors.HexColor("#0D9488")  # Turquoise / Sarcelle écologique
ACCENT = colors.HexColor("#D97706")     # Ambre / Orange chaud
TEXT_COLOR = colors.HexColor("#1F2937") # Gris foncé textuel
BG_LIGHT = colors.HexColor("#F8FAFC")   # Fond très clair
BG_ALT = colors.HexColor("#EDF2F7")     # Fond alternatif tableau
BORDER_COLOR = colors.HexColor("#CBD5E1") # Gris bordure

class NumberedCanvas(canvas.Canvas):
    """Canvas personnalisé pour ajouter les numéros de page 'Page X sur Y' et en-têtes"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # En-tête (sauf page 1)
        if self._pageNumber > 1:
            self.drawString(1.5 * cm, A4[1] - 1.0 * cm, "Laboratoire de Recherche Écologique — Étude Palmer Penguins (Mission N° 01)")
            self.drawRightString(A4[0] - 1.5 * cm, A4[1] - 1.0 * cm, "Rapport d'Analyse Exploratoire")
            self.setStrokeColor(BORDER_COLOR)
            self.setLineWidth(0.5)
            self.line(1.5 * cm, A4[1] - 1.15 * cm, A4[0] - 1.5 * cm, A4[1] - 1.15 * cm)

        # Pied de page (toutes les pages)
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(1.5 * cm, 1.2 * cm, A4[0] - 1.5 * cm, 1.2 * cm)
        self.drawString(1.5 * cm, 0.8 * cm, "Confidentiel — Usage Exclusif Laboratoire de Recherche Écologique | Python, Pandas & Seaborn")
        self.drawRightString(A4[0] - 1.5 * cm, 0.8 * cm, f"Page {self._pageNumber} sur {page_count}")
        self.restoreState()


def build_pdf():
    pdf_filename = "report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=A4,
        leftMargin=1.3 * cm,
        rightMargin=1.3 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm
    )

    styles = getSampleStyleSheet()

    # Définition des styles personnalisés
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=SECONDARY,
        spaceAfter=6
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569")
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=6,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=SECONDARY,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=TEXT_COLOR,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=12,
        bulletIndent=4,
        spaceAfter=2
    )

    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.8,
        leading=10.5,
        textColor=PRIMARY
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9,
        textColor=TEXT_COLOR
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold'
    )

    table_cell_header = ParagraphStyle(
        'TableHeader',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=colors.white
    )

    story = []

    # =========================================================================
    # PAGE 1 : CADRE GÉNÉRAL, QUALITÉ & DESCRIPTION
    # =========================================================================
    story.append(Paragraph("RAPPORT D'ANALYSE EXPLORATOIRE DE DONNÉES", title_style))
    story.append(Paragraph("MISSION N° 01 — DIVERSITÉ MORPHOLOGIQUE ET AUDIT STATISTIQUE DES MANCHOTS DE PALMER", subtitle_style))
    story.append(Paragraph("<b>Client :</b> Laboratoire de Recherche Écologique &nbsp;|&nbsp; <b>Jeu de données :</b> Palmer Penguins (LTER) &nbsp;|&nbsp; <b>Outils :</b> Python, Pandas, NumPy, Seaborn &nbsp;|&nbsp; <b>Date :</b> Octobre 2026", meta_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceBefore=2, spaceAfter=6))

    # Bloc Résumé Exécutif dans un cadre
    exec_summary_text = (
        "<b>RÉSUMÉ EXÉCUTIF :</b> Cette étude exploratoire analyse <b>344 observations</b> de manchots réparties sur 3 îles de l'Archipel Palmer (Antarctique). "
        "Les données révèlent une séparation biométrique marquée entre les trois espèces (<i>Adelie</i>, <i>Chinstrap</i>, <i>Gentoo</i>). "
        "Le <b>manchot Papou (Gentoo)</b> se singularise par un gigantisme relatif (+37% de masse corporelle, nageoires nettement plus longues) et un bec allongé et fin. "
        "À l'inverse, <i>Adelie</i> et <i>Chinstrap</i> partagent des gabarits corporels voisins mais se différencient nettement par la morphométrie de leur bec. "
        "Un résultat capital est la découverte du <b>Paradoxe de Simpson</b> sur les dimensions du bec : alors que la corrélation globale semble négative, elle est fortement positive au sein de chaque espèce. "
        "L'audit confirme une excellente qualité globale, malgré 2 lignes entièrement vides et un déséquilibre géographique à prendre impérativement en compte lors des modélisations ultérieures."
    )
    t_summary = Table(
        [[Paragraph(exec_summary_text, callout_style)]],
        colWidths=[18.4 * cm]
    )
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, SECONDARY),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 6))

    # 1. Structure du jeu de données (F1)
    story.append(Paragraph("1. Structure du Jeu de Données & Répartition Géographique (F1)", h1_style))
    story.append(Paragraph(
        "Le jeu de données original comporte <b>344 lignes</b> et <b>7 variables</b> (4 mesures morphologiques quantitatives continues et 3 variables qualitatives nominales).",
        body_style
    ))

    # Tableau descriptif des variables
    var_data = [
        [Paragraph("Variable", table_cell_header), Paragraph("Type", table_cell_header), Paragraph("Unité", table_cell_header), Paragraph("Rôle biologique & Description", table_cell_header)],
        [Paragraph("species", table_cell_bold), Paragraph("Catégorielle", table_cell), Paragraph("-", table_cell), Paragraph("Espèce étudiée : Adelie (152), Gentoo (124), Chinstrap (68)", table_cell)],
        [Paragraph("island", table_cell_bold), Paragraph("Catégorielle", table_cell), Paragraph("-", table_cell), Paragraph("Île de nidification : Biscoe (168), Dream (124), Torgersen (52)", table_cell)],
        [Paragraph("bill_length_mm", table_cell_bold), Paragraph("Numérique (float)", table_cell), Paragraph("mm", table_cell), Paragraph("Longueur du culmen (arête supérieure du bec)", table_cell)],
        [Paragraph("bill_depth_mm", table_cell_bold), Paragraph("Numérique (float)", table_cell), Paragraph("mm", table_cell), Paragraph("Hauteur / profondeur du bec à la base", table_cell)],
        [Paragraph("flipper_length_mm", table_cell_bold), Paragraph("Numérique (float)", table_cell), Paragraph("mm", table_cell), Paragraph("Longueur de la nageoire (puissance de propulsion aquatique)", table_cell)],
        [Paragraph("body_mass_g", table_cell_bold), Paragraph("Numérique (float)", table_cell), Paragraph("g", table_cell), Paragraph("Masse corporelle globale de l'individu", table_cell)],
        [Paragraph("sex", table_cell_bold), Paragraph("Catégorielle", table_cell), Paragraph("-", table_cell), Paragraph("Dimorphisme sexuel : Male (168), Female (165), Manquant (11)", table_cell)]
    ]
    t_vars = Table(var_data, colWidths=[3.2 * cm, 2.2 * cm, 1.3 * cm, 11.7 * cm])
    t_vars.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR)
    ]))
    story.append(t_vars)
    story.append(Spacer(1, 6))

    # Tableau croisé Île x Espèce + Image côte à côte
    cross_data = [
        [Paragraph("Île / Espèce", table_cell_header), Paragraph("Adelie", table_cell_header), Paragraph("Chinstrap", table_cell_header), Paragraph("Gentoo", table_cell_header), Paragraph("Total Île", table_cell_header)],
        [Paragraph("Biscoe", table_cell_bold), Paragraph("44 (26.2%)", table_cell), Paragraph("0 (0.0%)", table_cell), Paragraph("124 (73.8%)", table_cell), Paragraph("<b>168</b>", table_cell)],
        [Paragraph("Dream", table_cell_bold), Paragraph("56 (45.2%)", table_cell), Paragraph("68 (54.8%)", table_cell), Paragraph("0 (0.0%)", table_cell), Paragraph("<b>124</b>", table_cell)],
        [Paragraph("Torgersen", table_cell_bold), Paragraph("52 (100.0%)", table_cell), Paragraph("0 (0.0%)", table_cell), Paragraph("0 (0.0%)", table_cell), Paragraph("<b>52</b>", table_cell)],
        [Paragraph("Total Espèce", table_cell_header), Paragraph("152", table_cell_header), Paragraph("68", table_cell_header), Paragraph("124", table_cell_header), Paragraph("344", table_cell_header)]
    ]
    t_cross = Table(cross_data, colWidths=[2.2 * cm, 1.7 * cm, 1.8 * cm, 1.7 * cm, 1.6 * cm])
    t_cross.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BACKGROUND', (0,-1), (-1,-1), SECONDARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR)
    ]))

    # Bloc côte à côte : Table croisée + Image Répartition géographique
    img_island = Image("figures/06_species_by_island.png", width=9.0 * cm, height=5.2 * cm)
    t_layout1 = Table([[t_cross, img_island]], colWidths=[9.2 * cm, 9.2 * cm])
    t_layout1.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_layout1)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "<b>Constat méthodologique capital :</b> L'espèce <i>Adelie</i> est ubiquiste (présente sur les 3 îles), tandis que <i>Gentoo</i> est confinée exclusivement à l'île Biscoe et <i>Chinstrap</i> à l'île Dream. "
        "Toute analyse future cherchant à mesurer un effet 'île' sera par nature partiellement confondue avec l'effet 'espèce'.",
        body_style
    ))

    # =========================================================================
    # PAGE 2 : AUDIT QUALITÉ & ANALYSE STATISTIQUE DESCRIPTIVE
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("2. Audit de Qualité des Données & Stratégie de Nettoyage (F2)", h1_style))
    story.append(Paragraph(
        "Un audit exhaustif a été mené afin de fiabiliser les données préalablement aux inférences statistiques.",
        body_style
    ))

    # Tableau qualité
    qual_data = [
        [Paragraph("Variable", table_cell_header), Paragraph("Valeurs Manquantes", table_cell_header), Paragraph("Taux (%)", table_cell_header), Paragraph("Nature du problème & Diagnostic", table_cell_header), Paragraph("Action & Recommandation", table_cell_header)],
        [Paragraph("species / island", table_cell_bold), Paragraph("0", table_cell), Paragraph("0.0 %", table_cell), Paragraph("Aucune anomalie constatée. Intégrité parfaite.", table_cell), Paragraph("Conservation intégrale.", table_cell)],
        [Paragraph("Mesures physiques (4)", table_cell_bold), Paragraph("2 individus", table_cell), Paragraph("0.58 %", table_cell), Paragraph("Lignes 3 et 339 : 4 mesures biométriques absentes.", table_cell), Paragraph("Suppression justifiée (inutilisables).", table_cell)],
        [Paragraph("sex", table_cell_bold), Paragraph("11 individus", table_cell), Paragraph("3.20 %", table_cell), Paragraph("9 manchots sans sexe (hors les 2 lignes vides).", table_cell), Paragraph("Conservation pour EDA; Imputation par KNN si ML.", table_cell)],
        [Paragraph("Doublons stricts", table_cell_bold), Paragraph("0 doublon", table_cell), Paragraph("0.0 %", table_cell), Paragraph("Chaque ligne correspond à un spécimen distinct.", table_cell), Paragraph("Aucune action requise.", table_cell)],
        [Paragraph("Valeurs impossibles", table_cell_bold), Paragraph("0 valeur", table_cell), Paragraph("0.0 %", table_cell), Paragraph("Aucune mesure négative, nulle ou hors limites réelles.", table_cell), Paragraph("Cohérence biologique validée.", table_cell)]
    ]
    t_qual = Table(qual_data, colWidths=[2.7 * cm, 2.2 * cm, 1.4 * cm, 6.6 * cm, 5.5 * cm])
    t_qual.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR)
    ]))
    story.append(t_qual)
    story.append(Spacer(1, 4))

    # Encadré Règle d'or biologique
    bio_rule_text = (
        "<b>RÈGLE D'OR SUR LES VALEURS ATYPIQUES (OUTLIERS) :</b> "
        "L'application aveugle de la règle IQR (1.5 × IQR) ou du Z-score sur la population globale signale faussement des valeurs atypiques pour la masse corporelle et la longueur des nageoires. "
        "<b>Ces observations ne sont pas des erreurs expérimentales mais le reflet de variations biologiques réelles :</b> "
        "les manchots Gentoo sont naturellement beaucoup plus corpulents. Lorsque l'analyse est conduite <i>par espèce</i>, les outliers disparaissent presque intégralement. "
        "<b>Recommandation formelle :</b> Ne jamais supprimer ces individus sans justification écologique avérée."
    )
    t_bio = Table([[Paragraph(bio_rule_text, callout_style)]], colWidths=[18.4 * cm])
    t_bio.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FEF3C7")),
        ('BOX', (0,0), (-1,-1), 1, ACCENT),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(t_bio)
    story.append(Spacer(1, 6))

    # 3. Analyse Statistique Descriptive (F3)
    story.append(Paragraph("3. Analyse Statistique Descriptive & Théorie des Métriques (F3)", h1_style))
    story.append(Paragraph(
        "Le tableau ci-dessous récapitule les indicateurs statistiques de position (tendance centrale) et de dispersion pour l'ensemble de l'échantillon (N = 342 individus valides) :",
        body_style
    ))

    # Tableau stats
    stat_data = [
        [Paragraph("Métrique physique", table_cell_header), Paragraph("Moyenne", table_cell_header), Paragraph("Médiane", table_cell_header), Paragraph("Écart-type", table_cell_header), Paragraph("Min", table_cell_header), Paragraph("Q1 (25%)", table_cell_header), Paragraph("Q3 (75%)", table_cell_header), Paragraph("Max", table_cell_header), Paragraph("Étendue", table_cell_header), Paragraph("IQR", table_cell_header)],
        [Paragraph("Longueur bec (mm)", table_cell_bold), Paragraph("43.92", table_cell), Paragraph("44.45", table_cell), Paragraph("5.46", table_cell), Paragraph("32.10", table_cell), Paragraph("39.23", table_cell), Paragraph("48.50", table_cell), Paragraph("59.60", table_cell), Paragraph("27.50", table_cell), Paragraph("9.27", table_cell)],
        [Paragraph("Profondeur bec (mm)", table_cell_bold), Paragraph("17.15", table_cell), Paragraph("17.30", table_cell), Paragraph("1.97", table_cell), Paragraph("13.10", table_cell), Paragraph("15.60", table_cell), Paragraph("18.70", table_cell), Paragraph("21.50", table_cell), Paragraph("8.40", table_cell), Paragraph("3.10", table_cell)],
        [Paragraph("Longueur nageoires (mm)", table_cell_bold), Paragraph("200.91", table_cell), Paragraph("197.00", table_cell), Paragraph("14.06", table_cell), Paragraph("172.00", table_cell), Paragraph("190.00", table_cell), Paragraph("213.00", table_cell), Paragraph("231.00", table_cell), Paragraph("59.00", table_cell), Paragraph("23.00", table_cell)],
        [Paragraph("Masse corporelle (g)", table_cell_bold), Paragraph("4201.75", table_cell), Paragraph("4050.00", table_cell), Paragraph("801.95", table_cell), Paragraph("2700.00", table_cell), Paragraph("3550.00", table_cell), Paragraph("4750.00", table_cell), Paragraph("6300.00", table_cell), Paragraph("3600.00", table_cell), Paragraph("1200.00", table_cell)]
    ]
    t_stats = Table(stat_data, colWidths=[3.2 * cm, 1.6 * cm, 1.6 * cm, 1.7 * cm, 1.4 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 1.6 * cm, 1.5 * cm])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,1), (0,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR)
    ]))
    story.append(t_stats)
    story.append(Spacer(1, 4))

    # Explications didactiques demandées par F3
    concepts_text = (
        "<b>Clés de compréhension statistique pour non-spécialistes :</b><br/>"
        "• <b>Moyenne vs Médiane :</b> La moyenne est la somme des valeurs divisée par le nombre d'observations ; elle est très sensible aux valeurs extrêmes. "
        "La médiane est la valeur centrale qui sépare l'échantillon ordonné en deux moitiés égales (50% en dessous, 50% au-dessus) ; elle est robuste aux asymétries. "
        "Pour la masse corporelle, la moyenne (4202 g) dépasse la médiane (4050 g) en raison de la queue de distribution tirée vers le haut par les manchots Gentoo.<br/>"
        "• <b>Variance vs Écart-type :</b> La variance mesure la moyenne des écarts au carré par rapport à la moyenne, s'exprimant dans le carré de l'unité de mesure (ex: g²), ce qui la rend difficile à interpréter biologiquement. "
        "L'écart-type est la racine carrée de la variance : il s'exprime dans la <i>même unité que la mesure</i> (ex: 802 g), quantifiant immédiatement la variabilité typique autour de la moyenne."
    )
    story.append(Paragraph(concepts_text, body_style))
    story.append(Spacer(1, 4))

    # Image Histogrammes des distributions
    img_hist = Image("figures/01_histograms_distributions.png", width=17.5 * cm, height=7.2 * cm)
    story.append(img_hist)

    # =========================================================================
    # PAGE 3 : COMPARAISONS INTER-ESPÈCES & PARADOXE DE SIMPSON
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Comparaisons Morphologiques & Phénomènes Complexes (F4)", h1_style))
    story.append(Paragraph(
        "L'analyse conjointe des mesures physiques met en lumière deux phénomènes fondamentaux de l'écologie évolutive :",
        body_style
    ))

    # Deux figures côte à côte : Masse corporelle et Allométrie Nageoires/Masse
    img_mass = Image("figures/05_body_mass_by_species.png", width=8.8 * cm, height=5.8 * cm)
    img_flipper = Image("figures/07_flipper_vs_bodymass.png", width=9.2 * cm, height=5.8 * cm)
    t_layout2 = Table([[img_mass, img_flipper]], colWidths=[9.0 * cm, 9.4 * cm])
    t_layout2.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_layout2)
    story.append(Spacer(1, 4))

    # Synthèse comparative par espèce
    species_comp_text = (
        "<b>1. Ségrégation de la Masse Corporelle (Figure gauche) :</b> "
        "L'espèce <i>Gentoo</i> forme un groupe totalement dissocié avec une masse moyenne de <b>5076 g</b> (± 504 g), contre <b>3701 g</b> (± 459 g) pour <i>Adelie</i> "
        "et <b>3733 g</b> (± 384 g) pour <i>Chinstrap</i>. La masse corporelle seule permet d'isoler quasi parfaitement <i>Gentoo</i> des deux autres espèces.<br/>"
        "<b>2. Allométrie Biomécanique (Figure droite) :</b> "
        "Une corrélation linéaire positive majeure lie la longueur des nageoires à la masse corporelle (<b>r = 0.87</b>). "
        "Cette forte association statistique est cohérente avec une hypothèse d'adaptation hydrodynamique (ou peut refléter des contraintes allométriques de propulsion), sans pour autant qu'une causalité mécanique directe ne puisse être affirmée à partir de seules corrélations observationnelles."
    )
    story.append(Paragraph(species_comp_text, body_style))
    story.append(Spacer(1, 6))

    # Le Paradoxe de Simpson (Figure et explication détaillée)
    story.append(Paragraph("La Découverte Clé : Le Paradoxe de Simpson sur la Morphométrie du Bec", h2_style))
    img_simpson = Image("figures/03_scatter_bill_dimensions.png", width=12.0 * cm, height=6.2 * cm)
    
    simpson_explanation = (
        "<b>Comprendre le Paradoxe de Simpson :</b><br/>"
        "Si l'on calcule la corrélation linéaire globale entre la longueur et la profondeur du bec sur l'ensemble des manchots sans distinction d'espèce, "
        "on obtient une <b>corrélation négative trompeuse de r = -0.24</b> (droite grise en pointillé). Un observateur non averti conclurait qu'un bec plus long implique un bec moins épais.<br/><br/>"
        "Cependant, dès lors que l'on segmente par espèce (couleurs orange, violette et sarcelle), <b>la corrélation s'inverse radicalement et devient fortement positive</b> au sein de chaque groupe :<br/>"
        "• <i>Adelie</i> : r = <b>+0.39</b> (becs courts et trapus)<br/>"
        "• <i>Chinstrap</i> : r = <b>+0.65</b> (becs longs et profonds)<br/>"
        "• <i>Gentoo</i> : r = <b>+0.64</b> (becs longs mais fins)<br/><br/>"
        "<b>Origine biologique :</b> <i>Gentoo</i> combine des becs très longs et très peu profonds, décalant l'ensemble de son nuage vers le bas à droite. "
        "Ignorer cette sous-structure écologique conduit à une conclusion fallacieuse. Ce cas illustre la nécessité d'une analyse multivariée rigoureuse."
    )
    
    t_simpson_box = Table(
        [[Paragraph(simpson_explanation, body_style), img_simpson]],
        colWidths=[6.8 * cm, 11.6 * cm]
    )
    t_simpson_box.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_simpson_box)

    # =========================================================================
    # PAGE 4 : MATRICE DE CORRÉLATION, CONCLUSIONS & RÉPONSES AU CLIENT
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. Matrice de Corrélation & Réponses aux Questions Clés (F4 & F5)", h1_style))

    # Matrice de corrélation côte à côte avec le tableau des coefficients
    img_corr = Image("figures/04_correlation_matrix.png", width=8.8 * cm, height=5.8 * cm)
    
    corr_comment = (
        "<b>Lecture de la matrice de Pearson :</b><br/>"
        "• <b>Masse vs Nageoires (r = 0.87) :</b> Corrélation positive très forte. C'est le couple biométrique le plus cohérent de la physiologie du manchot.<br/>"
        "• <b>Longueur bec vs Nageoires (r = 0.66) :</b> Association positive modérée à forte.<br/>"
        "• <b>Longueur bec vs Masse (r = 0.60) :</b> Association positive modérée.<br/>"
        "• <b>Profondeur bec vs Masse (r = -0.47) :</b> Corrélation négative apparente provoquée par l'influence des Gentoo (fort poids mais bec plat).<br/>"
        "• <b>Longueur vs Profondeur bec (r = -0.24) :</b> Masque le paradoxe de Simpson détaillé en page 3.<br/><br/>"
        "<i>Rappel méthodologique fondamental :</i> <b>Une corrélation n'implique pas de relation de causalité.</b> "
        "La corrélation entre longueur des nageoires et masse ne signifie pas qu'agrandir les nageoires augmente le poids : "
        "toutes deux sont sous le contrôle commun d'un facteur allométrique latent (la croissance corporelle globale de l'organisme)."
    )

    t_corr_block = Table([[img_corr, Paragraph(corr_comment, body_style)]], colWidths=[9.0 * cm, 9.4 * cm])
    t_corr_block.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_corr_block)
    story.append(Spacer(1, 6))

    # Section Réponses formelles au cahier des charges F5
    story.append(Paragraph("Synthèse des Réponses aux Questions Stratégiques du Client (F5)", h1_style))

    q1_text = (
        "<b>Q1. Quelles variables présentent les différences les plus visibles entre les espèces ?</b><br/>"
        "• <b>Masse corporelle et longueur des nageoires :</b> Séparent avec une netteté remarquable l'espèce <i>Gentoo</i> des deux autres. La combinaison conjointe [nageoire ≥ 205 mm ET masse ≥ 4400 g] classe un individu comme <i>Gentoo</i> avec 97.4% de précision et 91.1% de rappel (112 vrais positifs sur 115 prédictions), résultat calculé et vérifié directement dans le notebook.<br/>"
        "• <b>Rapport Longueur / Profondeur du bec :</b> Permet une séparation quasi parfaite entre <i>Adelie</i> et <i>Chinstrap</i> (qui sont indiscernables sur la masse seule). <i>Adelie</i> possède un bec court (< 44 mm) et épais (> 17 mm), tandis que <i>Chinstrap</i> possède un bec long (> 45 mm) et épais."
    )
    story.append(Paragraph(q1_text, body_style))
    story.append(Spacer(1, 3))

    q2_text = (
        "<b>Q2. Existe-t-il une relation entre la longueur des nageoires et la masse corporelle ?</b><br/>"
        "<b>Oui, une relation linéaire positive très forte et hautement significative (r = 0.87, p < 0.001).</b> "
        "Chaque millimètre supplémentaire de nageoire s'accompagne en moyenne d'un gain d'environ 50 g de masse corporelle. Cette corrélation positive se vérifie également au sein de chaque espèce prise individuellement (Gentoo: r = 0.70; Chinstrap: r = 0.64; Adelie: r = 0.47), comme calculé explicitement dans le notebook."
    )
    story.append(Paragraph(q2_text, body_style))
    story.append(Spacer(1, 3))

    q3_text = (
        "<b>Q3. Quelles variables semblent corrélées ?</b><br/>"
        "Les paires fortement corrélées positivement sont [<i>flipper_length_mm</i> & <i>body_mass_g</i>], [<i>bill_length_mm</i> & <i>flipper_length_mm</i>], et [<i>bill_length_mm</i> & <i>body_mass_g</i>]. "
        "Au sein de chaque espèce, [<i>bill_length_mm</i> & <i>bill_depth_mm</i>] sont également corrélées positivement (r = 0.39 à 0.65)."
    )
    story.append(Paragraph(q3_text, body_style))
    story.append(Spacer(1, 3))

    q4_text = (
        "<b>Q4. Quels problèmes de qualité des données pourraient affecter une future modélisation prédictive ?</b><br/>"
        "1. <b>Les 11 valeurs manquantes sur la variable <i>sex</i> (3.2%) :</b> Le sexe introduit un dimorphisme intraspécifique significatif (les mâles sont en moyenne 10 à 15% plus lourds). L'omission ou une mauvaise imputation de cette variable affaiblira les classifieurs.<br/>"
        "2. <b>Le déséquilibre écologique d'échantillonnage par île :</b> Puisque <i>Gentoo</i> n'a été recensé que sur Biscoe et <i>Chinstrap</i> sur Dream, tout algorithme prédictif incluant l'île risque de créer un <i>biais de raccourci</i> (apprendre l'île plutôt que la véritable morphologie biologique de l'espèce)."
    )
    story.append(Paragraph(q4_text, body_style))
    story.append(Spacer(1, 5))

    # Recommandations finales et signature
    recom_box_text = (
        "<b>RECOMMANDATIONS DU CONSULTANT POUR LA PHASE 2 (MODÉLISATION) :</b><br/>"
        "1. <b>Nettoyage :</b> Écarter les 2 observations sans aucune mesure physique (n=342 exploitables).<br/>"
        "2. <b>Imputation :</b> Estimer le sexe manquant par un algorithme k-NN basé sur les 4 mesures physiques par espèce.<br/>"
        "3. <b>Feature Engineering :</b> Créer le ratio 'longueur / profondeur de bec' qui constitue un descripteur discriminant exceptionnel.<br/>"
        "4. <b>Validation :</b> Évaluer les modèles de classification sans la variable 'island' pour garantir la généralisation sur d'autres sites."
    )
    t_recom = Table([[Paragraph(recom_box_text, callout_style)]], colWidths=[18.4 * cm])
    t_recom.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, PRIMARY),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(t_recom)

    # Génération du document
    doc.build(story, canvasmaker=NumberedCanvas)
    print("Rapport PDF généré avec succès : report.pdf")

if __name__ == "__main__":
    build_pdf()
