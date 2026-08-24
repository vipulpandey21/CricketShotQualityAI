# -*- coding: utf-8 -*-
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 ListFlowable, ListItem, Table, TableStyle,
                                 HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

OUT = "cricket_ai_summary.pdf"

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleBig", parent=styles["Title"], fontSize=20, spaceAfter=4,
    textColor=colors.HexColor("#0F172A"))

subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"], fontSize=11,
    textColor=colors.HexColor("#64748B"), spaceAfter=14)

h2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=14, spaceBefore=18,
    spaceAfter=8, textColor=colors.HexColor("#0F172A"))

body = ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=10.5, leading=15,
    spaceAfter=6, textColor=colors.HexColor("#1E293B"), alignment=TA_LEFT)

bullet = ParagraphStyle(
    "Bullet", parent=body, leftIndent=0, spaceAfter=4)

small = ParagraphStyle(
    "Small", parent=body, fontSize=9, textColor=colors.HexColor("#64748B"))

def P(text, style=body):
    return Paragraph(text, style)

def bullets(items):
    return ListFlowable(
        [ListItem(P(t, bullet), leftIndent=14) for t in items],
        bulletType="bullet", start="•", leftIndent=14, spaceAfter=8,
    )

doc = SimpleDocTemplate(
    OUT, pagesize=letter,
    topMargin=0.7*inch, bottomMargin=0.7*inch,
    leftMargin=0.75*inch, rightMargin=0.75*inch,
)

story = []

story.append(P("Cricket Shot AI — What We Did", title_style))
story.append(P("A simple summary of the whole project, start to finish.", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0")))
story.append(Spacer(1, 10))

# ── Section 1 ────────────────────────────────────────────────────────────
story.append(P("1. The Big Goal", h2))
story.append(P("Here's the idea, plain and simple.", body))
story.append(P("You upload a cricket batting video.", body))
story.append(P("The app finds the batsman on strike.", body))
story.append(P("It guesses the shot type.", body))
story.append(P("It checks your form against pro players.", body))
story.append(P("You get a full report you can download.", body))
story.append(P("That's the whole app, in one line.", body))

# ── Section 2 ────────────────────────────────────────────────────────────
story.append(P("2. The Groundwork (done earlier)", h2))
story.append(P("Before any of this new work, a lot got fixed already:", body))
story.append(bullets([
    "The app now tracks the right player, every time. Not the bowler. Not the keeper. The batsman actually facing the ball.",
    "This works even on shaky, cropped, zoomed, or low-quality video.",
    "The app finds the exact moment of impact, when bat hits ball, using hand speed. This took real tuning to get right.",
    "Left-handed batsmen are handled correctly now. Before, the app only understood right-handers.",
    "The whole look of the app got a redesign. Dark theme, clean icons, no emojis, feels professional.",
]))

# ── Section 3 ────────────────────────────────────────────────────────────
story.append(P("3. The Big New Feature: Movement Graph", h2))
story.append(P("This was the main ask this time.", body))
story.append(P("Before, \"You vs Professionals\" only checked one single moment. Just the impact frame.", body))
story.append(P("Sir wanted more than that. He wanted the whole shot compared, not just one snapshot.", body))
story.append(P("So that's what we built.", body))
story.append(bullets([
    "The app now tracks your shot from the moment it starts, all the way to impact.",
    "It watches your knee, elbow, hip, and body-lean angles change over time, not just at one point.",
    "It lines this up against how real pro players move through the same shot.",
    "You see a graph now, not just a number. Your line, next to the pro range.",
]))
story.append(P("This is live in the app today. We tested it on real videos and it works correctly.", body))

# ── Section 4 ────────────────────────────────────────────────────────────
story.append(P("4. Trying to Boost Shot-Type Accuracy", h2))
story.append(P("The app also guesses what shot you played. Right now, it gets it right 62 times out of 100. And it gets it in the top 3 guesses 81 times out of 100.", body))
story.append(P("We tried hard to push that number higher. Here's the honest list of what we tried:", body))

data = [
    ["What we tried", "Result"],
    ["A new AI model that reads the skeleton directly (ST-GCN)", "Did not beat the current model"],
    ["Training with extra \"fairness\" weighting", "No change — data was already fair"],
    ["Combining two trained models together", "No real change"],
    ["Looking at more of the video, not just the start", "Made it worse"],
    ["Fine-tuning the AI's \"eyes\" on more training", "Made it worse — it memorized instead of learning"],
]
tbl = Table(data, colWidths=[3.4*inch, 2.5*inch])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
]))
story.append(Spacer(1, 4))
story.append(tbl)
story.append(Spacer(1, 10))

story.append(P("We tried 7 different things in total, across two work sessions.", body))
story.append(P("None of them beat the current model.", body))
story.append(P("<b>That's not bad luck. That's a clear signal.</b>", body))
story.append(P("The real problem is simple: we don't have enough training videos. Only 1250 clips for 10 shot types.", body))
story.append(P("A smarter AI trick can't fix a small dataset. More real videos would help. That's the actual next step, if we want higher accuracy.", body))

# ── Section 5 ────────────────────────────────────────────────────────────
story.append(P("5. Where Things Stand Now", h2))
story.append(bullets([
    "Shot-type accuracy stays at 62% (best guess) and 81% (top 3 guesses). Honest number, nothing hidden.",
    "We did not force in a worse model just to say something changed. If it didn't help, it didn't ship.",
    "The new movement graph feature is built, tested, and live in the app right now.",
    "Every experiment, every result, and every reason is saved and written down, in case more video data shows up later.",
]))

story.append(Spacer(1, 16))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0")))
story.append(Spacer(1, 6))
story.append(P("Simple version: the shot-analysis part got a real upgrade. The shot-guessing part is already near its limit, given the data we have.", small))

doc.build(story)
print("saved", OUT)
