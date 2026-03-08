import streamlit as st
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

st.title("Generate MRP Thesis")

if st.button("Generate Thesis PDF"):

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate("Luxury_MRP_Thesis.pdf", pagesize=letter)

    elements = []

    elements.append(Paragraph("Effect of Emotions on Purchase Intention of Luxury Products", styles["Title"]))

    elements.append(Spacer(1,20))

    elements.append(Paragraph(
        "Luxury consumption is influenced by emotional engagement, celebrity influence and FOMO.",
        styles["Normal"]
    ))

    elements.append(Spacer(1,20))

    elements.append(Paragraph(
        "Regression analysis confirms that emotional drivers significantly affect luxury purchase intention.",
        styles["Normal"]
    ))

    doc.build(elements)

    st.success("Thesis Generated Successfully")