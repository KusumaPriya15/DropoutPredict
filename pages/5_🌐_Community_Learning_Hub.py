# pages/5_🌐_Community_Learning_Hub.py
import streamlit as st

# ============================================================
# 🌐 Community Learning Hub — E-Learning Platform Integration
# ============================================================
st.set_page_config(page_title="Community Learning Hub", page_icon="🌐", layout="wide")

st.title("🌐 Community Learning Hub Platform")
st.markdown("""
### Empowering Education through Digital Learning

The **Community Learning Hub** is designed to support **rural and underserved communities**
by providing **access to quality digital education, personalized tutoring, and collaborative learning spaces**.

This platform directly aligns with **NEP 2020’s goal** of reducing dropout rates through **holistic, inclusive, and technology-driven education**.
""")

st.markdown("---")

# ============================================================
# 🎯 Platform Overview
# ============================================================
st.subheader("💡 Platform Overview")

st.markdown("""
The **E-Learning Platform** provides an **adaptive learning experience** that leverages AI, analytics, and automation to personalize
the educational journey for each student and enhance the teaching experience for instructors.

**Key Features:**
- 🧠 **AI-Generated Quizzes & Assignments** using GPT-based automation  
- 📊 **Performance Analytics Dashboard** for instructors and administrators  
- 🎯 **Adaptive Learning Paths** based on student performance  
- 🏫 **Virtual Classes & Resource Library** for students in community hubs  
- 💬 **Discussion Forums** and **Virtual Mentoring** for peer & teacher engagement
""")

st.markdown("---")

# ============================================================
# 🖥️ Website Integration or Link
# ============================================================
st.subheader("🚀 Explore the Platform")

# 🔗 Option 1 — If you have a hosted website:
st.markdown("""
You can explore the live platform here:
👉 [Open E-Learning Platform](https://api-dev-1.vercel.app/)
""")

# # 🔗 Option 2 — Embed the website inside Streamlit (iframe)
# st.components.v1.iframe("https://api-dev-1.vercel.app/", height=700, scrolling=True)

st.markdown("---")

# ============================================================
# 📚 How It Supports Dropout Reduction
# ============================================================
st.subheader("📚 How This Platform Supports Dropout Reduction")

st.markdown("""
1. **Personalized Learning:**  
   Adaptive assessments and AI feedback help students learn at their own pace.

2. **Access in Underserved Areas:**  
   Community hubs use this platform to reach students without traditional schools or resources.

3. **Continuous Engagement:**  
   Virtual mentoring, gamified quizzes, and community support keep students motivated.

4. **Data-Driven Interventions:**  
   Administrators and teachers can track progress, identify learning gaps, and offer targeted help.

5. **Alignment with NEP 2020:**  
   Encourages equitable access, digital literacy, and inclusive learning for all.
""")

st.markdown("---")

st.info("""
🌟 Together with the **DropoutPredict AI system**, this platform forms a complete **ecosystem for dropout prevention** —
predicting risks early, providing learning support, and empowering communities.
""")
