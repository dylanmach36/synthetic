import streamlit as st
from rdkit import Chem

import streamlit as st

page = st.navigation(
    [
        st.Page("app_pages/Home.py", title="Home"),
        st.Page("app_pages/Synthesis.py", title="Practice Synthesis")
    ]
)
page.run()