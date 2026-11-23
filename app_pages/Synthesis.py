import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pubchempy as pcp
import networkx as nx
import matplotlib.pyplot as plt
import random
import re
import rdchiral
from pyvis.network import Network

st.set_page_config(layout="wide")

# -------------------------
# ALKANES
# -------------------------

seed_molecules = [

    # Alkanes
    "C",                # methane
    "CC",               # ethane
    "CCC",              # propane
    "CCCC",             # butane
    "CCCCC",            # pentane
    "CC(C)C",           # isobutane
    "CC(C)(C)C",        # neopentane
    "C1CCCCC1",         # cyclohexane

    # Alkenes
    "C=C",              # ethene
    "CC=C",             # propene
    "CCC=C",            # 1-butene
    "CC=CC",            # 2-butene
    "C=CCCC",           # 1-pentene 
    "CC=CCC",           # e-pentene
    "C1CC=CC1",         # cyclopentene
    "C1CCC=CC1",        # cyclohexene
    "C1CCC=CCC1",       # cycloheptene
    "c1ccccc1C=C",      # styrene

    # Alkynes
    "C#C",              # acetylene
    "CC#C",             # propyne
    "CCC#C",            # 1-butyne
    "c1ccccc1C#C",      # phenylacetylene

    # Alcohols
    "CO",               # methanol
    "CCO",              # ethanol
    "CCCO",             # 1-propanol
    "CCCCO",            # 1-butanol
    "CC(O)C",           # isopropanol
    "CC(C)O",           # 2-butanol
    "CC(C)(C)O",        # tert-butanol
    "Oc1ccccc1",        # phenol

    # Alkyl halides
    "CBr",              # bromomethane
    "CCBr",             # bromoethane
    "CCCBr",            # 1-bromopropane
    "CC(C)Br",          # 2-bromopropane
    "CCl",              # chloromethane
    "CCCl",             # chloroethane

    # Aldehydes
    "C=O",              # formaldehyde
    "CC=O",             # acetaldehyde
    "c1ccccc1C=O",      # benzaldehyde

    # Ketones
    "CC(=O)C",          # acetone
    "CCC(=O)C",         # 2-butanone
    "CC(=O)c1ccccc1",   # acetophenone

    # Carboxylic acids
    "C(=O)O",           # formic acid
    "CC(=O)O",          # acetic acid
    "c1ccccc1C(=O)O",   # benzoic acid

    # Esters
    "CC(=O)OC",         # methyl acetate
    "CC(=O)OCC",        # ethyl acetate

    # Amines
    "CN",               # methylamine
    "CCN",              # ethylamine
    "c1ccccc1N",        # aniline

    # Ethers
    "COC",              # dimethyl ether
    "CCOCC",            # diethyl ether

    # Epoxide
    "C1CO1",            # ethylene oxide
    "CC1CO1",           # propylene oxide

    # Other Aromatics
    "c1ccccc1",         # benzene
    "c1ccc(cc1)N"       # aniline
]


REACTIONS = {

    # ==================================================
    # ALKYL HALIDE CREATION
    # ==================================================

    "PBr3": {
        "type": "substitution",
        "reagent": "PBr3",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1][OH:2]>>[C:1]Br"
        )
    },

    "SOCl2": {
        "type": "substitution",
        "reagent": "SOCl2",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1][OH:2]>>[C:1]Cl"
        )
    },

    # ==================================================
    # SN2
    # ==================================================

    "SN2 with NaOH": {
        "type": "substitution",
        "reagent": "NaOH",
        "rxn": AllChem.ReactionFromSmarts(
            "[C;X4:1]-[$([Cl,Br,I]),$([O]S(=O)(=O)[C,c])]>>[C:1]O"
        )
    },

    "SN2 with CN": {
        "type": "substitution",
        "reagent": "NaCN",
        "rxn": AllChem.ReactionFromSmarts(
            "[C;X4:1]-[$([Cl,Br,I]),$([O]S(=O)(=O)[C,c])]>>[C:1]C#N"
        )
    },

    "SN2 with N3": {
        "type": "substitution",
        "reagent": "NaN3",
        "rxn": AllChem.ReactionFromSmarts(
            "[C;X4:1]-[$([Cl,Br,I]),$([O]S(=O)(=O)[C,c])]>>[C:1][N-]=[N+]=N"
        )
    },

    # ==================================================
    # PROTECTION
    # ==================================================

    "Protection with OTs": {
        "type": "substitution",
        "reagent": "TsCl, pyridine",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1]-[OH:2]>>[C:1]-[O:2]S(=O)(=O)c1ccc(C)cc1"
        )
    },
    
    # ==================================================
    # ELIMINATION
    # ==================================================

    "E2 Bulky Base": {
        "type": "elimination",
        "reagent": "KOtBu",
        "rxn": AllChem.ReactionFromSmarts(
            "[C;H1,H2,H3:2]-[C:1]-[$([Cl,Br,I]),$([O]S(=O)(=O)[C,c])]>>[C:2]=[C:1]"
        )
    },

    # ==================================================
    # ALKENE ADDITIONS
    # ==================================================

    "HBr Addition": {
        "type": "alkene_addition",
        "reagent": "HBr",
        "rxn": AllChem.ReactionFromSmarts(
            "[CH2:1]=[CH2:2]>>[CH2:1][CH2:2]Br"
        )
    },

    "Hydration": {
        "type": "alkene_addition",
        "reagent": "H3O+",
        "rxn": AllChem.ReactionFromSmarts(
            "[CH2:1]=[CH2:2]>>[CH2:1][CH:2](O)"
        )
    },

    "Hydrogenation": {
        "type": "reduction",
        "reagent": "H2/Pd",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1]=[C:2]>>[C:1][C:2]"
        )
    },

    "Br2 Addition": {
        "type": "alkene_addition",
        "reagent": "Br2",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1]=[C:2]>>[C:1](Br)[C:2](Br)"
        )
    },

    "Cl2 Addition": {
        "type": "alkene_addition",
        "reagent": "Cl2",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1]=[C:2]>>[C:1](Cl)[C:2](Cl)"
        )
    },

    "I2 Addition": {
        "type": "alkene_addition",
        "reagent": "I2",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1]=[C:2]>>[C:1](I)[C:2](I)"
        )
    },

    "Syn Dihydroxylation": {
        "type": "alkene_addition",
        "reagent": "OsO4/NMO",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1]=[C:2]>>O[C:1][C:2]O"
        )
    },

    # ==================================================
    # OXIDATION
    # ==================================================

    "PCC, Primary": {
        "type": "oxidation",
        "reagent": "PCC",
        "rxn": AllChem.ReactionFromSmarts(
            "[CH2:1][OX2H1]>>[CH:1]=O"
        )
    },

    "Jones, Primary": {
        "type": "oxidation",
        "reagent": "CrO3/H2SO4",
        "rxn": AllChem.ReactionFromSmarts(
            "[CH2:1][OX2H1]>>[C:1](=O)O"
        )
    },

    "Jones, Secondary": {
        "type": "oxidation",
        "reagent": "CrO3/H2SO4",
        "rxn": AllChem.ReactionFromSmarts(
            "[CH:1]([OX2H1])>>[C:1](=O)"
        )
    },

    # ==================================================
    # REDUCTION
    # ==================================================

    "NaBH4, Aldehyde": {
        "type": "reduction",
        "reagent": "NaBH4",
        "rxn": AllChem.ReactionFromSmarts(
            "[CH:1]=O>>[CH2:1]O"
        )
    },

    "NaBH4, Ketone": {
        "type": "reduction",
        "reagent": "NaBH4",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1](=O)>>[CH:1](O)"
        )
    },

    "LiAlH4, Acid": {
        "type": "reduction",
        "reagent": "LiAlH4",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1](=O)O>>[CH2:1]O"
        )
    },

    "LiAlH4, Ester": {
        "type": "reduction",
        "reagent": "LiAlH4",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1](=O)O[C]>>[CH2:1]O"
        )
    },

    "Reduction of Nitrobenzene": {
        "type": "reduction",
        "reagent": "Zn, Hcl",
        "rxn": AllChem.ReactionFromSmarts(
            "[c:1]1[c:2][c:3]([N+](=O)[O-])[c:4][c:5][c:6]1>>c1ccc(cc1)N"
        )
    },

    # ==================================================
    # ETHER SYNTHESIS
    # ==================================================

    "Williamson": {
        "type": "ether_synthesis",
        "reagent": "RO-, RBr",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1]O.[C:2]Br>>[C:1]O[C:2]"
        )
    },

    # ==================================================
    # ESTERIFICATION
    # ==================================================

    "Fischer Esterification": {
        "type": "esterification",
        "reagent": "ROH, H+",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1](=O)O.[O:2][C:3]>>[C:1](=O)O[C:3]"
        )
    },

    # ==================================================
    # EPOXIDE CHEMISTRY
    # ==================================================

    "Epoxidation": {
        "type": "epoxidation",
        "reagent": "mCPBA",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1]=[C:2]>>[C:1]1O[C:2]1"
        )
    },

    "Epoxide Hydrolysis": {
        "type": "ring_opening",
        "reagent": "H3O+",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1]1O[C:2]1>>[C:1](O)[C:2](O)"
        )
    },

    # ==================================================
    # ALKYNE REACTIONS
    # ==================================================

    "Alkyne Hydrogenation": {
        "type": "reduction",
        "reagent": "H2/Pd",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1]#[C:2]>>[C:1][C:2]"
        )
    },

    "Lindlar": {
        "type": "partial_reduction",
        "reagent": "H2/Lindlar",
        "rxn": AllChem.ReactionFromSmarts(
            "[C:1]#[C:2]>>[C:1]=[C:2]"
        )
    },

    # ==================================================
    # AROMATIC
    # ==================================================

    "Nitration": {
        "type": "EAS",
        "reagent": "HNO3/H2SO4",
        "rxn": AllChem.ReactionFromSmarts(
            "[cH1:1]1[cH1:2][cH1:3][cH1:4][cH1:5][cH1:6]1>>[c:1]1[c:2][c:3]([N+](=O)[O-])[c:4][c:5][c:6]1"
        )
    },

    "Aromatic Bromination": {
        "type": "EAS",
        "reagent": "Br2/FeBr3",
        "rxn": AllChem.ReactionFromSmarts(
            "[cH1:1]1[cH1:2][cH1:3][cH1:4][cH1:5][cH1:6]1>>[c:1]1([Br])[cH1:2][cH1:3][cH1:4][cH1:5][cH1:6]1"
        )
    },

    "Aromatic Chlorination": {
        "type": "EAS",
        "reagent": "Cl2/AlCl3",
        "rxn": AllChem.ReactionFromSmarts(
            "[cH1:1]1[cH1:2][cH1:3][cH1:4][cH1:5][cH1:6]1>>[c:1]1([Cl])[cH1:2][cH1:3][cH1:4][cH1:5][cH1:6]1"
        )
    },

    "Arenediazonium Salt Preparation": {
        "type": "EAS",
        "reagent": "NaNO2, HCl",
        "rxn": AllChem.ReactionFromSmarts(
            "c1ccc(cc1)N>>c1ccccc1[N+]#N"
        )
    },

    "Sandmeyer, H30+": {
        "type": "EAS",
        "reagent": "H3O+",
        "rxn": AllChem.ReactionFromSmarts(
            "c1ccccc1[N+]#N>>c1ccccc1O"
        )
    },

    "Sandmeyer, CuCN": {
        "type": "EAS",
        "reagent": "CuCN",
        "rxn": AllChem.ReactionFromSmarts(
            "c1ccccc1[N+]#N>>c1ccccc1C#N"
        )
    },

    "Sandmeyer, HBF4": {
        "type": "EAS",
        "reagent": "HBF4",
        "rxn": AllChem.ReactionFromSmarts(
            "c1ccccc1[N+]#N>>c1ccccc1F"
        )
    },

    "Sandmeyer, KI": {
        "type": "EAS",
        "reagent": "KI",
        "rxn": AllChem.ReactionFromSmarts(
            "c1ccccc1[N+]#N>>c1ccccc1I"
        )
    },

    "Sandmeyer, H3PO2": {
        "type": "EAS",
        "reagent": "Cl2/AlCl3",
        "rxn": AllChem.ReactionFromSmarts(
            "c1ccccc1[N+]#N>>c1ccccc1"
        )
    }
}

G = nx.DiGraph()

# Canonicalization to ensure consistency

def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True)

# Build the graph

for node in seed_molecules:
    G.add_node(canonicalize(node))
for molecule in seed_molecules:
    reactant = canonicalize(molecule)
    mol = Chem.MolFromSmiles(molecule)
    for reaction_type, reaction in REACTIONS.items():
        try:
            prod = reaction["rxn"].RunReactants((mol, ))
            for products in prod:
                molprod = products[0]
                Chem.SanitizeMol(molprod)
                molecule_smiles = Chem.MolToSmiles(molprod)
                product = canonicalize(molecule_smiles)
                G.add_node(product)
                G.add_edge(reactant, product, reaction = reaction_type, reagent = reaction["reagent"])
        except:
            pass

# Derive all possible problems

candidate_paths = []

for start in G.nodes:
    for target in G.nodes:
        try:
            if start == target:
                continue
            path = nx.shortest_path(G, start, target)

            if 2 <= len(path) <= 6:
                candidate_paths.append((path))
        except:
            pass

st.write(len(candidate_paths))

# Create a new problem

def new_problem():
    st.session_state.problem = random.choice(candidate_paths)
    st.session_state.check = False

if "problem" not in st.session_state:
    new_problem()

st.button("New Problem", on_click=new_problem)

problem = st.session_state.problem
problem_start = problem[0]
problem_end = problem[-1]

cols = st.columns(len(problem) * 2 - 1, vertical_alignment="center")

if st.checkbox("View solution", key = "check"):
    for i, smi in enumerate(problem):
        mol = Chem.MolFromSmiles(smi)

        with cols[2 * i]:
            st.image(Draw.MolToImage(mol), width=200)

        if i < len(problem) - 1:
            next_smi = problem[i + 1]
            reagent_name = G.edges[smi, next_smi]["reagent"]
            def reagent_to_latex(text):
                return re.sub(r'(\d+)', r'_{\1}', text)

            latex = reagent_to_latex(reagent_name)

            with cols[2 * i + 1]:
                st.latex(rf"\begin{{array}}{{c}}\mathrm{{{latex}}}\\ \rightarrow\end{{array}}")


startmol, arrow, endmol = st.columns(3, vertical_alignment="center")

with startmol:
    startdraw = Chem.MolFromSmiles(problem_start)
    imgstart = Draw.MolToImage(startdraw, size=(300,300))
    st.image(imgstart)
with arrow:
    st.latex(rf"\begin{{array}}{{c}}\mathrm{{?}} \\ \rightarrow\end{{array}}")
with endmol:
    enddraw = Chem.MolFromSmiles(problem_end)
    imgend = Draw.MolToImage(enddraw, size=(300,300))
    st.image(imgend)

