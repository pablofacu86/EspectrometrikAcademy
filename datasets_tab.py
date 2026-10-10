# -*- coding: utf-8 -*-
"""'Public datasets' tab: where to find data to practise (and publish) chemometrics."""
import streamlit as st
from i18n import T

# (name, technique, task, description, url, how to cite / license)
DATASETS = [
    # ------------------------------------------------------------------ tables
    ("Wine (178 wines, 13 chemical variables)", "Table", "Classification",
     "The classic chemometrics dataset: Italian wines from 3 cultivars analysed for alcohol, phenols, colour, proline… Perfect for PCA, LDA, PLS-DA and SIMCA.",
     "https://archive.ics.uci.edu/dataset/109/wine",
     "Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J — CC BY 4.0."),
    ("Iris (150 flowers, 4 measurements)", "Table", "Classification",
     "Fisher's iris data: three species, four measurements. The smallest dataset to see classes that separate and classes that overlap.",
     "https://archive.ics.uci.edu/dataset/53/iris",
     "Fisher, R. A. (1936). Iris [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76 — CC BY 4.0."),
    ("Diabetes (442 patients, 10 clinical variables)", "Table", "Regression",
     "Baseline clinical variables (age, BMI, blood pressure, six serum measurements) and disease progression one year later. Distributed with scikit-learn.",
     "https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset",
     "Efron, B., Hastie, T., Johnstone, I. & Tibshirani, R. (2004). Least angle regression. Annals of Statistics, 32(2), 407–499."),
    ("UCI Machine Learning Repository", "Table", "Classification · Regression",
     "Hundreds of curated datasets (many chemical/food/medical), each with its own page, DOI, licence and citation.",
     "https://archive.ics.uci.edu/",
     "Use the 'Cite' box of each dataset page (it gives the DOI)."),
    # ------------------------------------------------------------------ NIR
    ("Eigenvector Research — benchmark datasets", "NIR · IR · process", "Regression · Calibration transfer",
     "Corn NIR measured on three instruments (moisture, oil, protein, starch), the 2002 pharmaceutical-tablet 'shootout', diesel fuel NIR, NIR/IR images and more. A standard source for calibration and standardisation studies.",
     "https://eigenvector.com/resources/data-sets/",
     "Each dataset page gives its acknowledgement/citation (e.g. Cargill for corn, SWRI for diesel, the IDRC 2002 shootout for tablets). Quote it."),
    ("Tecator — NIR of meat (fat, water, protein)", "NIR", "Regression",
     "215 minced-meat samples, 100-channel NIR transmittance (850–1050 nm), fat/water/protein by chemical analysis. Very widely used for benchmarking.",
     "https://lib.stat.cmu.edu/datasets/tecator",
     "Follow the citation request of the StatLib page (Tecator, Borggaard & Thodberg)."),
    ("Gasoline NIR and octane number", "NIR", "Regression",
     "60 gasoline samples, NIR 900–1700 nm, octane number. Included in the R package 'pls'.",
     "https://cran.r-project.org/package=pls",
     "Kalivas, J. H. (1997). Two data sets of near infrared spectra. Chemometrics and Intelligent Laboratory Systems, 37, 255–259."),
    ("Open Soil Spectral Library (OSSL)", "NIR · Vis-NIR · MIR", "Regression",
     "Large soil spectral libraries with laboratory reference values, open data and open tools (explorer, engine, manual).",
     "https://soilspectroscopy.org/",
     "Follow the citation given in the OSSL manual (soilspectroscopy.github.io/ossl-manual)."),
    # ------------------------------------------------------------------ MIR
    ("OliveOil — FT-MIR of extra-virgin olive oils (4 countries)", "FT-MIR", "Classification",
     "120 spectra (570 points) of olive oils from Greece, Italy, Portugal and Spain. A classic authentication problem: PCA, SIMCA and PLS-DA.",
     "https://www.timeseriesclassification.com/description.php?Dataset=OliveOil",
     "Tapp, H. S., Defernez, M. & Kemsley, E. K. (2003). FTIR spectroscopy and multivariate analysis can distinguish the geographic origin of extra virgin olive oils. J. Agric. Food Chem., 51, 6110–6115."),
    ("Coffee — FT-IR of instant coffee (Arabica vs Robusta)", "FT-MIR", "Classification",
     "56 FT-IR spectra (286 points) to distinguish Arabica from Robusta.",
     "https://www.timeseriesclassification.com/description.php?Dataset=Coffee",
     "Briandet, R., Kemsley, E. K. & Wilson, R. H. (1996). Discrimination of Arabica and Robusta in instant coffee by Fourier transform infrared spectroscopy and chemometrics. J. Agric. Food Chem., 44(1), 170–174."),
    ("Olive and vegetable oils — FTIR-ATR and Raman (Zenodo)", "FT-MIR · Raman", "Classification · Regression",
     "Raw spectra (CSV) of olive oil, sunflower oil and blends, to quantify adulteration.",
     "https://zenodo.org/doi/10.5281/zenodo.14651815",
     "Cite the Zenodo record (DOI 10.5281/zenodo.14651815) and the associated paper (LWT – Food Science and Technology, 86, 174–184, 2017)."),
    ("NIST Chemistry WebBook — reference IR and mass spectra", "IR · MS", "Identification",
     "Reference spectra of thousands of compounds (gas-phase IR, mass spectra, quantitative IR database).",
     "https://webbook.nist.gov/chemistry/",
     "Use the Citation Guide of the site (database DOI 10.18434/T4D303)."),
    ("SDBS — Spectral Database for Organic Compounds (AIST, Japan)", "NMR · IR · Raman · MS", "Identification",
     "Free spectra of organic compounds (¹H and ¹³C NMR, IR, Raman, MS). Useful to assign bands and peaks.",
     "https://sdbs.db.aist.go.jp/",
     "Read the terms of use of the site and credit AIST/SDBS."),
    ("USGS Spectral Library", "Vis-NIR-SWIR", "Identification",
     "Laboratory reflectance spectra of minerals, vegetation and man-made materials.",
     "https://www.usgs.gov/labs/spectroscopy-lab/science/spectral-library",
     "See the citation recommended on the library page."),
    # ------------------------------------------------------------------ Raman
    ("RRUFF — Raman, XRD and chemistry of minerals", "Raman", "Identification",
     "Integrated database of well-characterised mineral Raman spectra (many laser wavelengths), freely downloadable.",
     "https://rruff.info/",
     "Check the terms on the site and cite Lafuente, B. et al. (2015), The power of databases: the RRUFF project, in Highlights in Mineralogical Crystallography."),
    ("Bacteria-ID — Raman spectra of pathogenic bacteria", "Raman", "Classification",
     "Thousands of Raman spectra of 30 bacterial and yeast isolates (code on GitHub, data on a linked Dropbox folder).",
     "https://github.com/csho33/bacteria-ID",
     "Ho, C.-S. et al. (2019). Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning. Nature Communications, 10, 4927. Code: MIT licence."),
    ("Open Specy — Raman and FTIR reference spectra (microplastics)", "Raman · FTIR", "Identification",
     "Open-source tool and reference libraries to process, identify and share Raman and FTIR spectra.",
     "https://openspecy.org/",
     "Cowger, W. et al. (2025). Open Specy 1.0. Analytical Chemistry, 97(32), 17345–17356. https://doi.org/10.1021/acs.analchem.5c00962"),
    # ------------------------------------------------------------------ NMR / metabolomics / chromatography
    ("MetaboLights (EMBL-EBI)", "NMR · MS · GC/LC-MS", "Classification · Regression",
     "The reference repository of metabolomics studies, with raw and processed NMR and MS data and rich sample metadata. Search by organism, matrix or technique.",
     "https://www.ebi.ac.uk/metabolights/",
     "Cite the study accession (MTBLS…) and the MetaboLights paper (Yurekten, O. et al., Nucleic Acids Res., 2024)."),
    ("Metabolomics Workbench (NIH)", "NMR · GC-MS · LC-MS", "Classification · Regression",
     "More than 2 000 public studies (NMR, GC-MS, LC-MS) with downloadable data and analysis tools.",
     "https://www.metabolomicsworkbench.org/",
     "Cite the study ID (ST…) and the Workbench (Sud, M. et al., Nucleic Acids Res., 2016)."),
    ("Maize grains — NMR and LC-MS (Zenodo)", "NMR · LC-MS", "Classification",
     "¹H NMR (and CPMG) spectra of maize grain and pellet extracts, plus LC-MS raw files. Large (7.8 GB): download only the NMR folders.",
     "https://doi.org/10.5281/zenodo.1097409",
     "Bernillon, S. et al. (2017). Maize grains and maize grain-based diets dataset. Zenodo. https://doi.org/10.5281/zenodo.1097409 — CC BY-NC 4.0 (non-commercial)."),
    # ------------------------------------------------------------------ general
    ("Zenodo", "Any", "Any",
     "General open repository (CERN). Search for 'NIR spectra', 'Raman dataset', 'chromatogram', 'chemometrics'. Every record has a DOI.",
     "https://zenodo.org/", "Cite the record's DOI and respect its licence."),
    ("Mendeley Data", "Any", "Any",
     "Open repository where many analytical-chemistry papers deposit their spectral datasets.",
     "https://data.mendeley.com/", "Cite the dataset DOI and the paper."),
    ("Figshare", "Any", "Any", "General repository used by many journals for supplementary data.",
     "https://figshare.com/", "Cite the DOI."),
    ("Dryad", "Any", "Any", "Curated repository for data underlying published papers.",
     "https://datadryad.org/", "Cite the DOI."),
    ("OpenML", "Table", "Classification · Regression", "Machine-learning datasets with standard tasks and benchmarks.",
     "https://www.openml.org/", "Cite the dataset and the original authors."),
    ("Kaggle Datasets", "Any", "Any", "Community datasets (quality varies: check the original source and licence).",
     "https://www.kaggle.com/datasets", "Cite the original authors, not only the Kaggle page."),
    ("Google Dataset Search", "Any", "Any", "A search engine for datasets across repositories.",
     "https://datasetsearch.research.google.com/", "Cite the original repository record."),
]

TECNICAS = ["All", "Table", "NIR", "FT-MIR", "Raman", "NMR", "GC/LC", "Any"]


def _coincide(tec, ds):
    if tec == "All":
        return True
    t = ds[1].lower()
    return {"Table": "table", "NIR": "nir", "FT-MIR": "mir", "Raman": "raman", "NMR": "nmr",
            "GC/LC": "ms", "Any": "any"}[tec] in t


def render():
    st.subheader("Where to find public datasets")
    st.markdown(
        "Learning chemometrics needs data. These are the places where researchers share spectra, chromatograms and tables, "
        "from the classic teaching datasets to large modern repositories. **Links reviewed in October 2026**; "
        "websites change, so if one stops working, search the dataset name — and let us know in the Contact tab."
    )

    st.warning(
        "**Please cite the authors.** Public data are the result of someone's work. If you use these datasets in a course, "
        "a poster, a thesis or a paper, reference the original authors and the repository (every entry below gives the citation "
        "to use), and respect the licence: *CC BY* requires attribution; *CC BY-NC* does not allow commercial use.",
        icon="📌")

    c1, c2 = st.columns([2, 3])
    tec = c1.selectbox("Technique", TECNICAS, key="ds_tec",
                       help="Filter the list by the kind of data you are looking for.")
    q = c2.text_input("Search", key="ds_q", placeholder="olive, soil, Raman, metabolomics…",
                      help="Searches in names, descriptions and tasks.")

    filtrados = [d for d in DATASETS if _coincide(tec, d) and
                 (not q or q.lower() in (" ".join(d[:4]) + " " + " ".join(T(x) for x in d[:4])).lower())]
    st.caption(f"{len(filtrados)} of {len(DATASETS)} resources")
    for nombre, tecnica, tarea, desc, url, cita in filtrados:
        with st.container(border=True):
            st.markdown(f"**{nombre}**  \n`{tecnica}` · `{tarea}`")
            st.markdown(desc)
            st.markdown(f"🔗 [{url}]({url})")
            st.caption("📚 How to cite / licence: " + cita)

    with st.expander("🧰 How to bring a public dataset into this app"):
        st.markdown(
            "1. **Download** the data (CSV, Excel, or a format you can open and export to those).\n"
            "2. Arrange it as a table: **one row per sample**. First column = sample ID; then optional **class** and **reference value** "
            "columns; then the variables. For spectra, the *column headers* must be the wavenumbers / wavelengths / ppm / times.\n"
            "3. Save as `.csv` or `.xlsx` and upload it in the left-hand panel (*Spectra / signals* or *Tabular variables*).\n"
            "4. Choose the header row, ID, class and reference-value columns — the app tells you what it detected.\n"
            "5. **Write down the source and the citation** in your notes; the final report includes a field for it."
        )
        st.code(
            "# Example: classic datasets from scikit-learn, saved as Excel to load in the app\n"
            "from sklearn.datasets import load_wine\n"
            "d = load_wine(as_frame=True)\n"
            "df = d.frame.copy()\n"
            "df.insert(0, 'cultivar', [d.target_names[i] for i in df.pop('target')])\n"
            "df.insert(0, 'id', [f'W{i+1}' for i in range(len(df))])\n"
            "df.to_excel('wine.xlsx', index=False)\n",
            language="python")

    with st.expander("🔎 Tips to find (and judge) other datasets"):
        st.markdown(
            "- **Search terms**: *\"NIR\" + matrix + dataset*, *\"Raman\" + analyte + \"data availability\"*, *chemometrics dataset zenodo*. "
            "Most papers have a 'Data availability' section.\n"
            "- **Check the metadata**: how were reference values measured? how many replicates? which instrument? A dataset without that "
            "information is hard to model honestly.\n"
            "- **Beware of replicates**: if the same sample was measured several times, keep the replicates together when you validate "
            "(otherwise cross-validation looks better than reality).\n"
            "- **Test on data from another day or instrument** whenever possible — that is the real test of a model.\n"
            "- **Share your own data**: depositing your spectra in Zenodo or MetaboLights makes you citable and helps the community."
        )
