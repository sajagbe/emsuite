# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'emsuite'
copyright = '2026, Stephen 0. Ajagbe'
author = 'Stephen 0. Ajagbe'
release = '1.4.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']



extensions = [
    'sphinx.ext.autodoc',      # Pull docstrings from code
    'sphinx.ext.napoleon',     # Support Google/NumPy docstrings
    'sphinx.ext.viewcode',     # Add source code links
    'sphinx_copybutton',   # Add copy buttons to code blocks
    'sphinx_design',        # Add this
]

html_theme_options = {
    "accent_color": "purple",  # Base color (gradient via custom CSS)
    "dark_code": True,
    "github_url": "https://github.com/sajagbe/emsuite",
}

html_css_files = [
    'custom.css',
]