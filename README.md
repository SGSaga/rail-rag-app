# Rail Safety RAG App

A personal learning project: a retrieval-augmented generation (RAG) app that takes a rail-safety incident report, finds similar past incidents, and generates a summary with a severity and root-cause label.

**All data is synthetic.** It does not use any real records. This is a portfolio and learning project.

---

## What you need
- Python
- PyCharm Community
- An OpenAI API key

---

## Setup (do this once)

Follow in order. Everything happens inside PyCharm.

### 1. Open the project
File → Open → select this `railrag` folder.

### 2. Create a virtual environment
A virtual environment is a clean, isolated box for this project's packages so they don't clash with anything else.

- PyCharm usually offers this automatically when you open a project with a `requirements.txt`. If a yellow bar appears at the top saying it found requirements, click **Create a virtual environment** / **Install requirements**. Done, skip to step 4.
- If not: File → Settings → Project → Python Interpreter → click **Add Interpreter** → **Add Local Interpreter** → **Virtualenv Environment** → **New** → OK.

### 3. Install the packages
Open the Terminal tab at the bottom of PyCharm (View → Tool Windows → Terminal). You should see `(.venv)` at the start of the line. Then run:

```
pip install -r requirements.txt
```

This downloads the libraries. First run also downloads the small embedding model (~90MB) the first time it's used, not now.

### 4. Add your API key
1. In the project, find the file `.env.example`.
2. Right-click it → Copy, then Paste, and rename the copy to exactly `.env` (nothing before the dot).
3. Open `.env`, paste your key after `OPENAI_API_KEY=` with no quotes and no spaces. Save.

Your key stays on your machine. `.gitignore` already stops `.env` going to GitHub.

### 5. Set a spending cap (2 minutes, peace of mind)
Log in at platform.openai.com → Settings → Limits → set a low monthly budget (e.g. $5). This project will cost pennies, but a cap means no surprises.

### 6. Run the check
In the PyCharm Terminal:

```
python check_setup.py
```

You want four `OK` lines and "All good". If any line says FAIL, send me the exact message and I'll fix it.

---

## Project structure
```
railrag/
  data/
    generate_data.py    # makes the synthetic incidents (already run for you)
    incidents.json      # 80 fake incident reports
  rag/
    config.py           # settings + key loading
  check_setup.py        # stage 0: confirm everything works
  requirements.txt      # the packages to install
  .env.example          # template for your key
  .env                  # your key (you create this, never shared)
  .gitignore            # keeps .env and clutter off GitHub
```

We build the rest (ingestion, retrieval, generation, evaluation) in stages after the check passes.

---

## Windows / Python 3.14 note
Python 3.14 is very new. If `pip install -r requirements.txt` fails on one of the packages (a common symptom on brand-new Python versions), don't fight it. Send me the error. The usual fix is installing a slightly different version or, if needed, using Python 3.12 for this project, which I'll walk you through.
