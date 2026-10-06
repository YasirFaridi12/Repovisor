# RepoVisor - start here (no experience needed)

## Step 1: Install Python (one time)
Download Python 3.10 or newer from https://www.python.org/downloads/ and install it.
**On Windows, tick "Add Python to PATH" on the first installer screen.**
(Only needed for GitHub links: install Git from https://git-scm.com/downloads. Local folders work without it.)

## Step 2: Start the app
- Windows: unzip the folder, then double-click **run.bat**
- Mac/Linux: open a terminal in the folder and run `./run.sh`

The first run installs everything (2-3 minutes) and asks for a **free Gemini API key**.
Get it in 2 minutes: open https://aistudio.google.com/apikey, sign in with Google, click "Create API key", copy it.
You can press Enter to skip: the app still works and shows the matching code, but won't write AI answers.
(To add or change the key later, open the file `.env` and write: GEMINI_API_KEY=your-key)
Your browser opens at http://localhost:8000.

## Step 3: Use it
1. Paste a public GitHub link, for example https://github.com/pallets/itsdangerous, and click **Index repository**.
2. The graph shows how files depend on each other. Drag and zoom it.
3. Ask a question, or click one of the suggested ones.
4. Click **Run quality check** and note the numbers for your report (see below).

## For your report and viva
- Retrieval Quality = the "quality check" result (right file in the top 5, MRR, speed).
- Grounding = every answer lists its source files; invented citations are flagged in orange.
- Security = read-only tools, token never stored or put in prompts, path traversal blocked.
- Be ready to say: JS/TS parsing is regex-based (Python uses a real AST), and search uses TF-IDF
  (upgrade: set EMBED_BACKEND=neural and `pip install sentence-transformers`).

## Troubleshooting
- "python is not recognized": reinstall Python and tick "Add Python to PATH".
- "Indexing failed ... git": install Git, or use a local folder path instead of a URL.
- Port busy: close other copies of the app and run again.
