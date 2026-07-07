Write-Host ""
Write-Host "========================================="
Write-Host "     ARB-RAG Project Setup Started"
Write-Host "========================================="
Write-Host ""

# -------------------------------
# Create Folder Structure
# -------------------------------

$folders = @(
    "src",
    "src\core",
    "src\retrieval",
    "src\models",
    "src\evaluation",
    "src\utils",
    "data",
    "data\corpus",
    "data\embeddings",
    "data\results",
    "models",
    "experiments",
    "paper"
)

foreach ($folder in $folders) {
    if (!(Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "[OK] Created folder: $folder"
    }
}

# -------------------------------
# Create Files
# -------------------------------

$files = @(
    "README.md",
    "requirements.txt",
    ".gitignore",
    "config.py",
    "run.py",

    "src\core\query_analyzer.py",
    "src\core\adaptive_budget.py",

    "src\retrieval\embedder.py",
    "src\retrieval\retriever.py",
    "src\retrieval\rag_pipeline.py",

    "src\evaluation\evaluator.py",

    "src\utils\utils.py"
)

foreach ($file in $files) {
    if (!(Test-Path $file)) {
        New-Item -ItemType File -Path $file | Out-Null
        Write-Host "[OK] Created file: $file"
    }
}

# -------------------------------
# Create Virtual Environment
# -------------------------------

if (!(Test-Path ".venv")) {
    Write-Host ""
    Write-Host "Creating Virtual Environment..."
    python -m venv .venv
}

Write-Host ""
Write-Host "========================================="
Write-Host "Project Structure Created Successfully!"
Write-Host "========================================="
Write-Host ""

Write-Host "Next Commands:"
Write-Host ""
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host "python -m pip install --upgrade pip"
Write-Host "pip install -r requirements.txt"