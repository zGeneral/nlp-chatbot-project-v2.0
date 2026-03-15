# upload_artifacts_to_drive.ps1
#
# Zips the 5 artifact files required by train.py and uploads them to
# Google Drive so Colab can find them at:
#   MyDrive/nlp-chatbot-v2/artifacts/
#
# Required files:
#   stage5_spm.model              - SentencePiece BPE tokeniser
#   stage6_train_ids.jsonl        - ~1.1M training pairs (BPE token IDs)
#   stage6_val_ids.jsonl          - ~47K validation pairs
#   stage6_test_ids.jsonl         - ~47K test pairs
#   stage8_embedding_matrix.npy   - 16000x300 FastText embedding matrix
#
# Upload method (tried in order):
#   1. Google Drive for Desktop   - copies directly into your synced Drive folder
#   2. rclone                     - if installed (rclone.org)
#   3. Manual fallback            - saves a zip to Desktop with upload instructions
#
# Usage:
#   .\upload_artifacts_to_drive.ps1
#
# Override the artifacts source folder if needed:
#   .\upload_artifacts_to_drive.ps1 -ArtifactsDir "D:\my-project\artifacts"
# ---------------------------------------------------------------------------

param(
    [string]$ArtifactsDir = "",          # default: auto-detect from script location
    [string]$DriveDestFolder = "nlp-chatbot-v2\artifacts",   # subfolder inside My Drive
    [string]$RcloneRemote = "gdrive",    # rclone remote name (if using rclone)
    [switch]$ZipOnly                     # create zip but do not upload (manual upload)
)

$ErrorActionPreference = "Stop"

# ── Banner ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  NLP Chatbot — Upload artifacts to Google Drive" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ── Resolve artifacts source directory ───────────────────────────────────────
if ($ArtifactsDir -eq "") {
    # Auto-detect: look relative to this script's location
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $ArtifactsDir = Join-Path $ScriptDir "artifacts"
}

if (-not (Test-Path $ArtifactsDir)) {
    Write-Host ""
    Write-Host "ERROR: Artifacts directory not found:" -ForegroundColor Red
    Write-Host "  $ArtifactsDir" -ForegroundColor Red
    Write-Host ""
    Write-Host "Specify the correct path with:" -ForegroundColor Yellow
    Write-Host '  .\upload_artifacts_to_drive.ps1 -ArtifactsDir "C:\path\to\artifacts"' -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "  Source : $ArtifactsDir"

# ── Required files ────────────────────────────────────────────────────────────
$RequiredFiles = @(
    "stage5_spm.model",
    "stage6_train_ids.jsonl",
    "stage6_val_ids.jsonl",
    "stage6_test_ids.jsonl",
    "stage8_embedding_matrix.npy"
)

Write-Host ""
Write-Host "  Checking required files..." -ForegroundColor Yellow
$Missing = @()
$Found   = @()
foreach ($f in $RequiredFiles) {
    $path = Join-Path $ArtifactsDir $f
    if (Test-Path $path) {
        $size = (Get-Item $path).Length / 1MB
        Write-Host ("  [OK] {0,-40} {1,8:F1} MB" -f $f, $size) -ForegroundColor Green
        $Found += $path
    } else {
        Write-Host "  [MISSING] $f" -ForegroundColor Red
        $Missing += $f
    }
}

if ($Missing.Count -gt 0) {
    Write-Host ""
    Write-Host "ERROR: $($Missing.Count) required file(s) missing. Run phase1.py first." -ForegroundColor Red
    exit 1
}

# ── Create zip ────────────────────────────────────────────────────────────────
$ZipName = "nlp_artifacts_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
$ZipPath = Join-Path $env:TEMP $ZipName

Write-Host ""
Write-Host "  Creating zip: $ZipPath" -ForegroundColor Yellow

# Use a temp staging folder so the zip has a flat artifacts/ structure
$StageDir = Join-Path $env:TEMP "nlp_artifacts_stage"
if (Test-Path $StageDir) { Remove-Item $StageDir -Recurse -Force }
New-Item -ItemType Directory -Path $StageDir | Out-Null

foreach ($path in $Found) {
    Copy-Item $path $StageDir
}

Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $ZipPath -Force
Remove-Item $StageDir -Recurse -Force

$ZipSize = (Get-Item $ZipPath).Length / 1MB
Write-Host ("  Zip created: {0:F1} MB" -f $ZipSize) -ForegroundColor Green

if ($ZipOnly) {
    $DesktopZip = Join-Path ([Environment]::GetFolderPath("Desktop")) $ZipName
    Copy-Item $ZipPath $DesktopZip
    Write-Host ""
    Write-Host "  -ZipOnly flag set. Zip saved to Desktop:" -ForegroundColor Cyan
    Write-Host "  $DesktopZip"
    Write-Host ""
    Write-Host "  Upload it manually to Google Drive at:"
    Write-Host "  MyDrive/$($DriveDestFolder -replace '\\','/')" -ForegroundColor Yellow
    Write-Host "  Then unzip it there." -ForegroundColor Yellow
    exit 0
}

# ── Upload — Method 1: Google Drive for Desktop ───────────────────────────────
Write-Host ""
Write-Host "  Detecting Google Drive for Desktop..." -ForegroundColor Yellow

# Common Drive for Desktop mount points
$DriveCandidates = @(
    "$env:USERPROFILE\Google Drive\My Drive",
    "$env:USERPROFILE\Google Drive",
    "G:\My Drive",
    "G:\",
    "H:\My Drive",
    "H:\"
)

# Also search all drive letters
foreach ($letter in 70..90) {
    $dl = [char]$letter + ":\"
    if (Test-Path $dl) {
        $id = Join-Path $dl ".shortcut-targets-by-id"
        if (Test-Path $id) {
            $DriveCandidates += "$dl\My Drive"
            $DriveCandidates += $dl
        }
    }
}

$DriveRoot = $null
foreach ($candidate in $DriveCandidates) {
    if (Test-Path $candidate) {
        $DriveRoot = $candidate
        break
    }
}

if ($DriveRoot) {
    $DestDir = Join-Path $DriveRoot $DriveDestFolder
    Write-Host "  Drive for Desktop found at: $DriveRoot" -ForegroundColor Green
    Write-Host "  Copying files to: $DestDir" -ForegroundColor Yellow

    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null

    foreach ($path in $Found) {
        $fname = Split-Path -Leaf $path
        $dest  = Join-Path $DestDir $fname
        Write-Host "    Copying $fname..."
        Copy-Item $path $dest -Force
    }

    Write-Host ""
    Write-Host "  All files copied to Google Drive for Desktop." -ForegroundColor Green
    Write-Host "  Drive will sync them automatically."
    Write-Host ""
    Write-Host "  Destination on Drive: MyDrive/$($DriveDestFolder -replace '\\','/')" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  On Colab, they will be at:"  -ForegroundColor Cyan
    Write-Host "  /content/drive/MyDrive/$($DriveDestFolder -replace '\\','/')" -ForegroundColor Cyan
    Remove-Item $ZipPath -Force
    exit 0
}

# ── Upload — Method 2: rclone ─────────────────────────────────────────────────
Write-Host "  Google Drive for Desktop not found. Trying rclone..." -ForegroundColor Yellow

$rclone = Get-Command rclone -ErrorAction SilentlyContinue
if ($rclone) {
    $RcloneDest = "$($RcloneRemote):$($DriveDestFolder -replace '\\','/')"
    Write-Host "  Uploading via rclone to: $RcloneDest" -ForegroundColor Yellow

    foreach ($path in $Found) {
        $fname = Split-Path -Leaf $path
        Write-Host "    Uploading $fname..."
        rclone copy $path "$RcloneDest" --progress
    }

    Write-Host ""
    Write-Host "  Upload complete via rclone." -ForegroundColor Green
    Write-Host "  Remote: $RcloneDest" -ForegroundColor Cyan
    Remove-Item $ZipPath -Force
    exit 0
}

# ── Fallback: save zip to Desktop with instructions ───────────────────────────
Write-Host ""
Write-Host "  Neither Google Drive for Desktop nor rclone found." -ForegroundColor Yellow
Write-Host "  Saving zip to Desktop for manual upload..." -ForegroundColor Yellow

$DesktopZip = Join-Path ([Environment]::GetFolderPath("Desktop")) $ZipName
Copy-Item $ZipPath $DesktopZip
Remove-Item $ZipPath -Force

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Zip saved to your Desktop:" -ForegroundColor Cyan
Write-Host "  $DesktopZip" -ForegroundColor White
Write-Host ""
Write-Host "  Manual upload steps:" -ForegroundColor Yellow
Write-Host "  1. Go to https://drive.google.com"
Write-Host "  2. Navigate to: MyDrive/nlp-chatbot-v2/artifacts/"
Write-Host "     (create the folder if it doesn't exist)"
Write-Host "  3. Upload the zip file and extract it there"
Write-Host "     OR upload the 5 files individually from the zip"
Write-Host ""
Write-Host "  Files inside the zip:"
foreach ($f in $RequiredFiles) {
    Write-Host "    - $f"
}
Write-Host "============================================================" -ForegroundColor Cyan
