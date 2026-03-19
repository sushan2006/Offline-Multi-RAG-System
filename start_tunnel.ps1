# RAG Assistant: One-Click Cloud Deployment
# This script starts the backend and a public Cloudflare tunnel.

Write-Host "Starting RAG Assistant Cloud Deployment..." -ForegroundColor Cyan

# 1. Start FastAPI Backend in a new window
Write-Host "Launching FastAPI Backend on port 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; python main.py"

# 2. Check for cloudflared
$cfExe = "cloudflared"
if (Test-Path ".\cloudflared.exe") {
    $cfExe = ".\cloudflared.exe"
    Write-Host "Found local cloudflared.exe" -ForegroundColor Gray
}

# Start Cloudflare Quick Tunnel
Write-Host "Creating Secure Public Tunnel..." -ForegroundColor Yellow
Write-Host "Note: If you do not have 'cloudflared' installed, download it from https://github.com/cloudflare/cloudflared/releases" -ForegroundColor Gray

# Use a temporary file to capture the tunnel URL
$tmpFile = "$env:TEMP\tunnel_out.txt"
Remove-Item $tmpFile -ErrorAction SilentlyContinue

# Start cloudflared and wait for it to generate a URL
# Redirect both stdout and stderr since cloudflared logs can vary
$errFile = "$env:TEMP\tunnel_err.txt"
Remove-Item $errFile -ErrorAction SilentlyContinue
$process = Start-Process $cfExe -ArgumentList "tunnel", "--url", "http://localhost:8000" -NoNewWindow -PassThru -RedirectStandardError $errFile -RedirectStandardOutput $tmpFile

Write-Host "Waiting for public URL..." -ForegroundColor Gray
$tunnelUrl = ""
$retry = 0
while ($retry -lt 25 -and -not $tunnelUrl) {
    Start-Sleep -Seconds 1
    # Check both stdout and stderr files
    foreach ($file in $tmpFile, $errFile) {
        if (Test-Path $file) {
            $content = Get-Content $file -Raw
            if ($content -match "https://[a-zA-Z0-9-]+\.trycloudflare\.com") {
                $tunnelUrl = $matches[0]
                break
            }
        }
    }
    $retry++
}

if ($tunnelUrl) {
    Write-Host "`nSUCCESS! Your RAG Assistant is now LIVE at:" -ForegroundColor Green
    Write-Host "$tunnelUrl" -ForegroundColor White -BackgroundColor Blue
    Write-Host "`nUpdating frontend/config.js..." -ForegroundColor Gray
    
    # Update config.js with the new URL
    $configPath = "frontend/config.js"
$configContent = @"
// RAG Assistant Configuration
// This file is used to store global settings for the frontend.
const API_CONFIG = {
    BASE_URL: "$tunnelUrl"
};
window.API_CONFIG = API_CONFIG;
"@
    Set-Content -Path $configPath -Value $configContent
    
    Write-Host "config.js updated. You can now share the URL above!" -ForegroundColor Green
    Write-Host "Press Ctrl+C in the other window to stop the server." -ForegroundColor Gray
} else {
    Write-Host "Failed to detect Cloudflare URL. Please ensure 'cloudflared' is installed and your internet is active." -ForegroundColor Red
    Write-Host "Check details in: $tmpFile" -ForegroundColor Gray
}
