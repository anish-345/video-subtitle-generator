# Cloudflare Tunnel Configuration Helper for Whisper Subtitle Studio
param (
    [string]$Token = ""
)

$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$TokenFile = Join-Path $PSScriptRoot "tunnel_token.txt"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   CLOUDFLARE TUNNEL SETUP FOR WHISPER STUDIO           " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

if ($Token) {
    Set-Content -Path $TokenFile -Value $Token.Trim()
    Write-Host "`n[✓] Cloudflare Tunnel token saved to tunnel_token.txt!" -ForegroundColor Green
    Write-Host "Now, whenever you run 'startsub', your custom domain tunnel connects automatically." -ForegroundColor White
    exit 0
}

Write-Host "`nChoose an option:" -ForegroundColor Yellow
Write-Host "  [1] Instant Quick Tunnel (Free public https://*.trycloudflare.com URL, zero setup)" -ForegroundColor White
Write-Host "  [2] Configure Permanent Custom Domain (e.g. subtitles.yourdomain.com)" -ForegroundColor White
Write-Host "  [3] Remove saved tunnel token" -ForegroundColor White
Write-Host "  [Q] Quit" -ForegroundColor White

$choice = Read-Host "`nEnter selection (1/2/3/Q)"

if ($choice -eq "1") {
    Write-Host "`nStarting Quick Cloudflare Tunnel to http://localhost:8000..." -ForegroundColor Green
    Write-Host "Look for the https://*.trycloudflare.com URL below:`n" -ForegroundColor Yellow
    & "cloudflared.exe" tunnel --url http://localhost:8000
} elseif ($choice -eq "2") {
    Write-Host "`nTo connect your custom domain:" -ForegroundColor Cyan
    Write-Host "1. Go to Cloudflare Zero Trust (https://one.dash.cloudflare.com)" -ForegroundColor White
    Write-Host "2. Navigate to Networks -> Tunnels -> Add a Tunnel" -ForegroundColor White
    Write-Host "3. Name it 'whisper-studio' and copy the token string" -ForegroundColor White
    Write-Host "4. In the 'Public Hostname' tab, add:" -ForegroundColor White
    Write-Host "     - Subdomain: subtitles" -ForegroundColor White
    Write-Host "     - Domain:    yourdomain.com" -ForegroundColor White
    Write-Host "     - Type:      HTTP" -ForegroundColor White
    Write-Host "     - URL:       localhost:8000" -ForegroundColor White
    Write-Host ""
    $inputToken = Read-Host "Paste your Cloudflare Tunnel token here"
    if ($inputToken) {
        Set-Content -Path $TokenFile -Value $inputToken.Trim()
        Write-Host "`n[✓] Token saved! Whenever you run 'startsub', it connects your custom domain automatically!" -ForegroundColor Green
    }
} elseif ($choice -eq "3") {
    if (Test-Path $TokenFile) {
        Remove-Item $TokenFile -Force
        Write-Host "`n[✓] Token removed." -ForegroundColor Green
    } else {
        Write-Host "`nNo saved token found." -ForegroundColor Gray
    }
}
