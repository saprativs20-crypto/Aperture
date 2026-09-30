$root = $PSScriptRoot
try {
    $listener = New-Object System.Net.HttpListener
    $listener.Prefixes.Add('http://localhost:8792/')
    $listener.Start()
} catch {
    Write-Host ""
    Write-Host "===== SERVER FAILED TO START =====" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Yellow
    Write-Host "==================================="
    Write-Host ""
    Read-Host "Press Enter to close this window"
    exit 1
}
Write-Host "Server is running at http://localhost:8792/" -ForegroundColor Green
Write-Host "Leave this window open while using Aperture. Press Ctrl+C to stop."
$mime = @{
  '.html'='text/html'; '.js'='text/javascript'; '.css'='text/css';
  '.json'='application/json'; '.png'='image/png'; '.jpg'='image/jpeg'; '.svg'='image/svg+xml'
}
while ($listener.IsListening) {
    $ctx = $listener.GetContext()
    $path = $ctx.Request.Url.LocalPath.TrimStart('/')
    if ([string]::IsNullOrEmpty($path)) { $path = 'index.html' }
    $full = Join-Path $root $path
    if (Test-Path $full -PathType Leaf) {
        $ext = [System.IO.Path]::GetExtension($full)
        $ct = $mime[$ext]
        if (-not $ct) { $ct = 'application/octet-stream' }
        $bytes = [System.IO.File]::ReadAllBytes($full)
        $ctx.Response.ContentType = $ct
        $ctx.Response.ContentLength64 = $bytes.Length
        $ctx.Response.Headers.Add('Service-Worker-Allowed', '/')
        $ctx.Response.KeepAlive = $false
        $ctx.Response.OutputStream.Write($bytes, 0, $bytes.Length)
    } else {
        $ctx.Response.StatusCode = 404
    }
    $ctx.Response.OutputStream.Close()
}
