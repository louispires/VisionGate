# Simple build and push script for VisionGate Docker image

$ImageName = "valiente/gate-classifier-onnx-intel:latest"
$Dockerfile = "docker/Dockerfile.onnx.intel"

Write-Host "🐳 Building and Pushing VisionGate Docker Image" -ForegroundColor Green
Write-Host "Image: $ImageName" -ForegroundColor Cyan

# Check if ONNX model exists
$onnxModel = "models/gate_mobilenetv3_best.onnx"
if (-not (Test-Path $onnxModel)) {
    Write-Host "❌ ONNX model not found, exporting..." -ForegroundColor Red
    python scripts/export_onnx.py
    if (-not (Test-Path $onnxModel)) {
        Write-Host "❌ Failed to create ONNX model" -ForegroundColor Red
        exit 1
    }
}

# Copy files for Docker build
Write-Host "📋 Preparing files..." -ForegroundColor Yellow
Copy-Item $onnxModel "gate_mobilenetv3.onnx" -Force
Copy-Item "servers/server_mobilenetv3_onnx.py" "." -Force

# Build Docker image
Write-Host "📦 Building Docker image..." -ForegroundColor Yellow
docker build -f $Dockerfile -t $ImageName .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Build successful" -ForegroundColor Green

# Push to Docker Hub
Write-Host "📤 Pushing to Docker Hub..." -ForegroundColor Yellow
docker push $ImageName

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Push failed" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Successfully pushed to Docker Hub!" -ForegroundColor Green
Write-Host "🌐 Available at: https://hub.docker.com/r/valiente/gate-classifier-onnx-intel" -ForegroundColor Cyan

# Cleanup
Write-Host "🧹 Cleaning up..." -ForegroundColor Yellow
Remove-Item "gate_mobilenetv3.onnx" -ErrorAction SilentlyContinue
Remove-Item "server_mobilenetv3_onnx.py" -ErrorAction SilentlyContinue

Write-Host "🎉 Complete!" -ForegroundColor Green
