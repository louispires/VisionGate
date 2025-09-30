# Simple curl-based test for Gate Classifier API
$serverUrl = "http://10.10.10.72:8000"
$imagePath = "C:\Working\source\TrainGateModel\TEST\wishbone_gate_train_closed_0929103714.jpg"

Write-Host "Testing Gate Classifier API with curl..." -ForegroundColor Green

# 1. Health Check
Write-Host "`n1. Health Check:" -ForegroundColor Yellow
curl "$serverUrl/health"

# 2. API Information  
Write-Host "`n`n2. API Information:" -ForegroundColor Yellow
curl "$serverUrl/"

# 3. Image Classification
Write-Host "`n`n3. Image Classification:" -ForegroundColor Yellow
if (Test-Path $imagePath) {
    Write-Host "Uploading: $(Split-Path $imagePath -Leaf)" -ForegroundColor Cyan
    curl -X POST "$serverUrl/classify" -F "file=@$imagePath" -H "Accept: application/json"
} else {
    Write-Host "❌ Image file not found: $imagePath" -ForegroundColor Red
}

Write-Host "`n`nDone!" -ForegroundColor Green