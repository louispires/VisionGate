# PowerShell script to test Gate Classifier API
# Make sure the container is running first:
# docker run -d --name gate-classifier-onnx -p 8000:8000 valiente/gate-classifier-onnx-intel:latest

$baseUrl = "http://10.10.10.72:8000"

Write-Host "Testing Gate Classifier API..." -ForegroundColor Green

# Test 1: Health check
Write-Host "`n1. Health Check:" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "✅ Health Status: $($response.status)" -ForegroundColor Green
    Write-Host "   Providers: $($response.providers -join ', ')" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Health check failed: $_" -ForegroundColor Red
    exit 1
}

# Test 2: API Info
Write-Host "`n2. API Information:" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/" -Method Get
    Write-Host "✅ Service: $($response.message)" -ForegroundColor Green
    Write-Host "   Model: $($response.model)" -ForegroundColor Cyan
    Write-Host "   Input Size: $($response.input_size)" -ForegroundColor Cyan
    Write-Host "   Classes: $($response.classes -join ', ')" -ForegroundColor Cyan
} catch {
    Write-Host "❌ API info failed: $_" -ForegroundColor Red
}

# Test 3: Image classification (you'll need to provide an image)
Write-Host "`n3. Image Classification:" -ForegroundColor Yellow
$imagePath = Read-Host "Enter path to test image (or press Enter to skip)"

if ($imagePath -and (Test-Path $imagePath)) {
    try {
        Write-Host "Uploading image: $(Split-Path $imagePath -Leaf)" -ForegroundColor Cyan
        
        # Method 1: Try using curl (most reliable)
        $curlAvailable = Get-Command curl -ErrorAction SilentlyContinue
        if ($curlAvailable) {
            Write-Host "Using curl for upload..." -ForegroundColor Gray
            $curlResult = curl -s -X POST "$baseUrl/classify" -F "file=@$imagePath" -H "Accept: application/json"
            $response = $curlResult | ConvertFrom-Json
        } else {
            # Method 2: Use .NET HttpClient (more reliable than manual boundary)
            Write-Host "Using .NET HttpClient..." -ForegroundColor Gray
            Add-Type -AssemblyName System.Net.Http
            
            $httpClient = New-Object System.Net.Http.HttpClient
            $multipartContent = New-Object System.Net.Http.MultipartFormDataContent
            
            $fileStream = [System.IO.File]::OpenRead($imagePath)
            $fileName = [System.IO.Path]::GetFileName($imagePath)
            $streamContent = New-Object System.Net.Http.StreamContent($fileStream)
            
            # Set proper content type based on file extension
            $extension = [System.IO.Path]::GetExtension($imagePath).ToLower()
            $contentType = switch ($extension) {
                ".jpg" { "image/jpeg" }
                ".jpeg" { "image/jpeg" }
                ".png" { "image/png" }
                ".gif" { "image/gif" }
                ".bmp" { "image/bmp" }
                default { "image/jpeg" }
            }
            $streamContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse($contentType)
            
            $multipartContent.Add($streamContent, "file", $fileName)
            
            $httpResponse = $httpClient.PostAsync("$baseUrl/classify", $multipartContent).Result
            $responseContent = $httpResponse.Content.ReadAsStringAsync().Result
            
            $fileStream.Close()
            $httpClient.Dispose()
            
            if ($httpResponse.IsSuccessStatusCode) {
                $response = $responseContent | ConvertFrom-Json
            } else {
                throw "HTTP $($httpResponse.StatusCode): $responseContent"
            }
        }
        
        Write-Host "✅ Classification Result:" -ForegroundColor Green
        Write-Host "   Status: $($response.status)" -ForegroundColor $(if($response.status -eq "open") {"Red"} else {"Green"})
        Write-Host "   Confidence: $([math]::Round($response.confidence * 100, 2))%" -ForegroundColor Cyan
        Write-Host "   Probabilities:" -ForegroundColor Cyan
        Write-Host "     Closed: $([math]::Round($response.probabilities.closed * 100, 2))%" -ForegroundColor Green
        Write-Host "     Open: $([math]::Round($response.probabilities.open * 100, 2))%" -ForegroundColor Red
        
        # Handle different response field names (latency_ms vs inference_time_ms)
        $latency = if ($response.latency_ms) { $response.latency_ms } elseif ($response.inference_time_ms) { $response.inference_time_ms } else { "N/A" }
        Write-Host "   Inference Time: $latency ms" -ForegroundColor Cyan
        Write-Host "   Providers: $($response.providers -join ', ')" -ForegroundColor Cyan
        
    } catch {
        Write-Host "❌ Image classification failed:" -ForegroundColor Red
        Write-Host "$_" -ForegroundColor Red
        Write-Host "Error details: $($_.Exception.Message)" -ForegroundColor Red
        
        # Fallback suggestion
        Write-Host "`n💡 Try manual curl command:" -ForegroundColor Yellow
        Write-Host "curl -X POST `"$baseUrl/classify`" -F `"file=@$imagePath`"" -ForegroundColor Gray
    }
} else {
    Write-Host "⏭️  Skipping image classification test" -ForegroundColor Gray
}

Write-Host "`n🏁 Testing complete!" -ForegroundColor Green

# Additional curl examples for reference
Write-Host "`n📋 Manual curl commands for reference:" -ForegroundColor Blue
Write-Host @"
# Health check:
curl http://10.10.10.72:8000/health

# API info:
curl http://10.10.10.72:8000/

# Image classification:
curl -X POST http://10.10.10.72:8000/classify -F "file=@your_image.jpg"
"@ -ForegroundColor Gray