# Quick PowerShell curl commands for Gate Classifier API

# 1. Health Check
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | ConvertTo-Json -Depth 3

# 2. API Information
Invoke-RestMethod -Uri "http://localhost:8000/" -Method Get | ConvertTo-Json -Depth 3

# 3. Classify an image (replace 'path\to\your\image.jpg' with actual image path)
$imagePath = "dataset\val\open\wishbone_gate_train_open_0928231401.jpg"
if (Test-Path $imagePath) {
    # Method 1: Using Add-Type and proper multipart handling
    try {
        Add-Type -AssemblyName System.Net.Http
        
        $httpClientHandler = New-Object System.Net.Http.HttpClientHandler
        $httpClient = New-Object System.Net.Http.HttpClient($httpClientHandler)
        
        $multipartContent = New-Object System.Net.Http.MultipartFormDataContent
        $fileStream = [System.IO.File]::OpenRead($imagePath)
        $fileName = [System.IO.Path]::GetFileName($imagePath)
        $streamContent = New-Object System.Net.Http.StreamContent($fileStream)
        $streamContent.Headers.ContentType = New-Object System.Net.Http.Headers.MediaTypeHeaderValue("image/jpeg")
        
        $multipartContent.Add($streamContent, "file", $fileName)
        
        $response = $httpClient.PostAsync("http://localhost:8000/classify", $multipartContent).Result
        $responseContent = $response.Content.ReadAsStringAsync().Result
        
        $fileStream.Close()
        $httpClient.Dispose()
        
        $responseContent | ConvertFrom-Json | ConvertTo-Json -Depth 3
        
    } catch {
        Write-Host "❌ Method 1 failed: $_" -ForegroundColor Red
        
        # Method 2: Fallback using curl if available
        Write-Host "Trying with curl..." -ForegroundColor Yellow
        try {
            $curlResult = curl -X POST "http://10.10.10.72:8000/classify" -F "file=@$imagePath" -H "Accept: application/json" 2>$null
            $curlResult | ConvertFrom-Json | ConvertTo-Json -Depth 3
        } catch {
            Write-Host "❌ Curl also failed. Check if image file is valid and server is running." -ForegroundColor Red
        }
    }
} else {
    Write-Host "❌ Image file not found: $imagePath" -ForegroundColor Red
}