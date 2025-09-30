# VisionGate API Documentation

## Gate Classification REST API

The VisionGate system provides a RESTful API for classifying gate images as either "open" or "closed".

### Base URL
```
http://localhost:8000
```

### Authentication
No authentication required for this version.

## Endpoints

### 1. Health Check

Check if the service is running and get system information.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model": "MobileNetV3",
  "input_size": [64, 64],
  "classes": ["closed", "open"],
  "providers": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
  "gpu_available": true,
  "memory_usage": "0.64 GB"
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `500 Internal Server Error` - Service has issues

### 2. Service Information

Get detailed information about the service and model.

**Endpoint:** `GET /`

**Response:**
```json
{
  "message": "Gate Classification API - MobileNetV3",
  "model": "MobileNetV3-Large",
  "version": "1.0.0",
  "input_size": [64, 64],
  "classes": ["closed", "open"],
  "architecture": "MobileNetV3",
  "parameters": "4.2M",
  "accuracy": "98.88%"
}
```

### 3. Classify Image

Classify a gate image as open or closed.

**Endpoint:** `POST /classify`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): Image file (JPEG, PNG, GIF, BMP, TIFF)

**Example Request (cURL):**
```bash
curl -X POST "http://localhost:8000/classify" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@gate_image.jpg"
```

**Example Request (PowerShell):**
```powershell
$form = @{
    file = Get-Item "gate_image.jpg"
}
Invoke-RestMethod -Uri "http://localhost:8000/classify" -Method Post -Form $form
```

**Example Request (Python):**
```python
import requests

with open("gate_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/classify", files=files)
    result = response.json()
```

**Successful Response:**
```json
{
  "prediction": "open",
  "confidence": 0.9995,
  "probabilities": {
    "closed": 0.0005,
    "open": 0.9995
  },
  "processing_time": 0.023,
  "image_size": [64, 64],
  "model_used": "MobileNetV3-best"
}
```

**Error Response:**
```json
{
  "detail": "File must be an image"
}
```

**Status Codes:**
- `200 OK` - Classification successful
- `400 Bad Request` - Invalid file or request
- `422 Unprocessable Entity` - Invalid input format
- `500 Internal Server Error` - Processing error

## Image Requirements

### Supported Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tif, .tiff)

### Recommended Specifications
- **Resolution**: Any size (automatically resized to 64x64)
- **Color**: RGB or grayscale (converted to RGB)
- **File Size**: < 10MB recommended
- **Quality**: Higher quality images generally produce better results

### Preprocessing
All uploaded images undergo the following preprocessing:
1. Convert to RGB format
2. Resize to 64x64 pixels (maintains aspect ratio with padding if needed)
3. Normalize using ImageNet statistics:
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

## Error Handling

### Common Error Codes

| Status Code | Error | Description | Solution |
|-------------|-------|-------------|----------|
| 400 | Invalid file type | File is not an image | Upload a valid image file |
| 413 | File too large | Image exceeds size limit | Reduce file size |
| 422 | Missing file | No file provided | Include 'file' parameter |
| 500 | Model error | Internal processing error | Check server logs |

### Error Response Format
```json
{
  "detail": "Error description",
  "error_code": "INVALID_FILE_TYPE",
  "timestamp": "2025-09-30T10:30:00Z"
}
```

## Rate Limiting

Currently no rate limiting is implemented. For production use, consider implementing:
- Request rate limiting
- File size limits
- Concurrent request limits

## Performance

### Typical Response Times
- **GPU (CUDA)**: 20-30ms
- **GPU (Intel ARC)**: 25-35ms  
- **CPU**: 50-100ms

### Throughput
- **Concurrent requests**: Up to 4 simultaneous requests
- **Images per second**: 20-50 depending on hardware

## Integration Examples

### JavaScript (Frontend)
```javascript
async function classifyGate(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const response = await fetch('http://localhost:8000/classify', {
        method: 'POST',
        body: formData
    });
    
    return await response.json();
}
```

### Python (Backend)
```python
import requests
from pathlib import Path

def classify_gate_image(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            'http://localhost:8000/classify', 
            files=files
        )
        return response.json()

result = classify_gate_image('gate.jpg')
print(f"Gate is {result['prediction']} with {result['confidence']:.2%} confidence")
```

### C# (.NET)
```csharp
using System.Text.Json;

public async Task<GateClassificationResult> ClassifyGateAsync(string imagePath)
{
    using var client = new HttpClient();
    using var form = new MultipartFormDataContent();
    using var fileContent = new ByteArrayContent(await File.ReadAllBytesAsync(imagePath));
    
    fileContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg");
    form.Add(fileContent, "file", Path.GetFileName(imagePath));
    
    var response = await client.PostAsync("http://localhost:8000/classify", form);
    var json = await response.Content.ReadAsStringAsync();
    
    return JsonSerializer.Deserialize<GateClassificationResult>(json);
}
```

## Monitoring

### Health Monitoring
- Use `GET /health` for service health checks
- Monitor response times and error rates
- Check GPU memory usage (included in health response)

### Logging
Server logs include:
- Request timestamps
- Processing times
- Error details
- GPU memory usage
- Model performance metrics

### Metrics
Key metrics to monitor:
- Requests per second
- Average response time
- Error rate
- GPU utilization
- Memory usage