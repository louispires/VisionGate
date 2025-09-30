#!/usr/bin/env pwsh
# Deployment script for VisionGate Docker containers

param(
    [string]$Action = "build",
    [string]$Tag = "latest",
    [string]$Registry = "valiente",
    [switch]$Push = $false,
    [switch]$NoCachedBuild = $false
)

$ProjectName = "visiongate-onnx-intel"
$DockerFile = "docker/Dockerfile.onnx.intel"
$ImageTag = "${Registry}/${ProjectName}:${Tag}"

Write-Host "🐳 Gate Classifier Docker Deployment" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green

switch ($Action.ToLower()) {
    "build" {
        Write-Host "📦 Building Docker image: $ImageTag" -ForegroundColor Yellow
        
        # Check if ONNX model exists
        $onnxModel = "models/gate_mobilenetv3_best.onnx"
        if (-not (Test-Path $onnxModel)) {
            Write-Host "❌ ONNX model not found: $onnxModel" -ForegroundColor Red
            Write-Host "🔧 Exporting ONNX model first..." -ForegroundColor Yellow
            python scripts/export_onnx.py
            
            if (-not (Test-Path $onnxModel)) {
                Write-Host "❌ Failed to create ONNX model" -ForegroundColor Red
                exit 1
            }
        }
        
        # Copy model to root for Docker context
        Copy-Item $onnxModel "gate_mobilenetv3.onnx" -Force
        if (Test-Path "models/gate_mobilenetv3_best.onnx.data") {
            Copy-Item "models/gate_mobilenetv3_best.onnx.data" "gate_mobilenetv3.onnx.data" -Force
        }
        
        # Copy server file to root for Docker context
        Copy-Item "servers/server_mobilenetv3_onnx.py" "." -Force
        
        # Build command
        $buildArgs = @(
            "build",
            "-f", $DockerFile,
            "-t", $ImageTag
        )
        
        if ($NoCachedBuild) {
            $buildArgs += "--no-cache"
        }
        
        $buildArgs += "."
        
        Write-Host "Running: docker $($buildArgs -join ' ')" -ForegroundColor Cyan
        & docker @buildArgs
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Build successful: $ImageTag" -ForegroundColor Green
        } else {
            Write-Host "❌ Build failed" -ForegroundColor Red
            exit 1
        }
        
        # Cleanup temporary files
        Remove-Item "gate_mobilenetv3.onnx" -ErrorAction SilentlyContinue
        Remove-Item "gate_mobilenetv3.onnx.data" -ErrorAction SilentlyContinue
        Remove-Item "server_mobilenetv3_onnx.py" -ErrorAction SilentlyContinue
    }
    
    "run" {
        Write-Host "🚀 Running Docker container: $ImageTag" -ForegroundColor Yellow
        
        # Stop existing container
        docker stop gate-classifier-onnx 2>$null
        docker rm gate-classifier-onnx 2>$null
        
        # Run new container
        $runArgs = @(
            "run", "-d",
            "--name", "gate-classifier-onnx",
            "-p", "8000:8000"
        )
        
        # Add GPU support if available
        if (Get-Command "intel_gpu_top" -ErrorAction SilentlyContinue) {
            $runArgs += "--device=/dev/dri"
        }
        
        $runArgs += $ImageTag
        
        Write-Host "Running: docker $($runArgs -join ' ')" -ForegroundColor Cyan
        & docker @runArgs
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Container started successfully" -ForegroundColor Green
            Write-Host "🌐 API available at: http://localhost:8000" -ForegroundColor Cyan
            Write-Host "❤️ Health check: http://localhost:8000/health" -ForegroundColor Cyan
        } else {
            Write-Host "❌ Failed to start container" -ForegroundColor Red
            exit 1
        }
    }
    
    "push" {
        Write-Host "📤 Pushing to registry: $ImageTag" -ForegroundColor Yellow
        
        docker push $ImageTag
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Push successful: $ImageTag" -ForegroundColor Green
        } else {
            Write-Host "❌ Push failed" -ForegroundColor Red
            exit 1
        }
    }
    
    "all" {
        Write-Host "🔄 Building, running, and optionally pushing..." -ForegroundColor Yellow
        
        # Build
        & $PSCommandPath -Action "build" -Tag $Tag -Registry $Registry -NoCachedBuild:$NoCachedBuild
        if ($LASTEXITCODE -ne 0) { exit 1 }
        
        # Run
        & $PSCommandPath -Action "run" -Tag $Tag -Registry $Registry
        if ($LASTEXITCODE -ne 0) { exit 1 }
        
        # Push if requested
        if ($Push) {
            & $PSCommandPath -Action "push" -Tag $Tag -Registry $Registry
        }
        
        Write-Host "🎉 Deployment complete!" -ForegroundColor Green
    }
    
    "test" {
        Write-Host "🧪 Testing deployed container..." -ForegroundColor Yellow
        
        # Wait for container to start
        Start-Sleep -Seconds 5
        
        # Test health endpoint
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 10
            Write-Host "✅ Health check passed: $($response.status)" -ForegroundColor Green
            Write-Host "📊 Model: $($response.model)" -ForegroundColor Cyan
            Write-Host "🔧 Providers: $($response.providers -join ', ')" -ForegroundColor Cyan
        } catch {
            Write-Host "❌ Health check failed: $_" -ForegroundColor Red
            exit 1
        }
        
        # Run comprehensive tests
        Write-Host "🧪 Running comprehensive tests..." -ForegroundColor Yellow
        & ./tests/test_api.ps1
    }
    
    default {
        Write-Host "❌ Unknown action: $Action" -ForegroundColor Red
        Write-Host "Valid actions: build, run, push, all, test" -ForegroundColor Yellow
        exit 1
    }
}

if ($Push -and $Action -eq "build") {
    Write-Host "📤 Pushing to registry..." -ForegroundColor Yellow
    & $PSCommandPath -Action "push" -Tag $Tag -Registry $Registry
}

Write-Host "✨ Done!" -ForegroundColor Green