#!/bin/bash

# Railway Deployment Helper Script
# This script helps with local testing and Railway deployment

set -e

echo "🚀 FastAPI Railway Deployment Helper"
echo "===================================="

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker first."
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to build the Docker image
build_image() {
    echo "🔨 Building Docker image..."
    docker build -t safaricom-fastapi:latest .
    echo "✅ Docker image built successfully"
}

# Function to run container locally
run_local() {
    echo "🚀 Running container locally on port 8000..."
    docker run -d \
        --name safaricom-fastapi-local \
        -p 8000:8000 \
        -e PORT=8000 \
        -e USE_LIGHTWEIGHT_MODEL=true \
        -e MAX_MEMORY_MB=2048 \
        safaricom-fastapi:latest
    
    echo "✅ Container started successfully"
    echo "🌐 API available at: http://localhost:8000"
    echo "🏥 Health check: http://localhost:8000/health"
    echo "📊 API docs: http://localhost:8000/docs"
}

# Function to stop local container
stop_local() {
    echo "🛑 Stopping local container..."
    docker stop safaricom-fastapi-local 2>/dev/null || true
    docker rm safaricom-fastapi-local 2>/dev/null || true
    echo "✅ Local container stopped and removed"
}

# Function to test the API
test_api() {
    echo "🧪 Testing API endpoints..."
    
    # Check if Python is available for testing
    if command -v python3 &> /dev/null; then
        python3 test_api.py
    else
        echo "⚠️ Python3 not found. Running basic curl tests..."
        
        # Wait for container to be ready
        sleep 5
        
        # Test health endpoint
        echo "Testing health endpoint..."
        curl -f http://localhost:8000/health || {
            echo "❌ Health check failed"
            return 1
        }
        
        # Test prediction endpoint
        echo "Testing prediction endpoint..."
        curl -X POST "http://localhost:8000/predict" \
             -H "Content-Type: application/json" \
             -d '{"text": "This is a test tweet about Safaricom"}' || {
            echo "❌ Prediction test failed"
            return 1
        }
        
        echo "✅ Basic API tests passed"
    fi
}

# Function to clean up Docker resources
cleanup() {
    echo "🧹 Cleaning up Docker resources..."
    docker system prune -f
    echo "✅ Cleanup completed"
}

# Function to prepare for Railway deployment
prepare_railway() {
    echo "🚂 Preparing for Railway deployment..."
    
    # Check if railway.toml exists
    if [ ! -f "railway.toml" ]; then
        echo "❌ railway.toml not found. Creating one..."
        cp railway.toml.example railway.toml 2>/dev/null || echo "⚠️ Please ensure railway.toml is configured"
    fi
    
    # Check if .dockerignore exists
    if [ ! -f ".dockerignore" ]; then
        echo "❌ .dockerignore not found. Please ensure it exists."
        exit 1
    fi
    
    echo "✅ Railway deployment files are ready"
    echo "📋 Next steps:"
    echo "   1. Install Railway CLI: npm install -g @railway/cli"
    echo "   2. Login to Railway: railway login"
    echo "   3. Create project: railway init"
    echo "   4. Deploy: railway up"
}

# Function to show usage
usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build      Build the Docker image"
    echo "  run        Run the container locally"
    echo "  stop       Stop the local container"
    echo "  test       Test the API endpoints"
    echo "  restart    Stop and start the container"
    echo "  cleanup    Clean up Docker resources"
    echo "  railway    Prepare for Railway deployment"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run"
    echo "  $0 test"
}

# Main script logic
case "${1:-help}" in
    "build")
        check_docker
        build_image
        ;;
    "run")
        check_docker
        stop_local
        build_image
        run_local
        ;;
    "stop")
        stop_local
        ;;
    "test")
        test_api
        ;;
    "restart")
        check_docker
        stop_local
        build_image
        run_local
        test_api
        ;;
    "cleanup")
        cleanup
        ;;
    "railway")
        prepare_railway
        ;;
    "help"|*)
        usage
        ;;
esac
