# FastAPI Railway Deployment Guide

## Quick Start

1. **Build and test locally:**
   ```bash
   ./deploy.sh build
   ./deploy.sh run
   ./deploy.sh test
   ```

2. **Deploy to Railway:**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login to Railway
   railway login
   
   # Initialize project
   railway init
   
   # Deploy
   railway up
   ```

## Project Structure

```
FastAPI/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Railway-optimized Docker configuration
├── railway.toml        # Railway deployment configuration
├── docker-compose.yml  # Local development with Docker
├── .dockerignore       # Files to exclude from Docker build
├── .env.example        # Environment variables template
├── deploy.sh           # Deployment helper script
└── README.md           # This file
```

## Environment Variables

Set these in Railway dashboard or locally in `.env`:

- `USE_LIGHTWEIGHT_MODEL`: Use lightweight model configuration (default: true)
- `MAX_MEMORY_MB`: Maximum memory allocation in MB (default: 1024)
- `ENABLE_MODEL_QUANTIZATION`: Enable model quantization (default: true)
- `MODEL_CACHE_SIZE`: Model cache size (default: 1)
- `HF_TOKEN`: Hugging Face token (optional, for private models)

## API Endpoints

- `GET /` - Root endpoint with health information
- `GET /health` - Simple health check for Railway
- `GET /health/detailed` - Detailed health check with metrics
- `POST /predict` - Scikit-learn model prediction
- `POST /predict/transformer` - Transformer model prediction
- `POST /predict/batch` - Batch predictions (scikit-learn)
- `POST /predict/transformer/batch` - Batch predictions (transformer)
- `GET /docs` - Interactive API documentation

## Railway-Specific Optimizations

1. **Dynamic Port Binding**: Uses Railway's `$PORT` environment variable
2. **Memory Optimization**: Configured for Railway's memory limits
3. **Health Checks**: Simple endpoint for Railway health monitoring
4. **Lightweight Mode**: Optimized for Railway's container constraints
5. **Non-root User**: Security best practice for Railway deployment

## Local Development

### Using Docker Compose
```bash
docker-compose up --build
```

### Using Deploy Script
```bash
# Build and run
./deploy.sh run

# Test API
./deploy.sh test

# Stop container
./deploy.sh stop

# Cleanup
./deploy.sh cleanup
```

## Railway Deployment Steps

1. **Prepare the code:**
   ```bash
   ./deploy.sh railway
   ```

2. **Connect to Railway:**
   ```bash
   railway login
   railway init
   ```

3. **Set environment variables in Railway dashboard:**
   - Go to your project dashboard
   - Navigate to Variables tab
   - Add the environment variables from `.env.example`

4. **Deploy:**
   ```bash
   railway up
   ```

5. **Monitor deployment:**
   - Check Railway dashboard for deployment logs
   - Test the deployed API using the provided URL

## Troubleshooting

### Common Issues

1. **Memory Errors:**
   - Ensure `USE_LIGHTWEIGHT_MODEL=true`
   - Reduce `MAX_MEMORY_MB` if needed
   - Enable `ENABLE_MODEL_QUANTIZATION=true`

2. **Model Loading Issues:**
   - Check if model files exist in the Docker image
   - Verify Hugging Face token if using private models
   - Enable fallback to HF Inference API

3. **Port Issues:**
   - Railway automatically sets the PORT variable
   - Don't manually set PORT in Railway environment variables

4. **Build Timeouts:**
   - Model downloads might cause build timeouts
   - Consider using HF Inference API instead of local models

### Health Check

Test if your deployment is working:
```bash
curl https://your-railway-url.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00"
}
```

## Performance Tips

1. Use lightweight model configuration for Railway
2. Enable model quantization to reduce memory usage
3. Use HF Inference API for large transformer models
4. Implement proper caching strategies
5. Monitor memory usage through the metrics endpoint

## Support

For issues with:
- **Railway deployment**: Check Railway documentation
- **API functionality**: Check FastAPI logs in Railway dashboard
- **Model performance**: Adjust environment variables for your use case
