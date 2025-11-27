# Docker Usage Guide

## Build and Run

### 1. Build Docker Image
```bash
docker-compose build
```

### 2. Start Services
```bash
# Start API and UI
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Access Services
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

## Scaling for Load Testing

### 1. Scale API Instances
```bash
# Start with 3 API containers
docker-compose -f docker-compose.scale.yml up -d --scale api=3

# Scale to 5 containers
docker-compose -f docker-compose.scale.yml up -d --scale api=5
```

### 2. Check Running Containers
```bash
docker-compose ps
docker ps
```

### 3. Monitor Container Stats
```bash
docker stats
```

## Load Testing with Locust

### 1. Install Locust (if not in container)
```bash
pip install locust
```

### 2. Run Load Test
```bash
# Test single container
locust -f locust/loadtest.py --host=http://localhost:8000

# Test with nginx load balancer (scaled containers)
locust -f locust/loadtest.py --host=http://localhost:80
```

### 3. Access Locust UI
- Open: http://localhost:8089
- Set number of users and spawn rate
- Start test and monitor results

## Measuring Response Times

### Record Results for Different Scales:

**1 Container:**
```bash
docker-compose up -d
# Run locust test, record avg response time and requests/sec
```

**3 Containers:**
```bash
docker-compose -f docker-compose.scale.yml up -d --scale api=3
# Run locust test, record metrics
```

**5 Containers:**
```bash
docker-compose -f docker-compose.scale.yml up -d --scale api=5
# Run locust test, record metrics
```

**Results to Record:**
- Average response time (ms)
- Requests per second
- Failure rate
- 95th percentile response time

## Troubleshooting

### View Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs api
docker-compose logs ui

# Follow logs
docker-compose logs -f api
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart api
```

### Clean Up
```bash
# Stop and remove containers
docker-compose down

# Remove volumes (careful - deletes data!)
docker-compose down -v

# Remove images
docker rmi plant_disease_api plant_disease_ui
```

### Check Container Health
```bash
docker inspect --format='{{json .State.Health}}' plant_disease_api
```

## Production Deployment

### Environment Variables
Create `.env` file:
```
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=/app/models/plant_disease_model.keras
```

### Security (Production)
- Use HTTPS/TLS certificates
- Set up proper CORS origins
- Use secrets management for sensitive data
- Enable API rate limiting