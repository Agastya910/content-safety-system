# QUICK_COMMANDS.md

# Event Ingestion + Risk Screening - Quick Command Reference

## 🚀 Quick Start (Copy & Paste)

```bash
# 1. Start everything
docker-compose -f docker-compose-extended.yml up -d

# 2. Wait for services to be healthy
sleep 10

# 3. Verify services
curl http://localhost:8001/health && echo "✓ Event Ingestion OK"
curl http://localhost:8002/health && echo "✓ Risk Screening OK"
```

## 📤 Ingest Events

### Single Event

```bash
curl -X POST http://localhost:8001/v1/events/ingest \
  -H "Content-Type: application/json" \
  -H "api-key: test-key" \
  -d '{
    "event_id": "evt-'$(date +%s%N)'",
    "event_type": "message_created",
    "platform": "discord",
    "channel_id": "ch-123",
    "user_id": "user-123",
    "author_id": "author-123",
    "content": "This is a normal message",
    "metadata": {"user_reputation": 0.8}
  }'
```

### Toxic Content

```bash
curl -X POST http://localhost:8001/v1/events/ingest \
  -H "Content-Type: application/json" \
  -H "api-key: test-key" \
  -d '{
    "event_id": "evt-'$(date +%s%N)'",
    "event_type": "message_created",
    "platform": "discord",
    "user_id": "user-123",
    "author_id": "author-123",
    "content": "You are such a stupid idiot!!!",
    "metadata": {"user_reputation": 0.2}
  }'
```

### Spam Content

```bash
curl -X POST http://localhost:8001/v1/events/ingest \
  -H "Content-Type: application/json" \
  -H "api-key: test-key" \
  -d '{
    "event_id": "evt-'$(date +%s%N)'",
    "event_type": "message_created",
    "platform": "discord",
    "user_id": "user-123",
    "author_id": "author-123",
    "content": "CLICK HERE!!! AMAZING OFFER!!! BUY NOW!!!",
    "metadata": {}
  }'
```

### Batch Ingest (100 events)

```bash
BATCH_SIZE=100
EVENTS="["
for i in $(seq 1 $BATCH_SIZE); do
  EVENT="{\"event_id\":\"evt-$i\",\"event_type\":\"message_created\",\"platform\":\"discord\",\"user_id\":\"user-123\",\"author_id\":\"author-$i\",\"content\":\"Test message $i\"}"
  EVENTS="$EVENTS$EVENT"
  if [ $i -lt $BATCH_SIZE ]; then EVENTS="$EVENTS,"; fi
done
EVENTS="$EVENTS]"

curl -X POST http://localhost:8001/v1/events/ingest-batch \
  -H "Content-Type: application/json" \
  -H "api-key: test-key" \
  -d "{\"events\":$EVENTS}"
```

## 🎯 Screen Events

```bash
curl -X POST http://localhost:8002/v1/risk/screen \
  -H "api-key: test-key" \
  -d 'event_id=evt-test&content=This message should be checked'
```

## 📊 Check Metrics

### Event Ingestion Metrics

```bash
curl http://localhost:8001/v1/metrics | jq '.'

# Pretty print
curl http://localhost:8001/v1/metrics | jq '{
  queue_depth: .queue_depth,
  queue_capacity_percent: .queue_capacity_percent,
  backpressure_enabled: .backpressure_enabled
}'
```

### Risk Screening Metrics

```bash
curl http://localhost:8002/v1/metrics | jq '.'

# Pretty print
curl http://localhost:8002/v1/metrics | jq '{
  model: .model,
  device: .device,
  queue_depth: .queue_depth,
  threshold: .threshold,
  embedding_dim: .embedding_dim
}'
```

### Prometheus

```bash
# Query event rate
curl 'http://localhost:9090/api/v1/query?query=rate(events_received_total[5m])'

# Query latency P99
curl 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.99,ingestion_latency_ms)'
```

## 🔍 Monitor

### Logs

```bash
# All services
docker-compose -f docker-compose-extended.yml logs -f

# Event Ingestion only
docker-compose -f docker-compose-extended.yml logs -f event-ingestion

# Risk Screening only
docker-compose -f docker-compose-extended.yml logs -f risk-screening

# Follow with timestamps
docker-compose -f docker-compose-extended.yml logs -f --timestamps
```

### Dashboards

```bash
# Prometheus
open http://localhost:9090

# Grafana (admin/admin)
open http://localhost:3000

# Jaeger traces
open http://localhost:16686

# Locust load testing
open http://localhost:8089
```

### Redis Inspection

```bash
# Connect to Redis
docker exec -it redis redis-cli

# Check streams
XLEN events:raw
XLEN events:reasoning_queue
XLEN events:low_risk

# Check dedup cache size
DBSIZE

# Check memory
INFO memory

# Exit
QUIT
```

### Check Service Status

```bash
# List running containers
docker-compose -f docker-compose-extended.yml ps

# Check container health
docker inspect event-ingestion | jq '.[] | .State'

# Check resource usage
docker stats
```

## 🧪 Testing

### Run Integration Tests

```bash
# All tests
pytest testing/integration_tests.py -v

# Specific test class
pytest testing/integration_tests.py::TestEventIngestion -v

# Specific test
pytest testing/integration_tests.py::TestEventIngestion::test_single_event_ingestion -v

# With output
pytest testing/integration_tests.py -v -s

# With coverage
pytest testing/integration_tests.py --cov=services
```

### Run Performance Tests

```bash
# Ingestion latency benchmark
pytest testing/integration_tests.py::TestPerformance::test_ingestion_latency -v -s

# Screening latency benchmark
pytest testing/integration_tests.py::TestPerformance::test_screening_latency -v -s

# High throughput test
pytest testing/integration_tests.py::TestEndToEnd::test_high_throughput -v -s
```

### Load Testing with Locust

```bash
# Start Locust
docker-compose -f docker-compose-extended.yml up locust

# Access web UI
open http://localhost:8089

# Or run headless
locust -f testing/locustfile.py \
  --headless \
  --users 1000 \
  --spawn-rate 50 \
  --run-time 5m \
  --host http://localhost:8001
```

## 🛠️ Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose -f docker-compose-extended.yml logs event-ingestion

# Rebuild image
docker-compose -f docker-compose-extended.yml build --no-cache event-ingestion

# Restart service
docker-compose -f docker-compose-extended.yml restart event-ingestion

# Restart all
docker-compose -f docker-compose-extended.yml restart
```

### High Queue Depth

```bash
# Check queue depth
curl http://localhost:8001/v1/metrics | jq '.queue_depth'

# Check screening service
curl http://localhost:8002/v1/metrics | jq '.queue_depth'

# Check logs
docker-compose logs -f risk-screening

# Restart screening
docker-compose restart risk-screening
```

### High Memory Usage

```bash
# Check memory
docker stats

# Check model loading
docker-compose logs risk-screening | grep -i "loading\|memory\|cuda"

# Reduce batch size
docker-compose -f docker-compose-extended.yml exec risk-screening \
  env | grep BATCH_SIZE

# Restart with new config
docker-compose -f docker-compose-extended.yml down
# Edit environment variables
docker-compose -f docker-compose-extended.yml up -d risk-screening
```

### Redis Connection Error

```bash
# Check Redis status
docker-compose -f docker-compose-extended.yml ps redis

# Check Redis logs
docker-compose -f docker-compose-extended.yml logs redis

# Restart Redis
docker-compose -f docker-compose-extended.yml restart redis

# Verify connection
docker exec redis redis-cli PING
```

## 📈 Performance Tuning

### For Higher Throughput

```bash
# Increase batch sizes
BATCH_SIZE=1000                    # Event Ingestion
SCREENING_BATCH_SIZE=200           # Risk Screening

# Increase workers
WORKERS=8                          # Both services

# Increase concurrency
docker-compose -f docker-compose-extended.yml up -d --scale event-ingestion=2
```

### For Lower Latency

```bash
# Decrease batch timeout
BATCH_TIMEOUT_MS=50                # Event Ingestion
BATCH_TIMEOUT_MS=100               # Risk Screening

# Smaller batches
BATCH_SIZE=50                      # Event Ingestion
BATCH_SIZE=32                      # Risk Screening
```

### For GPU Support

```bash
# Enable GPU in docker-compose
# Uncomment in risk-screening service:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]

# Or via environment
DEVICE=cuda
MODEL_BATCH_SIZE=64                # Can increase with GPU
USE_FP32=false                     # Use FP16 on GPU
```

## 🧹 Cleanup

### Stop Services

```bash
# Stop all containers
docker-compose -f docker-compose-extended.yml down

# Stop and remove volumes
docker-compose -f docker-compose-extended.yml down -v

# Remove images
docker-compose -f docker-compose-extended.yml down --rmi all
```

### Clear Caches

```bash
# Clear Redis
docker exec redis redis-cli FLUSHALL

# Clear deduplication cache
docker exec redis redis-cli FLUSHDB

# Clear embedding cache
docker exec redis redis-cli DEL "embedding:*"
```

### Clean Logs

```bash
# View logs without history
docker-compose -f docker-compose-extended.yml logs --tail=100

# Clear logs
docker-compose -f docker-compose-extended.yml logs --no-log-prefix > /dev/null
```

## 📝 Common Scenarios

### Test High Load

```bash
# Ingest 10,000 events rapidly
for i in {1..10}; do
  curl -X POST http://localhost:8001/v1/events/ingest-batch \
    -H "api-key: test-key" \
    -d @- << 'EOF'
{
  "events": [
    {"event_id":"evt-$i-$j","content":"Test $i-$j"}
    for j in {1..1000}
  ]
}
EOF
done

# Monitor queue
watch 'curl -s http://localhost:8001/v1/metrics | jq ".queue_depth"'
```

### Test Duplicate Detection

```bash
# Create and send duplicate
CONTENT="duplicate content test"
for i in {1..3}; do
  curl -X POST http://localhost:8001/v1/events/ingest \
    -H "api-key: test-key" \
    -d "{\"event_id\":\"evt-dup-$i\",\"content\":\"$CONTENT\",\"author_id\":\"same-author\",\"platform\":\"discord\"}"
done

# Check results - last 2 should show "duplicate" status
```

### Monitor During Load Test

```bash
# In one terminal
docker-compose -f docker-compose-extended.yml logs -f --tail=50 | grep -E "Ingested|Screened|error"

# In another terminal
watch -n 1 'curl -s http://localhost:8001/v1/metrics | jq ".queue_depth"'

# In another
watch -n 1 'curl -s http://localhost:8002/v1/metrics | jq ".queue_depth"'

# Run load test
pytest testing/integration_tests.py::TestEndToEnd::test_high_throughput -v -s
```

## 📞 Quick Links

- Event Ingestion API Docs: http://localhost:8001/docs
- Risk Screening API Docs: http://localhost:8002/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Jaeger: http://localhost:16686
- Locust: http://localhost:8089

---

**Pro Tip:** Save this file and run `source QUICK_COMMANDS.md` in your shell for quick access to all commands.
