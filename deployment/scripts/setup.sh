#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Error handling
set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo -e "${RED}ERROR: Command \"${last_command}\" failed with exit code $?.${NC}"' EXIT

# Function to check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    
    # Check Docker version
    if ! docker --version > /dev/null 2>&1; then
        echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    
    # Check Docker Compose version
    if ! docker-compose --version > /dev/null 2>&1; then
        echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
        exit 1
    fi
    
    # Check if ports are available
    if netstat -tln | grep -q ':80\|:443\|:3000\|:3001\|:9090'; then
        echo -e "${YELLOW}Warning: Some required ports (80, 443, 3000, 3001, 9090) are already in use.${NC}"
        read -p "Do you want to continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to create directory structure
create_directories() {
    echo -e "${BLUE}Creating directory structure...${NC}"
    directories=(
        "logs/nginx"
        "ssl"
        "config"
        "grafana/provisioning/datasources"
        "grafana/provisioning/dashboards"
        "prometheus"
        "nginx/conf.d"
        "data/redis"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        echo -e "${GREEN}Created directory: $dir${NC}"
    done
}

# Function to generate SSL certificates
generate_ssl_certificates() {
    echo -e "${BLUE}Setting up SSL certificates...${NC}"
    
    if [ ! -f ssl/certificate.crt ]; then
        echo -e "${YELLOW}No SSL certificate found. Generating self-signed certificate...${NC}"
        
        # Generate SSL configuration
        cat > ssl/openssl.conf << EOL
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = Organization
OU = DevOps
CN = localhost

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = node-dashboard
IP.1 = 127.0.0.1
EOL
        
        # Generate private key and certificate
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ssl/private.key -out ssl/certificate.crt \
            -config ssl/openssl.conf
        
        echo -e "${GREEN}SSL certificate generated successfully${NC}"
    else
        echo -e "${GREEN}Using existing SSL certificate${NC}"
    fi
}

# Function to create environment configuration
create_env_config() {
    echo -e "${BLUE}Setting up environment configuration...${NC}"
    
    if [ ! -f .env ]; then
        echo -e "${YELLOW}No .env file found. Creating new one...${NC}"
        
        # Generate random auth token
        AUTH_TOKEN=$(openssl rand -hex 32)
        
        # Generate random Grafana password
        GRAFANA_PASSWORD=$(openssl rand -base64 12)
        
        cat > .env << EOL
# Node Configuration
NODE_API_URL=http://localhost:8000
NODE_WS_URL=ws://localhost:8001
AUTH_TOKEN=${AUTH_TOKEN}

# Grafana Configuration
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}

# Redis Configuration
REDIS_PASSWORD=$(openssl rand -base64 12)

# Monitoring Configuration
METRICS_RETENTION_DAYS=30
ENABLE_DETAILED_METRICS=true

# Security Configuration
ENABLE_SSL=true
ENABLE_RATE_LIMITING=true
MAX_REQUESTS_PER_MINUTE=60
EOL
        echo -e "${GREEN}Environment configuration created successfully${NC}"
    else
        echo -e "${GREEN}Using existing .env file${NC}"
    fi
}

# Function to setup Grafana
setup_grafana() {
    echo -e "${BLUE}Setting up Grafana...${NC}"
    
    # Create Prometheus datasource
    cat > grafana/provisioning/datasources/prometheus.yml << EOL
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
EOL
    
    # Create default dashboard
    cat > grafana/provisioning/dashboards/node-metrics.json << EOL
{
  "annotations": {
    "list": []
  },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 0,
  "links": [],
  "liveNow": false,
  "panels": [
    {
      "datasource": {
        "type": "prometheus",
        "uid": "prometheus"
      },
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisCenteredZero": false,
            "axisColorMode": "text",
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "drawStyle": "line",
            "fillOpacity": 20,
            "gradientMode": "none",
            "hideFrom": {
              "legend": false,
              "tooltip": false,
              "viz": false
            },
            "lineInterpolation": "smooth",
            "lineWidth": 2,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "never",
            "spanNulls": true,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              }
            ]
          },
          "unit": "percent"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "id": 1,
      "options": {
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom",
          "showLegend": true
        },
        "tooltip": {
          "mode": "single",
          "sort": "none"
        }
      },
      "title": "GPU Utilization",
      "type": "timeseries"
    }
  ],
  "refresh": "5s",
  "schemaVersion": 38,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Node Metrics",
  "version": 0
}
EOL
}

# Function to setup monitoring
setup_monitoring() {
    echo -e "${BLUE}Setting up monitoring...${NC}"
    
    # Configure node exporter
    docker run -d \
        --name node-exporter \
        --network node-network \
        --restart unless-stopped \
        -p 9100:9100 \
        -v "/proc:/host/proc:ro" \
        -v "/sys:/host/sys:ro" \
        -v "/:/rootfs:ro" \
        quay.io/prometheus/node-exporter:latest
        
    # Setup prometheus alerts
    cat > prometheus/alerts.yml << EOL
groups:
  - name: node_alerts
    rules:
      - alert: HighGPUTemperature
        expr: gpu_temperature > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High GPU temperature
          description: GPU temperature has been above 80°C for 5 minutes
          
      - alert: LowGPUUtilization
        expr: avg_over_time(gpu_utilization[15m]) < 30
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: Low GPU utilization
          description: GPU utilization has been below 30% for 30 minutes
EOL
}

# Function to check service health
check_service_health() {
    local service=$1
    local max_attempts=30
    local attempt=1
    
    echo -e "${BLUE}Checking health of $service...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps $service | grep -q "Up"; then
            echo -e "${GREEN}$service is healthy${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    
    echo -e "\n${RED}$service failed to start${NC}"
    return 1
}

# Main setup function
main() {
    echo -e "${BLUE}Starting Node Dashboard setup...${NC}"
    
    # Run setup steps
    check_prerequisites
    create_directories
    generate_ssl_certificates
    create_env_config
    setup_grafana
    setup_monitoring
    
    # Start services
    echo -e "${BLUE}Starting services...${NC}"
    docker-compose up -d
    
    # Check service health
    services=("dashboard" "redis" "prometheus" "grafana" "nginx")
    failed_services=()
    
    for service in "${services[@]}"; do
        if ! check_service_health $service; then
            failed_services+=($service)
        fi
    done
    
    # Print setup results
    if [ ${#failed_services[@]} -eq 0 ]; then
        echo -e "\n${GREEN}Setup completed successfully!${NC}"
        echo -e "\nAccess URLs:"
        echo -e "Dashboard: https://localhost"
        echo -e "Grafana: https://localhost/grafana (admin/$(grep GRAFANA_PASSWORD .env | cut -d= -f2))"
        echo -e "Prometheus: https://localhost/prometheus (admin/secure_password)"
        
        echo -e "\n${YELLOW}Important Notes:${NC}"
        echo "1. The SSL certificate is self-signed. You may see browser warnings."
        echo "2. Save your credentials from the .env file"
        echo "3. Configure your firewall to protect the monitoring endpoints"
        
        echo -e "\n${BLUE}Useful commands:${NC}"
        echo "- View logs: docker-compose logs -f [service]"
        echo "- Restart services: docker-compose restart"
        echo "- Stop services: docker-compose down"
        echo "- Update services: ./update.sh"
    else
        echo -e "\n${RED}Setup failed for the following services:${NC}"
        for service in "${failed_services[@]}"; do
            echo "- $service"
        done
        echo -e "\nCheck the logs with: docker-compose logs ${failed_services[0]}"
        exit 1
    fi
}

# Run main setup
main "$@"

# Remove error trap
trap - EXIT- View logs: docker-compose logs -f"
echo "- Restart services: docker-compose restart"
echo "- Stop services: docker-compose down"
echo "- Update services: docker-compose pull && docker-compose up -d"