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

# Function to backup current configuration
backup_config() {
    echo -e "${BLUE}Backing up current configuration...${NC}"
    
    backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup configuration files
    cp -r config "$backup_dir"/ 2>/dev/null || true
    cp -r grafana/provisioning "$backup_dir"/grafana_provisioning 2>/dev/null || true
    cp -r prometheus "$backup_dir"/prometheus 2>/dev/null || true
    cp .env "$backup_dir"/ 2>/dev/null || true
    
    echo -e "${GREEN}Configuration backed up to $backup_dir${NC}"
}

# Function to update Docker images
update_images() {
    echo -e "${BLUE}Updating Docker images...${NC}"
    
    # Pull latest images
    docker-compose pull
    
    echo -e "${GREEN}Docker images updated${NC}"
}

# Function to check configuration changes
check_config_changes() {
    echo -e "${BLUE}Checking configuration changes...${NC}"
    
    # Compare current config with new version
    if [ -f "config/dashboard.config.js.new" ]; then
        diff -u config/dashboard.config.js config/dashboard.config.js.new || true
        read -p "Apply new configuration? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            mv config/dashboard.config.js.new config/dashboard.config.js
        fi
    fi
}

# Function to verify services health
verify_services() {
    echo -e "${BLUE}Verifying services health...${NC}"
    
    services=("dashboard" "redis" "prometheus" "grafana" "nginx")
    failed_services=()
    
    for service in "${services[@]}"; do
        echo -n "Checking $service... "
        if docker-compose ps $service | grep -q "Up"; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}Failed${NC}"
            failed_services+=($service)
        fi
    done
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        echo -e "${RED}Some services failed to start:${NC}"
        for service in "${failed_services[@]}"; do
            echo "- $service"
        done
        exit 1
    fi
}

# Function to update monitoring configuration
update_monitoring() {
    echo -e "${BLUE}Updating monitoring configuration...${NC}"
    
    # Update Prometheus config if needed
    if [ -f "prometheus/prometheus.yml.new" ]; then
        mv prometheus/prometheus.yml.new prometheus/prometheus.yml
        docker-compose restart prometheus
    fi
    
    # Update Grafana dashboards
    if [ -d "grafana/provisioning/dashboards/new" ]; then
        cp -r grafana/provisioning/dashboards/new/* grafana/provisioning/dashboards/
        rm -rf grafana/provisioning/dashboards/new
        docker-compose restart grafana
    fi
}

# Main update function
main() {
    echo -e "${BLUE}Starting update process...${NC}"
    
    # Check if docker-compose is running
    if ! docker-compose ps > /dev/null 2>&1; then
        echo -e "${RED}Docker Compose services are not running${NC}"
        exit 1
    fi
    
    # Perform update steps
    backup_config
    update_images
    check_config_changes
    
    echo -e "${BLUE}Restarting services...${NC}"
    docker-compose down
    docker-compose up -d
    
    update_monitoring
    verify_services
    
    echo -e "${GREEN}Update completed successfully!${NC}"
    
    # Show version information
    echo -e "\n${BLUE}Current versions:${NC}"
    docker-compose exec dashboard npm version | grep node-dashboard
    docker-compose exec grafana grafana-server -v
    docker-compose exec prometheus prometheus --version
    
    echo -e "\n${YELLOW}Note: Check the documentation for any breaking changes${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_UPDATE=true
            shift
            ;;
        --no-backup)
            NO_BACKUP=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --force      Force update without prompts"
            echo "  --no-backup  Skip configuration backup"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main update
main