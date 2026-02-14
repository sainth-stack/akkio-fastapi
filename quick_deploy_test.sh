#!/bin/bash
# Quick Deployment API Test Script
# Run this on your FastAPI EC2 server

echo "======================================"
echo "Deployment API Quick Test"
echo "======================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BASE_URL="http://localhost:3001"
PROJECT_NAME="generated_app_1770640849576"
USER_EMAIL="admin@gmail.com"

# Test 1: Health Check
echo -e "\n${YELLOW}[1/5]${NC} Testing FastAPI health..."
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" $BASE_URL/health)
if [ "$HEALTH" = "200" ]; then
    echo -e "${GREEN}✅${NC} FastAPI is running"
else
    echo -e "${RED}❌${NC} FastAPI is not responding (HTTP $HEALTH)"
    echo "   Please start FastAPI first"
    exit 1
fi

# Test 2: Check if deployment routes exist
echo -e "\n${YELLOW}[2/5]${NC} Checking if deployment routes are registered..."
ROUTES=$(curl -s $BASE_URL/openapi.json | grep -c "/api/deployment")
if [ "$ROUTES" -gt "0" ]; then
    echo -e "${GREEN}✅${NC} Deployment routes found"
else
    echo -e "${RED}❌${NC} Deployment routes not found"
    echo "   Did you restart FastAPI after code changes?"
    exit 1
fi

# Test 3: Check environment variables
echo -e "\n${YELLOW}[3/5]${NC} Checking environment variables..."
if [ -z "$DEPLOY_EC2_IP" ]; then
    echo -e "${RED}❌${NC} DEPLOY_EC2_IP not set"
    echo "   Set it in .env file"
else
    echo -e "${GREEN}✅${NC} DEPLOY_EC2_IP = $DEPLOY_EC2_IP"
fi

if [ -z "$DEPLOY_SSH_KEY_PATH" ]; then
    echo -e "${RED}❌${NC} DEPLOY_SSH_KEY_PATH not set"
else
    echo -e "${GREEN}✅${NC} DEPLOY_SSH_KEY_PATH = $DEPLOY_SSH_KEY_PATH"
fi

# Test 4: Test deploy endpoint
echo -e "\n${YELLOW}[4/5]${NC} Testing deployment endpoint..."
DEPLOY_RESPONSE=$(curl -s -X POST $BASE_URL/api/deployment/deploy \
  -H "Content-Type: application/json" \
  -d "{\"app_id\":null,\"project_name\":\"$PROJECT_NAME\",\"user_email\":\"$USER_EMAIL\"}")

DEPLOY_STATUS=$(echo $DEPLOY_RESPONSE | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

if [ "$DEPLOY_STATUS" = "started" ]; then
    echo -e "${GREEN}✅${NC} Deploy endpoint working"
    echo "   Response: $DEPLOY_RESPONSE"
else
    echo -e "${RED}❌${NC} Deploy endpoint failed"
    echo "   Response: $DEPLOY_RESPONSE"
fi

# Test 5: Test status endpoint
echo -e "\n${YELLOW}[5/5]${NC} Testing status endpoint..."
STATUS_RESPONSE=$(curl -s "$BASE_URL/api/deployment/status?project_name=$PROJECT_NAME&user_email=$USER_EMAIL")
echo "   Response: $STATUS_RESPONSE"

echo -e "\n======================================"
echo -e "${GREEN}All tests complete!${NC}"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. If any tests failed, check the error messages above"
echo "2. Try deploying from the React UI"
echo "3. Monitor logs: tail -f /path/to/fastapi.log"
echo ""
