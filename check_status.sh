#!/bin/bash

# Application Status Checker
# Verifies the Streamlit app is running correctly

echo "🔍 TELECOM CHURN APP - STATUS CHECK"
echo "======================================"

# Check if Streamlit is running
echo ""
echo "1. Checking Streamlit Process..."
if pgrep -f "streamlit run app.py" > /dev/null; then
    echo "   ✅ Streamlit is running (PID: $(pgrep -f 'streamlit run app.py'))"
else
    echo "   ❌ Streamlit is NOT running"
    echo "   Run: ./run_app.sh"
    exit 1
fi

# Check HTTP response
echo ""
echo "2. Checking HTTP Response..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501)
if [ "$HTTP_CODE" -eq 200 ]; then
    echo "   ✅ App responding (HTTP $HTTP_CODE)"
else
    echo "   ❌ App not responding (HTTP $HTTP_CODE)"
    exit 1
fi

# Check required files
echo ""
echo "3. Checking Required Files..."
FILES=(
    "app.py"
    "utils/session_state.py"
    "utils/data_utils.py"
    "utils/model_utils.py"
    "pages/1_📤_Data_Upload.py"
    "pages/2_🤖_Model_Training.py"
    "pages/3_🔍_SHAP_Analysis.py"
    "pages/4_⚠️_Risk_Segmentation.py"
    "pages/5_📊_Dashboard.py"
    "data/churndata.csv"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ MISSING: $file"
        exit 1
    fi
done

# Check Python environment
echo ""
echo "4. Checking Python Environment..."
if [ -d ".venv" ]; then
    echo "   ✅ Virtual environment found"
    PYTHON_VERSION=$(.venv/bin/python --version)
    echo "   ℹ️  $PYTHON_VERSION"
else
    echo "   ⚠️  No virtual environment found"
fi

# Summary
echo ""
echo "======================================"
echo "✅ ALL CHECKS PASSED!"
echo "======================================"
echo ""
echo "📊 App URL: http://localhost:8501"
echo "🌐 Network URL: http://$(ipconfig getifaddr en0):8501"
echo ""
echo "🛑 To stop the app:"
echo "   pkill -f 'streamlit run app.py'"
echo ""
echo "🔄 To restart:"
echo "   ./run_app.sh"
echo "======================================"
