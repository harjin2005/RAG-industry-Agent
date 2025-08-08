from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from rag_news_agent import RAGNewsAgent
import threading
import time

app = Flask(__name__)
CORS(app)

# Global agent instance
agent = None
agent_initialized = False

def initialize_agent():
    """Initialize RAG agent in background"""
    global agent, agent_initialized
    try:
        agent = RAGNewsAgent()
        agent_initialized = True
        print("✅ RAG Agent initialized for web interface")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        agent_initialized = True  # Mark as attempted

# ✅ FIXED: Initialize agent immediately when app starts
with app.app_context():
    threading.Thread(target=initialize_agent, daemon=True).start()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_industry():
    try:
        data = request.get_json()
        industry = data.get('industry', '').strip()
        
        if not industry:
            return jsonify({'error': 'Industry name is required'}), 400
        
        if not agent:
            return jsonify({'error': 'RAG Agent is still initializing. Please try again.'}), 503
        
        # Use your existing analyze method
        result = agent.analyze(industry)
        
        return jsonify({
            'success': True,
            'industry': industry,
            'analysis': result,
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/status')
def get_status():
    return jsonify({
        'agent_ready': agent is not None and agent_initialized,
        'timestamp': time.time()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
