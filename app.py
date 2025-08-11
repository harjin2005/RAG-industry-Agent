from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
from rag_news_agent import RAGNewsAgent
import threading
import time
import logging

app = Flask(__name__)
CORS(app)

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instance with thread lock
agent = None
agent_initialized = False
initialization_error = None
agent_lock = threading.Lock()  # Add thread lock for safety

def initialize_agent():
    """Initialize RAG agent with comprehensive error catching"""
    global agent, agent_initialized, initialization_error
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🚀 Attempting to initialize RAG Agent (attempt {attempt + 1}/{max_retries})")
            
            # Add detailed logging for each step
            logger.info("📝 Step 1: About to import RAGNewsAgent...")
            
            # Import and create agent with detailed error catching
            try:
                from rag_news_agent import RAGNewsAgent
                logger.info("📝 Step 2: RAGNewsAgent imported successfully")
            except Exception as import_error:
                logger.error(f"❌ Import failed: {import_error}")
                raise import_error
            
            logger.info("📝 Step 3: Creating RAGNewsAgent instance...")
            temp_agent = RAGNewsAgent()
            logger.info("📝 Step 4: RAGNewsAgent instance created successfully")
            
            # Use lock to ensure atomic assignment
            with agent_lock:
                agent = temp_agent
                agent_initialized = True
                initialization_error = None
            
            logger.info("✅ RAG Agent initialized successfully for web interface")
            return
            
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            logger.error(f"❌ Full error details: {repr(e)}")
            
            with agent_lock:
                initialization_error = str(e)
            
            if attempt < max_retries - 1:
                logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("❌ All initialization attempts failed")
                with agent_lock:
                    agent_initialized = True  # Mark as attempted to stop retries

# Initialize agent when app starts - HYBRID APPROACH
logger.info("🎯 Starting RAG News Intelligence Platform...")

# Try synchronous initialization first
logger.info("🔄 Attempting synchronous initialization...")
try:
    from rag_news_agent import RAGNewsAgent
    agent = RAGNewsAgent()
    agent_initialized = True
    logger.info("✅ Synchronous initialization successful!")
except Exception as sync_error:
    logger.error(f"❌ Synchronous initialization failed: {sync_error}")
    # Fall back to threaded initialization
    logger.info("🔄 Falling back to threaded initialization...")
    initialization_thread = threading.Thread(target=initialize_agent, daemon=True)
    initialization_thread.start()

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return jsonify({'error': 'Could not load main page'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_industry():
    """Main endpoint for industry analysis with RAG enhancement"""
    try:
        # Check if request has valid JSON
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        industry = data.get('industry', '').strip()
        
        # Validate input
        if not industry:
            return jsonify({'error': 'Industry name is required'}), 400
        
        if len(industry) > 100:
            return jsonify({'error': 'Industry name too long (max 100 characters)'}), 400
        
        # Thread-safe agent check with small delay for initialization
        with agent_lock:
            current_agent = agent
            current_initialized = agent_initialized
            current_error = initialization_error
        
        # If agent is still initializing, wait a bit
        if not current_initialized:
            logger.info("⏳ Agent still initializing, waiting...")
            time.sleep(3)  # Wait 3 seconds
            
            # Check again after waiting
            with agent_lock:
                current_agent = agent
                current_initialized = agent_initialized
                current_error = initialization_error
        
        # Check agent status after potential wait
        if not current_initialized:
            return jsonify({
                'error': 'RAG Agent is still initializing. Please wait and try again.',
                'retry_after': 10
            }), 503
        
        if not current_agent:
            error_detail = f"Agent failed to initialize: {current_error}" if current_error else "Unknown initialization error"
            return jsonify({
                'error': 'RAG Agent initialization failed',
                'details': error_detail
            }), 503
        
        # Log the analysis request
        logger.info(f"🔍 Analyzing industry: {industry}")
        
        # Perform analysis with timeout protection
        start_time = time.time()
        result = current_agent.analyze(industry)
        analysis_time = time.time() - start_time
        
        logger.info(f"✅ Analysis completed for '{industry}' in {analysis_time:.2f}s")
        
        return jsonify({
            'success': True,
            'industry': industry,
            'analysis': result,
            'timestamp': time.time(),
            'analysis_time': round(analysis_time, 2),
            'server_info': 'RAG-Powered News Intelligence Platform by Harjinder Singh'
        })
        
    except Exception as e:
        logger.error(f"❌ Analysis failed for industry '{industry if 'industry' in locals() else 'unknown'}': {e}")
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'industry': industry if 'industry' in locals() else 'unknown'
        }), 500

@app.route('/api/status')
def get_status():
    """Health check endpoint with detailed debugging info"""
    try:
        # Check environment variables
        groq_key_exists = bool(os.getenv("GROQ_API_KEY"))
        thenews_key_exists = bool(os.getenv("THENEWS_API_KEY"))
        
        # Thread-safe status check
        with agent_lock:
            current_agent = agent
            current_initialized = agent_initialized
            current_error = initialization_error
        
        status_info = {
            'agent_ready': current_agent is not None and current_initialized,
            'agent_initialized': current_initialized,
            'timestamp': time.time(),
            'server_status': 'running',
            'platform': 'RAG News Intelligence Platform',
            'author': 'Harjinder Singh',
            # DEBUG INFO - This will show us what's wrong
            'debug': {
                'groq_api_key_exists': groq_key_exists,
                'thenews_api_key_exists': thenews_key_exists,
                'initialization_error': current_error,
                'agent_object_exists': current_agent is not None,
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
                'working_directory': os.getcwd(),
                'environment_vars_count': len([k for k in os.environ.keys() if not k.startswith('_')]),
                'threading_active_count': threading.active_count()
            }
        }
        
        return jsonify(status_info)
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({
            'agent_ready': False,
            'server_status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/api/retry-init', methods=['POST'])
def retry_initialization():
    """Endpoint to manually retry agent initialization"""
    global agent_initialized
    
    with agent_lock:
        current_agent = agent
        current_initialized = agent_initialized
    
    if current_agent and current_initialized:
        return jsonify({
            'message': 'Agent is already initialized',
            'agent_ready': True
        })
    
    # Reset initialization flag and retry
    with agent_lock:
        agent_initialized = False
        initialization_error = None
    
    initialization_thread = threading.Thread(target=initialize_agent, daemon=True)
    initialization_thread.start()
    
    return jsonify({
        'message': 'Initialization retry started',
        'agent_ready': False,
        'check_status_in': 10
    })

@app.route('/api/debug-logs')
def get_debug_logs():
    """Endpoint to get recent debug information"""
    try:
        debug_info = {
            'timestamp': time.time(),
            'system_info': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'working_directory': os.getcwd(),
                'threading_count': threading.active_count(),
                'environment_variables': {
                    'GROQ_API_KEY_SET': bool(os.getenv("GROQ_API_KEY")),
                    'THENEWS_API_KEY_SET': bool(os.getenv("THENEWS_API_KEY")),
                    'PYTHON_VERSION': os.getenv("PYTHON_VERSION", "Not set"),
                    'PORT': os.getenv("PORT", "Not set")
                }
            },
            'agent_status': {
                'initialized': agent_initialized,
                'object_exists': agent is not None,
                'initialization_error': initialization_error
            }
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({
            'error': f'Debug info failed: {str(e)}',
            'timestamp': time.time()
        }), 500

@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors"""
    return '', 204

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Health check for Render deployment
@app.route('/health')
def health_check():
    """Simple health check for deployment platforms"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'service': 'RAG News Intelligence Platform'
    })

if __name__ == '__main__':
    # Development server configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"🚀 Starting Flask server on port {port}")
    logger.info(f"🔧 Debug mode: {debug}")
    
    app.run(
        debug=debug,
        host='0.0.0.0',
        port=port,
        threaded=True  # Enable threading for better concurrent handling
    )
