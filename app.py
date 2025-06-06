import os
import logging
from flask import Flask, request, redirect, session, make_response, render_template_string
from werkzeug.middleware.proxy_fix import ProxyFix
import shopify
from dotenv import load_dotenv
import ssl
import certifi
from fix_certificates import fix_certificates

# Configure SSL
ssl_context = fix_certificates()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure Shopify
SHOPIFY_API_KEY = os.getenv('SHOPIFY_API_KEY')
SHOPIFY_API_SECRET = os.getenv('SHOPIFY_API_SECRET')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')

# Environment configuration
IS_REPLIT = os.getenv('REPL_ID') is not None
if IS_REPLIT:
    APP_URL = "https://customer-lifetime-analyzer-justinleu1.replit.app"
else:
    APP_URL = "http://localhost:8000"

SHOPIFY_SCOPES = [
    'read_products', 'write_products',
    'read_orders', 'write_orders',
    'read_customers', 'write_customers',
    'read_analytics'
]

# Your store URL
STORE_URL = "https://clv-test-store.myshopify.com"  # You can change this to your actual store URL

@app.route('/')
def index():
    logger.debug("Starting index route")
    return render_template_string("""
        <h1>Welcome to CustomerLifetimeAnalyzer</h1>
        <p>Click below to authenticate with Shopify:</p>
        <a href="/auth/start">Start Shopify Authentication</a>
        """)

@app.route('/auth/start')
def auth_start():
    logger.debug("Starting auth start route")
    shop_url = "https://clv-test-store.myshopify.com"
    
    # Generate a random state value
    state = os.urandom(16).hex()
    session['state'] = state
    
    # Set up the Shopify session
    shopify.Session.setup(api_key=SHOPIFY_API_KEY, secret=SHOPIFY_API_SECRET)
    shopify_session = shopify.Session(shop_url, '2023-07')
    
    # Generate the authorization URL
    auth_url = shopify_session.create_permission_url(
        SHOPIFY_SCOPES,
        request.url_root + 'auth/callback',
        state
    )
    
    logger.debug(f"Generated auth URL: {auth_url}")
    return redirect(auth_url)

@app.route('/auth/callback')
def callback():
    logger.debug("Starting callback route")
    logger.debug(f"Session state: {session.get('state')}")
    logger.debug(f"Request args: {request.args}")
    
    # Verify state parameter
    if session.get('state') != request.args.get('state'):
        logger.error("State verification failed")
        return "State verification failed", 403
    
    shop_url = "https://clv-test-store.myshopify.com"
    
    # Set up the Shopify session
    shopify.Session.setup(api_key=SHOPIFY_API_KEY, secret=SHOPIFY_API_SECRET)
    shopify_session = shopify.Session(shop_url, '2023-07')
    
    try:
        # Request the access token
        access_token = shopify_session.request_token(request.args)
        
        # Create a new session with the access token
        shopify_session = shopify.Session(shop_url, '2023-07', access_token)
        shopify.ShopifyResource.activate_session(shopify_session)
        
        # Get the shop information
        shop = shopify.Shop.current()
        
        # Return success message
        return f"""Successfully authenticated with Shopify!<br>
                Shop Name: {shop.name}<br>
                Access Token: {access_token[:5]}...{access_token[-5:]}<br>
                Shop Email: {shop.email}"""
                
    except Exception as e:
        logger.error(f"Error in Shopify authentication: {e}")
        return f"Error authenticating with Shopify: {e}"
    finally:
        shopify.ShopifyResource.clear_session()

if __name__ == '__main__':
    # For local development
    app.run(host='0.0.0.0', port=8000, debug=True)
