from flask import Flask, request, session as flask_session, redirect, url_for
from shopify import Session as ShopifySession
from dotenv import load_dotenv
import os
import logging
import traceback

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', '37o3N6yz]$=a')

API_KEY = os.getenv('API_KEY', '94bd6d0489b4468e9e4d8e26afca790e')
API_SECRET = os.getenv('API_SECRET', 'b52b3dca3c62631ff2dec6d28f04c125')
API_VERSION = '2023-07'
SCOPE = 'read_orders,read_products,read_customers'
REDIRECT_URI = 'https://fa1d-2806-2a0-b12-84ce-a500-a468-fc24-9c2d.ngrok-free.app/auth/callback'  # Update with your current ngrok URL

logging.basicConfig(level=logging.DEBUG)

@app.route('/auth')
def auth():
    shop = request.args.get('shop')
    logging.debug(f"Received shop: {shop}")
    if not shop or not shop.endswith('.myshopify.com'):
        return "Invalid shop domain", 400
    auth_url = (
        f"https://{shop}/admin/oauth/authorize?client_id={API_KEY}&"
        f"scope={SCOPE}&redirect_uri={REDIRECT_URI}&response_type=code"
    )
    logging.debug(f"Generated auth URL: {auth_url}")
    return redirect(auth_url)

@app.route('/auth/callback')
def callback():
    shop = request.args.get('shop')
    code = request.args.get('code')
    hmac = request.args.get('hmac')
    timestamp = request.args.get('timestamp')

    logging.debug(f"Callback params: shop={shop}, code={code}, hmac={hmac}, timestamp={timestamp}")

    if not shop or not code:
        return "Missing shop or code", 400

    session = ShopifySession(shop, API_VERSION)

    try:
        token = session.request_token({
            "code": code,
            "hmac": hmac,
            "shop": shop,
            "timestamp": timestamp
        })
    except Exception as e:
        logging.exception("Error during request_token:")
        return f"Failed to get access token: {e}", 500

    flask_session['access_token'] = token
    flask_session['shop_url'] = shop
    return redirect(url_for('dashboard'))



@app.route('/dashboard')
def dashboard():
    return "Dashboard (placeholder)", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
