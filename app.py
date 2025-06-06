import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1, x_for=1, x_port=1)

# Configure the database
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise RuntimeError("DATABASE_URL environment variable is required")
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Shopify API configuration
app.config["SHOPIFY_API_KEY"] = os.environ.get("SHOPIFY_API_KEY")
app.config["SHOPIFY_API_SECRET"] = os.environ.get("SHOPIFY_API_SECRET")
app.config["SHOPIFY_SCOPES"] = "read_customers,read_orders,read_products,read_analytics"

# Production SSL configuration
app.config["PREFERRED_URL_SCHEME"] = "https"
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Debug Shopify configuration
logging.info(f"Shopify API Key configured: {'Yes' if app.config['SHOPIFY_API_KEY'] else 'No'}")
logging.info(f"Shopify API Secret configured: {'Yes' if app.config['SHOPIFY_API_SECRET'] else 'No'}")
if app.config["SHOPIFY_API_KEY"]:
    logging.info(f"Shopify API Key (first 8 chars): {app.config['SHOPIFY_API_KEY'][:8]}...")
else:
    logging.error("SHOPIFY_API_KEY environment variable not found!")
app.config["SHOPIFY_SCOPES"] = "read_orders,read_customers,read_analytics"

# Initialize the app with the extension
db.init_app(app)

with app.app_context():
    # Import models and routes
    import models
    import routes
    
    # Create all tables
    db.create_all()
    
    logging.info("Database tables created successfully")
