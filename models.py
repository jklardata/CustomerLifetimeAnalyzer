from app import db
from sqlalchemy import JSON
from datetime import datetime
import json

class ShopifyStore(db.Model):
    __tablename__ = 'shopify_stores'
    
    id = db.Column(db.Integer, primary_key=True)
    shop_domain = db.Column(db.String(255), unique=True, nullable=False)
    access_token = db.Column(db.String(255), nullable=False)
    shop_id = db.Column(db.String(100), nullable=False)
    shop_name = db.Column(db.String(255))
    email = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    customers = db.relationship('Customer', backref='store', lazy=True, cascade='all, delete-orphan')
    orders = db.relationship('Order', backref='store', lazy=True, cascade='all, delete-orphan')

class Customer(db.Model):
    __tablename__ = 'customers'
    
    id = db.Column(db.Integer, primary_key=True)
    shopify_customer_id = db.Column(db.String(100), nullable=False)
    store_id = db.Column(db.Integer, db.ForeignKey('shopify_stores.id'), nullable=False)
    email = db.Column(db.String(255))
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    total_spent = db.Column(db.Numeric(10, 2), default=0)
    orders_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # CLV metrics
    predicted_clv = db.Column(db.Numeric(10, 2))
    purchase_frequency = db.Column(db.Float)
    avg_order_value = db.Column(db.Numeric(10, 2))
    customer_lifespan = db.Column(db.Float)
    return_rate = db.Column(db.Float, default=0.0)
    
    # Raw Shopify data
    shopify_data = db.Column(JSON)
    
    # Relationships
    orders = db.relationship('Order', backref='customer', lazy=True)
    
    __table_args__ = (db.UniqueConstraint('shopify_customer_id', 'store_id'),)

class Order(db.Model):
    __tablename__ = 'orders'
    
    id = db.Column(db.Integer, primary_key=True)
    shopify_order_id = db.Column(db.String(100), nullable=False)
    store_id = db.Column(db.Integer, db.ForeignKey('shopify_stores.id'), nullable=False)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'))
    
    # Order details
    order_number = db.Column(db.String(100))
    total_price = db.Column(db.Numeric(10, 2))
    subtotal_price = db.Column(db.Numeric(10, 2))
    total_tax = db.Column(db.Numeric(10, 2))
    currency = db.Column(db.String(10))
    financial_status = db.Column(db.String(50))
    fulfillment_status = db.Column(db.String(50))
    
    # Timestamps
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime)
    processed_at = db.Column(db.DateTime)
    
    # Return tracking
    is_returned = db.Column(db.Boolean, default=False)
    return_reason = db.Column(db.String(255))
    
    # Raw Shopify data
    shopify_data = db.Column(JSON)
    
    __table_args__ = (db.UniqueConstraint('shopify_order_id', 'store_id'),)

class CLVPrediction(db.Model):
    __tablename__ = 'clv_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    store_id = db.Column(db.Integer, db.ForeignKey('shopify_stores.id'), nullable=False)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'), nullable=False)
    
    predicted_clv = db.Column(db.Numeric(10, 2))
    confidence_score = db.Column(db.Float)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Features used for prediction
    features = db.Column(JSON)
    model_version = db.Column(db.String(50))
    
    store = db.relationship('ShopifyStore', backref='clv_predictions')
    customer = db.relationship('Customer', backref='clv_predictions')
