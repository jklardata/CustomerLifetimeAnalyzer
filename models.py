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
    orders = db.relationship('Order', backref='store', lazy=True, cascade='all, delete-orphan')

# Customer model removed - using orders-based CLV analysis to avoid protected customer data requirements

class Order(db.Model):
    __tablename__ = 'orders'
    
    id = db.Column(db.Integer, primary_key=True)
    shopify_order_id = db.Column(db.String(100), nullable=False)
    store_id = db.Column(db.Integer, db.ForeignKey('shopify_stores.id'), nullable=False)
    
    # Anonymous customer grouping (hashed identifier, no personal data)
    customer_hash = db.Column(db.String(64))  # SHA256 hash for grouping orders
    
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
    
    # CLV Analysis fields (no personal data)
    order_sequence = db.Column(db.Integer)  # 1st, 2nd, 3rd order for this customer
    days_since_first_order = db.Column(db.Integer)
    
    # Raw Shopify data (anonymized)
    shopify_data = db.Column(JSON)
    
    __table_args__ = (db.UniqueConstraint('shopify_order_id', 'store_id'),)

class Product(db.Model):
    __tablename__ = 'products'
    
    id = db.Column(db.Integer, primary_key=True)
    store_id = db.Column(db.Integer, db.ForeignKey('shopify_stores.id'), nullable=False)
    shopify_product_id = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    vendor = db.Column(db.String(255))
    product_type = db.Column(db.String(255))
    price = db.Column(db.Numeric(10, 2))
    category = db.Column(db.String(100))
    inventory_quantity = db.Column(db.Integer, default=0)
    
    # CLV metrics at product level
    avg_clv_contribution = db.Column(db.Numeric(10, 2))
    total_sales = db.Column(db.Numeric(10, 2), default=0)
    units_sold = db.Column(db.Integer, default=0)
    return_rate = db.Column(db.Float, default=0.0)
    predicted_clv_impact = db.Column(db.Numeric(10, 2))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Raw Shopify data
    shopify_data = db.Column(JSON)
    
    # Relationships
    line_items = db.relationship('OrderLineItem', backref='product', lazy=True)
    store = db.relationship('ShopifyStore', backref='products')
    
    __table_args__ = (db.UniqueConstraint('shopify_product_id', 'store_id'),)

class OrderLineItem(db.Model):
    __tablename__ = 'order_line_items'
    
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'))
    store_id = db.Column(db.Integer, db.ForeignKey('shopify_stores.id'), nullable=False)
    
    # Line item details
    variant_title = db.Column(db.String(255))
    quantity = db.Column(db.Integer, default=1)
    price = db.Column(db.Numeric(10, 2))
    total_discount = db.Column(db.Numeric(10, 2), default=0)
    
    # Return tracking
    is_returned = db.Column(db.Boolean, default=False)
    return_reason = db.Column(db.String(255))
    
    # Raw data
    shopify_data = db.Column(JSON)
    
    # Relationships
    order = db.relationship('Order', backref='line_items')

class AbandonedCart(db.Model):
    __tablename__ = 'abandoned_carts'
    
    id = db.Column(db.Integer, primary_key=True)
    store_id = db.Column(db.Integer, db.ForeignKey('shopify_stores.id'), nullable=False)
    customer_id = db.Column(db.Integer, db.ForeignKey('customers.id'))
    shopify_checkout_id = db.Column(db.String(100), nullable=False)
    
    # Cart details
    email = db.Column(db.String(255))
    total_price = db.Column(db.Numeric(10, 2))
    currency = db.Column(db.String(10))
    line_items_count = db.Column(db.Integer, default=0)
    
    # Recovery prediction
    recovery_probability = db.Column(db.Float)
    predicted_clv_value = db.Column(db.Numeric(10, 2))
    recovery_email_sent = db.Column(db.Boolean, default=False)
    recovered = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    abandoned_at = db.Column(db.DateTime)
    
    # Raw Shopify data
    shopify_data = db.Column(JSON)
    
    # Relationships
    store = db.relationship('ShopifyStore', backref='abandoned_carts')
    customer = db.relationship('Customer', backref='abandoned_carts')
    cart_line_items = db.relationship('AbandonedCartLineItem', backref='cart', lazy=True, cascade='all, delete-orphan')
    
    __table_args__ = (db.UniqueConstraint('shopify_checkout_id', 'store_id'),)

class AbandonedCartLineItem(db.Model):
    __tablename__ = 'abandoned_cart_line_items'
    
    id = db.Column(db.Integer, primary_key=True)
    cart_id = db.Column(db.Integer, db.ForeignKey('abandoned_carts.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'))
    
    # Line item details
    variant_title = db.Column(db.String(255))
    quantity = db.Column(db.Integer, default=1)
    price = db.Column(db.Numeric(10, 2))
    title = db.Column(db.String(255))
    
    # Raw data
    shopify_data = db.Column(JSON)
    
    # Relationships
    product = db.relationship('Product', backref='cart_line_items')

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
