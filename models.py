from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import JSON

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

class Shop(db.Model):
    """Represents a Shopify store."""
    id = db.Column(db.Integer, primary_key=True)
    shop_name = db.Column(db.String(255), unique=True, nullable=False)
    shop_email = db.Column(db.String(255))
    access_token = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_sync = db.Column(db.DateTime)

    def __repr__(self):
        return f'<Shop {self.shop_name}>'

class Customer(db.Model):
    """Represents a customer with their lifetime value metrics."""
    id = db.Column(db.Integer, primary_key=True)
    shop_id = db.Column(db.Integer, db.ForeignKey('shop.id'), nullable=False)
    shopify_customer_id = db.Column(db.BigInteger, unique=True, nullable=False)
    email = db.Column(db.String(255))
    first_name = db.Column(db.String(255))
    last_name = db.Column(db.String(255))
    total_spent = db.Column(db.Float, default=0.0)
    orders_count = db.Column(db.Integer, default=0)
    last_order_date = db.Column(db.DateTime)
    average_order_value = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    shop = db.relationship('Shop', backref=db.backref('customers', lazy=True))

    def __repr__(self):
        return f'<Customer {self.email}>'

class Order(db.Model):
    """Represents a customer order."""
    id = db.Column(db.Integer, primary_key=True)
    shop_id = db.Column(db.Integer, db.ForeignKey('shop.id'), nullable=False)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    shopify_order_id = db.Column(db.BigInteger, unique=True, nullable=False)
    order_number = db.Column(db.String(50))
    total_price = db.Column(db.Float, nullable=False)
    order_date = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    shop = db.relationship('Shop', backref=db.backref('orders', lazy=True))
    customer = db.relationship('Customer', backref=db.backref('orders', lazy=True))

    def __repr__(self):
        return f'<Order {self.order_number}>'

class CLVPrediction(db.Model):
    """Represents a customer lifetime value prediction."""
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    predicted_value = db.Column(db.Float, nullable=False)
    confidence_score = db.Column(db.Float)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    features = db.Column(JSON)
    
    # Relationships
    customer = db.relationship('Customer', backref=db.backref('predictions', lazy=True))

    def __repr__(self):
        return f'<CLVPrediction {self.customer_id}>'
