from flask import render_template, request, redirect, url_for, session, flash, jsonify
from app import app, db
from models import ShopifyStore, Customer, Order, CLVPrediction
from shopify_client import ShopifyClient
from clv_calculator import CLVCalculator
import logging
import traceback
from urllib.parse import urlencode
import secrets

@app.route('/')
def index():
    """Landing page inspired by klardata.com design"""
    return render_template('index.html')

@app.route('/auth')
def auth():
    """Shopify OAuth initiation"""
    shop = request.args.get('shop')
    if not shop:
        return render_template('auth.html')
    
    # Validate shop domain
    if not shop.endswith('.myshopify.com'):
        shop = f"{shop}.myshopify.com"
    
    # Generate state for security
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state
    session['shop'] = shop
    
    # Build OAuth URL
    params = {
        'client_id': app.config['SHOPIFY_API_KEY'],
        'scope': app.config['SHOPIFY_SCOPES'],
        'redirect_uri': url_for('callback', _external=True),
        'state': state
    }
    
    oauth_url = f"https://{shop}/admin/oauth/authorize?{urlencode(params)}"
    return redirect(oauth_url)

@app.route('/callback')
def callback():
    """Shopify OAuth callback"""
    try:
        # Verify state parameter
        if request.args.get('state') != session.get('oauth_state'):
            flash('Invalid OAuth state. Please try again.', 'error')
            return redirect(url_for('index'))
        
        shop = session.get('shop')
        code = request.args.get('code')
        
        if not shop or not code:
            flash('Missing required OAuth parameters.', 'error')
            return redirect(url_for('index'))
        
        # Exchange code for access token
        shopify_client = ShopifyClient(app.config['SHOPIFY_API_KEY'], app.config['SHOPIFY_API_SECRET'])
        access_token = shopify_client.get_access_token(shop, code)
        
        if not access_token:
            flash('Failed to obtain access token from Shopify.', 'error')
            return redirect(url_for('index'))
        
        # Get shop information
        shopify_client.set_access_token(access_token)
        shop_info = shopify_client.get_shop_info(shop)
        
        if not shop_info:
            flash('Failed to retrieve shop information.', 'error')
            return redirect(url_for('index'))
        
        # Save or update store in database
        store = ShopifyStore.query.filter_by(shop_domain=shop).first()
        if not store:
            store = ShopifyStore(
                shop_domain=shop,
                access_token=access_token,
                shop_id=str(shop_info.get('id')),
                shop_name=shop_info.get('name'),
                email=shop_info.get('email')
            )
            db.session.add(store)
        else:
            store.access_token = access_token
            store.shop_name = shop_info.get('name')
            store.email = shop_info.get('email')
        
        db.session.commit()
        
        # Store in session
        session['store_id'] = store.id
        session['shop_domain'] = shop
        
        # Clean up OAuth session data
        session.pop('oauth_state', None)
        session.pop('shop', None)
        
        flash('Successfully connected to Shopify!', 'success')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        logging.error(f"OAuth callback error: {str(e)}")
        logging.error(traceback.format_exc())
        flash('An error occurred during authentication. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Main dashboard with CLV analytics"""
    store_id = session.get('store_id')
    if not store_id:
        flash('Please authenticate with Shopify first.', 'warning')
        return redirect(url_for('index'))
    
    try:
        store = ShopifyStore.query.get(store_id)
        if not store:
            flash('Store not found. Please re-authenticate.', 'error')
            return redirect(url_for('logout'))
        
        # Get dashboard metrics
        metrics = get_dashboard_metrics(store)
        
        return render_template('dashboard.html', store=store, metrics=metrics)
        
    except Exception as e:
        logging.error(f"Dashboard error: {str(e)}")
        logging.error(traceback.format_exc())
        flash('An error occurred loading the dashboard.', 'error')
        return redirect(url_for('index'))

@app.route('/sync-data')
def sync_data():
    """Sync data from Shopify API"""
    store_id = session.get('store_id')
    if not store_id:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        store = ShopifyStore.query.get(store_id)
        if not store:
            return jsonify({'error': 'Store not found'}), 404
        
        # Initialize Shopify client
        shopify_client = ShopifyClient(app.config['SHOPIFY_API_KEY'], app.config['SHOPIFY_API_SECRET'])
        shopify_client.set_access_token(store.access_token)
        
        # Sync customers and orders
        customers_synced = sync_customers(shopify_client, store)
        orders_synced = sync_orders(shopify_client, store)
        
        # Calculate CLV for all customers
        clv_calculator = CLVCalculator()
        clv_updates = clv_calculator.calculate_store_clv(store)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'customers_synced': customers_synced,
            'orders_synced': orders_synced,
            'clv_updates': clv_updates
        })
        
    except Exception as e:
        logging.error(f"Data sync error: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({'error': 'Failed to sync data'}), 500

@app.route('/logout')
def logout():
    """Clear session and logout"""
    session.clear()
    flash('Successfully logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': str(datetime.utcnow())})

def get_dashboard_metrics(store):
    """Calculate dashboard metrics for a store"""
    try:
        # Basic counts
        total_customers = Customer.query.filter_by(store_id=store.id).count()
        total_orders = Order.query.filter_by(store_id=store.id).count()
        
        # Revenue metrics
        total_revenue = db.session.query(db.func.sum(Order.total_price)).filter_by(store_id=store.id).scalar() or 0
        avg_order_value = db.session.query(db.func.avg(Order.total_price)).filter_by(store_id=store.id).scalar() or 0
        
        # CLV metrics
        avg_clv = db.session.query(db.func.avg(Customer.predicted_clv)).filter_by(store_id=store.id).scalar() or 0
        total_clv = db.session.query(db.func.sum(Customer.predicted_clv)).filter_by(store_id=store.id).scalar() or 0
        
        # Return rate
        total_returns = Order.query.filter_by(store_id=store.id, is_returned=True).count()
        return_rate = (total_returns / total_orders * 100) if total_orders > 0 else 0
        
        # Top customers by CLV
        top_customers = Customer.query.filter_by(store_id=store.id)\
            .filter(Customer.predicted_clv.isnot(None))\
            .order_by(Customer.predicted_clv.desc())\
            .limit(10).all()
        
        # Recent orders
        recent_orders = Order.query.filter_by(store_id=store.id)\
            .order_by(Order.created_at.desc())\
            .limit(10).all()
        
        return {
            'total_customers': total_customers,
            'total_orders': total_orders,
            'total_revenue': float(total_revenue),
            'avg_order_value': float(avg_order_value),
            'avg_clv': float(avg_clv),
            'total_clv': float(total_clv),
            'return_rate': round(return_rate, 2),
            'top_customers': top_customers,
            'recent_orders': recent_orders
        }
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        return {
            'total_customers': 0,
            'total_orders': 0,
            'total_revenue': 0,
            'avg_order_value': 0,
            'avg_clv': 0,
            'total_clv': 0,
            'return_rate': 0,
            'top_customers': [],
            'recent_orders': []
        }

def sync_customers(shopify_client, store):
    """Sync customers from Shopify"""
    try:
        customers_data = shopify_client.get_customers(store.shop_domain)
        customers_synced = 0
        
        for customer_data in customers_data:
            customer = Customer.query.filter_by(
                shopify_customer_id=str(customer_data['id']),
                store_id=store.id
            ).first()
            
            if not customer:
                customer = Customer(
                    shopify_customer_id=str(customer_data['id']),
                    store_id=store.id
                )
                db.session.add(customer)
            
            # Update customer data
            customer.email = customer_data.get('email')
            customer.first_name = customer_data.get('first_name')
            customer.last_name = customer_data.get('last_name')
            customer.total_spent = float(customer_data.get('total_spent', 0))
            customer.orders_count = customer_data.get('orders_count', 0)
            customer.shopify_data = customer_data
            
            customers_synced += 1
        
        return customers_synced
        
    except Exception as e:
        logging.error(f"Error syncing customers: {str(e)}")
        return 0

def sync_orders(shopify_client, store):
    """Sync orders from Shopify"""
    try:
        orders_data = shopify_client.get_orders(store.shop_domain)
        orders_synced = 0
        
        for order_data in orders_data:
            order = Order.query.filter_by(
                shopify_order_id=str(order_data['id']),
                store_id=store.id
            ).first()
            
            if not order:
                order = Order(
                    shopify_order_id=str(order_data['id']),
                    store_id=store.id
                )
                db.session.add(order)
            
            # Find customer
            if order_data.get('customer'):
                customer = Customer.query.filter_by(
                    shopify_customer_id=str(order_data['customer']['id']),
                    store_id=store.id
                ).first()
                if customer:
                    order.customer_id = customer.id
            
            # Update order data
            order.order_number = order_data.get('order_number')
            order.total_price = float(order_data.get('total_price', 0))
            order.subtotal_price = float(order_data.get('subtotal_price', 0))
            order.total_tax = float(order_data.get('total_tax', 0))
            order.currency = order_data.get('currency')
            order.financial_status = order_data.get('financial_status')
            order.fulfillment_status = order_data.get('fulfillment_status')
            order.shopify_data = order_data
            
            # Parse timestamps
            if order_data.get('created_at'):
                order.created_at = shopify_client.parse_datetime(order_data['created_at'])
            if order_data.get('updated_at'):
                order.updated_at = shopify_client.parse_datetime(order_data['updated_at'])
            if order_data.get('processed_at'):
                order.processed_at = shopify_client.parse_datetime(order_data['processed_at'])
            
            orders_synced += 1
        
        return orders_synced
        
    except Exception as e:
        logging.error(f"Error syncing orders: {str(e)}")
        return 0
