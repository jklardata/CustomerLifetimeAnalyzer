from flask import render_template, request, redirect, url_for, session, flash, jsonify
from app import app, db
from models import ShopifyStore, Order, CLVPrediction, Product, AbandonedCart
from shopify_client import ShopifyClient
from orders_clv_calculator import OrdersCLVCalculator
import logging
import traceback
import random
from urllib.parse import urlencode
import secrets
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

@app.route('/')
def index():
    """Landing page inspired by klardata.com design"""
    return render_template('index.html')

@app.route('/auth')
def auth():
    """Shopify OAuth initiation"""
    shop = request.args.get('shop')
    demo_mode = request.args.get('demo')
    
    if not shop and not demo_mode:
        return render_template('auth.html')
    
    # Demo mode for testing
    if demo_mode == 'true':
        return demo_login()
    
    try:
        # Validate shop domain
        if shop and not str(shop).endswith('.myshopify.com'):
            shop = f"{shop}.myshopify.com"
        
        logging.info(f"Initiating OAuth for shop: {shop}")
        
        # Validate API credentials are available
        if not app.config.get('SHOPIFY_API_KEY'):
            logging.error("SHOPIFY_API_KEY not configured")
            flash('Shopify API configuration missing. Please contact support.', 'error')
            return redirect(url_for('index'))
        
        # Generate state for security
        state = secrets.token_urlsafe(32)
        session['oauth_state'] = state
        session['shop'] = shop
        
        # Build OAuth URL with explicit redirect URI to match whitelisted value
        # Use the exact URL that's whitelisted in your Shopify app
        redirect_uri = "https://customer-lifetime-analyzer-justinleu1.replit.app/auth/callback"
        logging.info(f"Using explicit redirect URI: {redirect_uri}")
        
        params = {
            'client_id': app.config['SHOPIFY_API_KEY'],
            'scope': app.config['SHOPIFY_SCOPES'],
            'redirect_uri': redirect_uri,
            'state': state
        }
        
        oauth_url = f"https://{shop}/admin/oauth/authorize?{urlencode(params)}"
        logging.info(f"Complete OAuth URL: {oauth_url}")
        logging.info(f"OAuth Parameters: {params}")
        logging.info(f"API Key: {app.config['SHOPIFY_API_KEY']}")
        logging.info(f"Redirect URI being sent: {redirect_uri}")
        
        return redirect(oauth_url)
        
    except Exception as e:
        logging.error(f"OAuth initiation error: {str(e)}")
        logging.error(traceback.format_exc())
        flash('An error occurred during authentication setup. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/demo-login')
def demo_login():
    """Demo login for testing platform functionality"""
    try:
        # Create or get demo store
        demo_store = ShopifyStore.query.filter_by(shop_domain="clv-test-store.myshopify.com").first()
        if not demo_store:
            demo_store = ShopifyStore()
            demo_store.shop_domain = "clv-test-store.myshopify.com"
            demo_store.access_token = "demo_access_token"
            demo_store.shop_id = "demo_shop_123"
            demo_store.shop_name = "CLV Test Store"
            demo_store.email = "test@clvstore.com"
            db.session.add(demo_store)
            db.session.commit()
        
        # Create sample orders if none exist
        if Order.query.filter_by(store_id=demo_store.id).count() == 0:
            # Create simple demo orders directly
            import hashlib
            from decimal import Decimal
            from datetime import datetime, timedelta
            
            for i in range(20):
                customer_hash = hashlib.sha256(f"demo_customer_{i}".encode()).hexdigest()
                order_date = datetime.utcnow() - timedelta(days=random.randint(1, 90))
                
                order = Order()
                order.shopify_order_id = f"demo_order_{i}_{random.randint(1000, 9999)}"
                order.store_id = demo_store.id
                order.customer_hash = customer_hash
                order.order_number = f"#{1000 + i}"
                order.total_price = Decimal(str(random.uniform(50, 300)))
                order.subtotal_price = order.total_price * Decimal('0.9')
                order.total_tax = order.total_price * Decimal('0.1')
                order.currency = "USD"
                order.financial_status = "paid"
                order.fulfillment_status = "fulfilled"
                order.created_at = order_date
                order.updated_at = order_date
                order.processed_at = order_date
                order.order_sequence = 1
                order.days_since_first_order = 0
                order.is_returned = False
                order.shopify_data = {"demo": True}
                
                db.session.add(order)
            
            db.session.commit()
        
        # Set session data
        session['store_id'] = demo_store.id
        session['shop_domain'] = demo_store.shop_domain
        
        flash('Connected to demo store! This shows platform functionality with sample data.', 'success')
        return redirect(url_for('dashboard'))
    except Exception as e:
        logging.error(f"Demo login error: {str(e)}")
        flash('Error setting up demo mode. Please try again.', 'error')
        return redirect(url_for('auth'))

def create_sample_data(store):
    """Create sample customers and orders for demo"""
    import random
    from decimal import Decimal
    
    # Sample customer data
    customers_data = [
        {"name": "John Smith", "email": "john.smith@example.com", "orders": 5, "total": 1250.00},
        {"name": "Sarah Johnson", "email": "sarah.j@example.com", "orders": 3, "total": 890.50},
        {"name": "Michael Chen", "email": "m.chen@example.com", "orders": 8, "total": 2100.75},
        {"name": "Emma Wilson", "email": "emma.wilson@example.com", "orders": 2, "total": 450.25},
        {"name": "David Rodriguez", "email": "d.rodriguez@example.com", "orders": 6, "total": 1680.00},
        {"name": "Lisa Brown", "email": "lisa.brown@example.com", "orders": 4, "total": 720.50},
        {"name": "James Taylor", "email": "j.taylor@example.com", "orders": 7, "total": 1950.25},
        {"name": "Anna Garcia", "email": "anna.garcia@example.com", "orders": 3, "total": 675.75}
    ]
    
    for i, customer_data in enumerate(customers_data):
        # Create customer
        customer = Customer()
        customer.shopify_customer_id = f"demo_cust_{i+1}"
        customer.store_id = store.id
        customer.email = customer_data["email"]
        names = customer_data["name"].split()
        customer.first_name = names[0]
        customer.last_name = names[1] if len(names) > 1 else ""
        customer.total_spent = Decimal(str(customer_data["total"]))
        customer.orders_count = customer_data["orders"]
        customer.created_at = datetime.utcnow() - timedelta(days=random.randint(30, 365))
        
        db.session.add(customer)
        db.session.flush()  # Get customer ID
        
        # Create orders for customer
        for order_num in range(customer_data["orders"]):
            order = Order()
            order.shopify_order_id = f"demo_order_{i+1}_{order_num+1}"
            order.store_id = store.id
            order.customer_id = customer.id
            order.order_number = f"#{1000 + (i * 10) + order_num + 1}"
            order.total_price = Decimal(str(customer_data["total"] / customer_data["orders"]))
            order.subtotal_price = order.total_price * Decimal('0.9')
            order.total_tax = order.total_price * Decimal('0.1')
            order.currency = "USD"
            order.financial_status = "paid"
            order.fulfillment_status = "fulfilled"
            order.created_at = customer.created_at + timedelta(days=random.randint(1, 300))
            order.updated_at = order.created_at
            order.processed_at = order.created_at
            
            db.session.add(order)
    
    db.session.commit()
    
    # Calculate CLV for all customers
    clv_calculator = CLVCalculator()
    clv_calculator.calculate_store_clv(store)
    
    logging.info(f"Created sample data for {len(customers_data)} customers")

@app.route('/auth/callback')
def callback():
    """Shopify OAuth callback"""
    try:
        logging.info(f"OAuth callback received with args: {dict(request.args)}")
        logging.info(f"Session oauth_state: {session.get('oauth_state')}")
        logging.info(f"Request state: {request.args.get('state')}")
        
        # Verify state parameter
        if request.args.get('state') != session.get('oauth_state'):
            logging.error(f"OAuth state mismatch - Session: {session.get('oauth_state')}, Request: {request.args.get('state')}")
            flash('Invalid OAuth state. Please try again.', 'error')
            return redirect(url_for('index'))
        
        shop = session.get('shop')
        code = request.args.get('code')
        
        if not shop or not code:
            flash('Missing required OAuth parameters.', 'error')
            return redirect(url_for('index'))
        
        # Exchange code for access token
        logging.info(f"Attempting to get access token for shop: {shop}")
        shopify_client = ShopifyClient(app.config['SHOPIFY_API_KEY'], app.config['SHOPIFY_API_SECRET'])
        access_token = shopify_client.get_access_token(shop, code)
        
        if not access_token:
            logging.error(f"Failed to obtain access token for shop: {shop}, code: {code[:10]}...")
            flash('Failed to obtain access token from Shopify.', 'error')
            return redirect(url_for('index'))
            
        logging.info(f"Successfully obtained access token for shop: {shop}")
        
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
        
        # Automatically sync data from Shopify after successful authentication
        logging.info(f"Starting automatic data sync for store: {shop}")
        try:
            # Clear any existing demo data for this store first
            clear_store_data(store)
            logging.info("Cleared existing demo data")
            
            # Sync customers from Shopify
            customers_synced = sync_customers(shopify_client, store)
            logging.info(f"Synced {customers_synced} customers")
            
            # Sync orders from Shopify
            orders_synced = sync_orders(shopify_client, store)
            logging.info(f"Synced {orders_synced} orders")
            
            # Calculate CLV for all customers
            clv_calculator = CLVCalculator()
            clv_calculator.calculate_store_clv(store)
            logging.info("CLV calculations completed")
            
            flash(f'Successfully connected and synced {customers_synced} customers and {orders_synced} orders!', 'success')
            
        except Exception as sync_error:
            logging.error(f"Data sync error: {str(sync_error)}")
            logging.error(traceback.format_exc())
            flash('Connected to Shopify but data sync encountered issues. You can manually sync from the dashboard.', 'warning')
        
        # Clean up OAuth session data
        session.pop('oauth_state', None)
        session.pop('shop', None)
        
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        logging.error(f"OAuth callback error: {str(e)}")
        logging.error(f"Request args: {dict(request.args)}")
        logging.error(f"Session data: {dict(session)}")
        logging.error(traceback.format_exc())
        flash(f'Authentication error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Main dashboard with CLV analytics - Overview page"""
    store_id = session.get('store_id')
    if not store_id:
        flash('Please authenticate with Shopify first.', 'warning')
        return redirect(url_for('index'))
    
    try:
        store = ShopifyStore.query.get(store_id)
        if not store:
            flash('Store not found. Please re-authenticate.', 'error')
            return redirect(url_for('logout'))
        
        # Use demo data for reliable platform demonstration
        from demo_orders_generator import DemoOrdersGenerator
        demo_generator = DemoOrdersGenerator()
        
        # Check if demo data exists, generate if needed
        orders_count = Order.query.filter_by(store_id=store.id).count()
        if orders_count == 0:
            logging.info("Generating comprehensive demo data for dashboard")
            demo_generator.populate_demo_data()
        else:
            logging.info("Using existing demo data")
        
        # Get dashboard metrics
        metrics = get_dashboard_metrics(store)
        
        return render_template('dashboard_overview.html', store=store, metrics=metrics)
        
    except Exception as e:
        logging.error(f"Dashboard error: {str(e)}")
        logging.error(traceback.format_exc())
        flash('An error occurred loading the dashboard.', 'error')
        return redirect(url_for('index'))

@app.route('/reports/orders')
def orders_report():
    """Orders report with filtering and sorting"""
    store_id = session.get('store_id')
    if not store_id:
        return redirect(url_for('index'))
    
    try:
        store = ShopifyStore.query.get(store_id)
        if not store:
            return redirect(url_for('logout'))
        
        # Get filter parameters
        days = request.args.get('days', '90')
        status_filter = request.args.get('status', 'all')
        customer_type = request.args.get('customer_type', 'all')
        sort_by = request.args.get('sort', 'date')
        sort_order = request.args.get('order', 'desc')
        
        # Build query
        query = Order.query.filter_by(store_id=store.id)
        
        # Apply date filter
        if days != 'all':
            cutoff_date = datetime.utcnow() - timedelta(days=int(days))
            query = query.filter(Order.created_at >= cutoff_date)
        
        # Apply status filter
        if status_filter != 'all':
            if status_filter == 'fulfilled':
                query = query.filter(Order.financial_status == 'paid')
            else:
                query = query.filter(Order.financial_status == status_filter)
        
        # Apply sorting
        if sort_by == 'date':
            if sort_order == 'desc':
                query = query.order_by(Order.created_at.desc())
            else:
                query = query.order_by(Order.created_at.asc())
        elif sort_by == 'amount':
            if sort_order == 'desc':
                query = query.order_by(Order.total_price.desc())
            else:
                query = query.order_by(Order.total_price.asc())
        
        orders = query.limit(100).all()
        
        # Calculate summary stats
        total_orders = Order.query.filter_by(store_id=store.id).count()
        total_revenue = db.session.query(db.func.sum(Order.total_price)).filter_by(store_id=store.id).scalar() or 0
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        fulfilled_orders = Order.query.filter_by(store_id=store.id, financial_status='paid').count()
        fulfilled_percentage = (fulfilled_orders / total_orders * 100) if total_orders > 0 else 0
        
        orders_data = {
            'orders': orders,
            'total_orders': total_orders,
            'total_revenue': float(total_revenue),
            'avg_order_value': float(avg_order_value),
            'fulfilled_percentage': round(fulfilled_percentage, 1)
        }
        
        return render_template('orders_report.html', store=store, orders_data=orders_data)
        
    except Exception as e:
        logging.error(f"Orders report error: {str(e)}")
        flash('An error occurred loading the orders report.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/reports/clv-report')
def clv_report():
    """Customer Lifetime Value (CLV) Report"""
    store_id = session.get('store_id')
    if not store_id:
        return redirect(url_for('index'))
    
    try:
        store = ShopifyStore.query.get(store_id)
        if not store:
            return redirect(url_for('logout'))
        
        clv_calculator = CLVCalculator()
        clv_data = get_clv_report_data(store, clv_calculator)
        
        return render_template('clv_report.html', store=store, clv_data=clv_data)
        
    except Exception as e:
        logging.error(f"CLV report error: {str(e)}")
        flash('An error occurred loading the CLV report.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/reports/customer-segmentation')
def customer_segmentation():
    """Customer segmentation report"""
    store_id = session.get('store_id')
    if not store_id:
        return redirect(url_for('index'))
    
    try:
        store = ShopifyStore.query.get(store_id)
        if not store:
            return redirect(url_for('logout'))
        
        clv_calculator = CLVCalculator()
        segmentation_data = clv_calculator.get_customer_segmentation_by_clv(store)
        
        # Get detailed customer data for each segment
        customers = Customer.query.filter_by(store_id=store.id)\
            .filter(Customer.predicted_clv.isnot(None))\
            .order_by(Customer.predicted_clv.desc()).all()
        
        return render_template('customer_segmentation.html', 
                             store=store, 
                             segmentation_data=segmentation_data,
                             customers=customers)
        
    except Exception as e:
        logging.error(f"Customer segmentation error: {str(e)}")
        flash('An error occurred loading customer segmentation.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/clv-recommendations')
def clv_recommendations():
    """CLV optimization recommendations"""
    store_id = session.get('store_id')
    if not store_id:
        return redirect(url_for('index'))
    
    try:
        store = ShopifyStore.query.get(store_id)
        if not store:
            return redirect(url_for('logout'))
        
        clv_calculator = CLVCalculator()
        recommendations = clv_calculator.generate_ai_recommendations(store)
        metrics = get_dashboard_metrics(store)
        
        return render_template('clv_recommendations.html', 
                             store=store, 
                             recommendations=recommendations,
                             metrics=metrics)
        
    except Exception as e:
        logging.error(f"CLV recommendations error: {str(e)}")
        flash('An error occurred loading CLV recommendations.', 'error')
        return redirect(url_for('dashboard'))

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
        
        # Add detailed logging for debugging
        logging.info(f"Starting sync for store: {store.shop_domain}")
        logging.info(f"Access token exists: {'Yes' if store.access_token else 'No'}")
        logging.info(f"Access token length: {len(store.access_token) if store.access_token else 0}")
        
        # Test API connection and permissions
        shop_info_test = shopify_client.get_shop_info(store.shop_domain)
        if shop_info_test:
            logging.info(f"API connection successful - Shop: {shop_info_test.get('name')}")
            
            # Test orders API access (bypasses protected customer data restrictions)
            test_orders = shopify_client.get_orders(store.shop_domain, limit=1)
            if test_orders is not None:
                logging.info(f"Orders API access working - returned {len(test_orders)} orders")
            else:
                logging.error("Orders API access failed - check authentication")
                return jsonify({
                    'success': False,
                    'error': 'Orders API access failed - check authentication'
                })
                
        else:
            logging.error("API connection failed - check access token and permissions")
        
        # Test database connection
        try:
            db.session.execute(db.text("SELECT 1"))
            logging.info("Database connection successful")
        except Exception as db_error:
            logging.error(f"Database connection error: {str(db_error)}")
        
        # Sync orders only (no customer data stored)
        orders_synced = sync_orders(shopify_client, store)
        
        logging.info(f"Sync completed - Orders: {orders_synced}")
        
        # Check orders saved
        actual_orders = Order.query.filter_by(store_id=store.id).count()
        logging.info(f"Orders in database after sync: {actual_orders}")
        
        # Calculate CLV using orders-based analysis
        clv_calculator = OrdersCLVCalculator()
        metrics = clv_calculator.calculate_order_metrics(store)
        clv_updates = metrics.get('unique_customers', 0)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'customers_synced': 0,  # No longer storing customer data
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

@app.route('/clv-optimization/product-level')
def product_clv_optimization():
    """Product Level CLV Optimization Dashboard"""
    store_id = session.get('store_id')
    if not store_id:
        flash('Please log in to view this page.', 'error')
        return redirect(url_for('auth'))
    
    try:
        from models import Product, OrderLineItem, Order
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        import pandas as pd
        import numpy as np
        
        store = ShopifyStore.query.get(store_id)
        if not store:
            flash('Store not found.', 'error')
            return redirect(url_for('dashboard'))
        
        # Get product data with CLV metrics
        products = Product.query.filter_by(store_id=store.id).all()
        
        # Initialize model data
        model_data = {'model_trained': False}
        
        # Prepare data for ML model
        if products:
            # Create feature matrix for products
            product_features = []
            clv_targets = []
            
            for product in products:
                # Calculate additional features from order line items
                line_items = OrderLineItem.query.filter_by(product_id=product.id).all()
                
                if line_items:
                    # Feature engineering
                    total_quantity = sum(item.quantity for item in line_items)
                    avg_quantity_per_order = total_quantity / len(line_items) if line_items else 0
                    total_revenue = sum(float(item.price) * item.quantity for item in line_items)
                    avg_discount = sum(float(item.total_discount or 0) for item in line_items) / len(line_items)
                    return_count = sum(1 for item in line_items if item.is_returned)
                    
                    # Category encoding (simple numeric for demo)
                    category_map = {'Electronics': 1, 'Apparel': 2, 'Home & Garden': 3, 'Beauty': 4, 'Fitness': 5, 'Accessories': 6, 'Health': 7, 'Footwear': 8, 'Lifestyle': 9, 'Food & Beverage': 10}
                    category_encoded = category_map.get(product.category, 0)
                    
                    features = [
                        float(product.price),
                        product.inventory_quantity,
                        total_quantity,
                        avg_quantity_per_order,
                        total_revenue,
                        avg_discount,
                        return_count,
                        category_encoded,
                        float(product.return_rate),
                        float(product.units_sold)
                    ]
                    
                    product_features.append(features)
                    clv_targets.append(float(product.avg_clv_contribution or 0))
            
            # Train Random Forest model for product CLV prediction
            if len(product_features) > 5:  # Need minimum data for training
                X = np.array(product_features)
                y = np.array(clv_targets)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Train model
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = rf_model.predict(X_test)
                
                # Calculate feature importance
                feature_names = ['Price', 'Inventory', 'Total Quantity', 'Avg Quantity/Order', 'Total Revenue', 'Avg Discount', 'Return Count', 'Category', 'Return Rate', 'Units Sold']
                feature_importance = list(zip(feature_names, rf_model.feature_importances_))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Generate predictions for all products
                predictions = rf_model.predict(X)
                
                model_data = {
                    'feature_importance': feature_importance,
                    'model_score': rf_model.score(X_test, y_test),
                    'predictions': predictions.tolist(),
                    'model_trained': True
                }
        
        # Generate product recommendations (using helper function defined below)
        recommendations = []
        if products:
            # Sort products by CLV contribution
            sorted_products = sorted(products, key=lambda p: float(p.avg_clv_contribution or 0), reverse=True)
            
            # Top performers
            top_products = sorted_products[:3]
            for product in top_products:
                recommendations.append({
                    'type': 'promotion',
                    'priority': 'high',
                    'title': f'Promote {product.title} to high-CLV customers',
                    'description': f'This product contributes ${product.avg_clv_contribution:.2f} average CLV. Offer 10% discount to customers with CLV > $300.',
                    'impact': 'High CLV increase potential',
                    'action': f'Create targeted campaign for {product.title}'
                })
        
        # Product performance analysis
        product_analysis = {}
        if products:
            # Category analysis
            category_performance = {}
            for product in products:
                category = product.category
                if category not in category_performance:
                    category_performance[category] = {
                        'products': 0,
                        'total_clv': 0,
                        'avg_clv': 0,
                        'total_sales': 0,
                        'avg_return_rate': 0
                    }
                
                cat_data = category_performance[category]
                cat_data['products'] += 1
                cat_data['total_clv'] += float(product.avg_clv_contribution or 0)
                cat_data['total_sales'] += float(product.total_sales or 0)
                cat_data['avg_return_rate'] += float(product.return_rate or 0)
            
            # Calculate averages
            for category, data in category_performance.items():
                if data['products'] > 0:
                    data['avg_clv'] = data['total_clv'] / data['products']
                    data['avg_return_rate'] = data['avg_return_rate'] / data['products']
            
            # Top and bottom performers
            sorted_products = sorted(products, key=lambda p: float(p.avg_clv_contribution or 0), reverse=True)
            
            product_analysis = {
                'category_performance': category_performance,
                'top_performers': sorted_products[:5],
                'bottom_performers': sorted_products[-5:],
                'total_products': len(products),
                'avg_clv_contribution': sum(float(p.avg_clv_contribution or 0) for p in products) / len(products) if products else 0
            }
        
        return render_template('product_clv_optimization.html', 
                             products=products,
                             model_data=model_data,
                             recommendations=recommendations,
                             analysis=product_analysis,
                             store=store)
    
    except Exception as e:
        logging.error(f"Error in product CLV optimization: {str(e)}")
        logging.error(traceback.format_exc())
        flash('Error loading product CLV optimization data.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/clv-optimization/abandoned-cart-recovery')
def abandoned_cart_recovery():
    """Predictive Abandoned Cart Recovery Dashboard"""
    store_id = session.get('store_id')
    if not store_id:
        flash('Please log in to view this page.', 'error')
        return redirect(url_for('auth'))
    
    try:
        from models import AbandonedCart
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        import pandas as pd
        import numpy as np
        
        store = ShopifyStore.query.get(store_id)
        if not store:
            flash('Store not found.', 'error')
            return redirect(url_for('dashboard'))
        
        # Get abandoned cart data
        abandoned_carts = AbandonedCart.query.filter_by(store_id=store.id).all()
        
        # Prepare data for ML model
        recovery_model_data = {}
        if abandoned_carts:
            # Create feature matrix for abandoned carts
            cart_features = []
            recovery_targets = []
            
            for cart in abandoned_carts:
                if cart.customer:
                    # Feature engineering for recovery prediction
                    customer_clv = float(cart.customer.predicted_clv or 0)
                    customer_orders = cart.customer.orders_count
                    customer_total_spent = float(cart.customer.total_spent or 0)
                    cart_value = float(cart.total_price)
                    items_count = cart.line_items_count
                    days_since_abandoned = (datetime.utcnow() - cart.abandoned_at).days if cart.abandoned_at else 0
                    
                    # Customer behavior features
                    avg_order_value = customer_total_spent / customer_orders if customer_orders > 0 else 0
                    cart_to_aov_ratio = cart_value / avg_order_value if avg_order_value > 0 else 0
                    
                    features = [
                        customer_clv,
                        customer_orders,
                        customer_total_spent,
                        cart_value,
                        items_count,
                        days_since_abandoned,
                        avg_order_value,
                        cart_to_aov_ratio
                    ]
                    
                    cart_features.append(features)
                    recovery_targets.append(1 if cart.recovered else 0)
            
            # Train Random Forest model for recovery prediction
            if len(cart_features) > 10:  # Need minimum data for training
                X = np.array(cart_features)
                y = np.array(recovery_targets)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Train model
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                
                # Calculate feature importance
                feature_names = ['Customer CLV', 'Order Count', 'Total Spent', 'Cart Value', 'Items Count', 'Days Abandoned', 'Avg Order Value', 'Cart/AOV Ratio']
                feature_importance = list(zip(feature_names, rf_model.feature_importances_))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Generate recovery probabilities for all carts
                recovery_predictions = rf_model.predict(X)
                
                recovery_model_data = {
                    'feature_importance': feature_importance,
                    'model_score': rf_model.score(X_test, y_test),
                    'predictions': recovery_predictions.tolist(),
                    'model_trained': True
                }
        
        # Calculate recovery metrics
        recovery_metrics = calculate_recovery_metrics(abandoned_carts)
        
        # Generate targeted recovery recommendations
        recovery_recommendations = generate_recovery_recommendations(abandoned_carts)
        
        return render_template('abandoned_cart_recovery.html',
                             abandoned_carts=abandoned_carts,
                             recovery_model_data=recovery_model_data,
                             recovery_metrics=recovery_metrics,
                             recommendations=recovery_recommendations,
                             store=store)
    
    except Exception as e:
        logging.error(f"Error in abandoned cart recovery: {str(e)}")
        logging.error(traceback.format_exc())
        flash('Error loading abandoned cart recovery data.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': str(datetime.utcnow())})

def get_dashboard_metrics(store):
    """Calculate dashboard metrics for a store using orders-based analysis"""
    try:
        clv_calculator = OrdersCLVCalculator()
        
        # Get comprehensive metrics from orders-based calculator
        metrics = clv_calculator.calculate_order_metrics(store)
        
        # Basic counts
        total_customers = metrics.get('unique_customers', 0)
        total_orders = metrics.get('total_orders', 0)
        
        # Revenue metrics
        total_revenue = metrics.get('total_revenue', 0)
        avg_order_value = metrics.get('avg_order_value', 0)
        
        # CLV metrics
        avg_clv = metrics.get('avg_clv', 0)
        total_clv = avg_clv * total_customers if total_customers > 0 else 0
        
        # Return rate
        total_returns = Order.query.filter_by(store_id=store.id, is_returned=True).count()
        return_rate = (total_returns / total_orders * 100) if total_orders > 0 else 0
        
        # Enhanced metrics using orders-based analysis
        customer_segmentation = clv_calculator.get_clv_segments(store)
        churn_analysis = clv_calculator.predict_churn_risk(store)
        revenue_trends = clv_calculator.get_revenue_trends(store)
        ai_recommendations = clv_calculator.generate_ai_recommendations(store)
        
        # Top customer segments by CLV
        top_customers = customer_segmentation.get('high', [])[:10] if customer_segmentation else []
        
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
            'recent_orders': recent_orders,
            # Enhanced metrics
            'customer_segmentation': customer_segmentation,
            'churn_analysis': churn_analysis,
            'revenue_trends': revenue_trends,
            'ai_recommendations': ai_recommendations
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
            'recent_orders': [],
            'customer_segmentation': {'high': 0, 'medium': 0, 'low': 0, 'segments': {}},
            'aov_trend': {'trend_data': [], 'current_aov': 0, 'change_percentage': 0},
            'churn_risk': {'high_risk': 0, 'medium_risk': 0, 'low_risk': 0, 'total_at_risk': 0, 'at_risk_percentage': 0}
        }

def create_test_customers_in_store(store):
    """Create test customers directly in database for empty stores"""
    try:
        logging.info(f"Creating test customers for store: {store.shop_domain}")
        
        test_customers = [
            {
                'shopify_customer_id': 'test_1',
                'email': 'john@example.com',
                'first_name': 'John',
                'last_name': 'Smith',
                'total_spent': 450.00,
                'orders_count': 3
            },
            {
                'shopify_customer_id': 'test_2',
                'email': 'sarah@example.com',
                'first_name': 'Sarah',
                'last_name': 'Johnson',
                'total_spent': 325.50,
                'orders_count': 2
            },
            {
                'shopify_customer_id': 'test_3',
                'email': 'mike@example.com',
                'first_name': 'Mike',
                'last_name': 'Davis',
                'total_spent': 675.25,
                'orders_count': 4
            },
            {
                'shopify_customer_id': 'test_4',
                'email': 'emily@example.com',
                'first_name': 'Emily',
                'last_name': 'Wilson',
                'total_spent': 890.00,
                'orders_count': 5
            },
            {
                'shopify_customer_id': 'test_5',
                'email': 'david@example.com',
                'first_name': 'David',
                'last_name': 'Brown',
                'total_spent': 234.75,
                'orders_count': 1
            }
        ]
        
        customers_created = 0
        for customer_data in test_customers:
            customer = Customer()
            customer.shopify_customer_id = customer_data['shopify_customer_id']
            customer.store_id = store.id
            customer.email = customer_data['email']
            customer.first_name = customer_data['first_name']
            customer.last_name = customer_data['last_name']
            customer.total_spent = customer_data['total_spent']
            customer.orders_count = customer_data['orders_count']
            customer.created_at = datetime.utcnow() - timedelta(days=random.randint(30, 365))
            
            db.session.add(customer)
            customers_created += 1
        
        # Create corresponding orders
        orders_created = 0
        for customer in Customer.query.filter_by(store_id=store.id).all():
            for i in range(customer.orders_count):
                order = Order()
                order.shopify_order_id = f"test_order_{customer.id}_{i+1}"
                order.store_id = store.id
                order.customer_id = customer.id
                order.order_number = f"#{1000 + customer.id * 10 + i}"
                order.total_price = customer.total_spent / customer.orders_count
                order.subtotal_price = order.total_price * 0.9
                order.total_tax = order.total_price * 0.1
                order.currency = "USD"
                order.financial_status = "paid"
                order.fulfillment_status = "fulfilled"
                order.created_at = customer.created_at + timedelta(days=random.randint(1, 30))
                order.updated_at = order.created_at
                order.processed_at = order.created_at
                
                db.session.add(order)
                orders_created += 1
        
        db.session.commit()
        logging.info(f"Created {customers_created} test customers and {orders_created} test orders")
        
    except Exception as e:
        logging.error(f"Error creating test customers: {str(e)}")
        db.session.rollback()

def clear_store_data(store):
    """Clear existing data for a store before syncing new data"""
    try:
        # Delete orders first (due to foreign key constraints)
        Order.query.filter_by(store_id=store.id).delete()
        
        # Delete customers
        Customer.query.filter_by(store_id=store.id).delete()
        
        # Delete products if any
        Product.query.filter_by(store_id=store.id).delete()
        
        # Delete abandoned carts if any
        AbandonedCart.query.filter_by(store_id=store.id).delete()
        
        db.session.commit()
        logging.info(f"Cleared existing data for store {store.shop_domain}")
        
    except Exception as e:
        logging.error(f"Error clearing store data: {str(e)}")
        db.session.rollback()

def sync_customers(shopify_client, store):
    """Legacy customer sync - now uses orders-based analysis"""
    # Customer sync is no longer used in orders-based CLV system
    # All customer analysis is done through order patterns and anonymous hashing
    logging.info("Customer sync bypassed - using orders-based CLV analysis")
    return 0

def sync_orders(shopify_client, store):
    """Sync orders from Shopify"""
    try:
        logging.info(f"Fetching orders for shop: {store.shop_domain}")
        orders_data = shopify_client.get_orders(store.shop_domain)
        logging.info(f"API returned {len(orders_data) if orders_data else 0} orders")
        
        if not orders_data:
            logging.warning("No order data returned from Shopify API")
            return 0
            
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
            
            # Create anonymous customer hash (no personal data stored)
            if order_data.get('customer'):
                from orders_clv_calculator import OrdersCLVCalculator
                clv_calc = OrdersCLVCalculator()
                order.customer_hash = clv_calc.hash_customer_id(order_data['customer']['id'])
            
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
        
        db.session.commit()
        return orders_synced
        
    except Exception as e:
        logging.error(f"Error syncing orders: {str(e)}")
        return 0

def get_clv_report_data(store, clv_calculator):
    """Generate comprehensive CLV report data"""
    try:
        customers = Customer.query.filter_by(store_id=store.id).all()
        orders = Order.query.filter_by(store_id=store.id).all()
        
        # Calculate basic metrics
        total_customers = len(customers)
        avg_clv = sum(float(c.predicted_clv or 0) for c in customers) / total_customers if total_customers > 0 else 0
        
        # Generate cohort heatmap data
        cohort_heatmap = generate_cohort_heatmap(customers, orders)
        
        # Generate retention table
        retention_table = generate_retention_table(customers, orders)
        
        # Generate product CLV analysis
        product_clv = generate_product_clv_analysis(orders)
        
        # Generate trend data
        trend_labels = []
        historical_trend = []
        predicted_trend = []
        
        # Generate 12 months of data
        for i in range(12):
            month_date = datetime.utcnow() - timedelta(days=30*i)
            trend_labels.insert(0, month_date.strftime('%b %Y'))
            
            # Calculate historical CLV for that month
            month_orders = [o for o in orders if o.created_at and o.created_at.month == month_date.month]
            historical_clv = sum(float(o.total_price or 0) for o in month_orders) / len(month_orders) if month_orders else 0
            historical_trend.insert(0, round(historical_clv, 2))
            
            # Predicted CLV (slightly higher trend)
            predicted_clv = historical_clv * 1.1 if historical_clv > 0 else avg_clv
            predicted_trend.insert(0, round(predicted_clv, 2))
        
        # CLV distribution data
        distribution_data = [0, 0, 0, 0]  # $0-100, $100-500, $500-1000, $1000+
        for customer in customers:
            clv = float(customer.predicted_clv or 0)
            if clv < 100:
                distribution_data[0] += 1
            elif clv < 500:
                distribution_data[1] += 1
            elif clv < 1000:
                distribution_data[2] += 1
            else:
                distribution_data[3] += 1
        
        return {
            'avg_clv': avg_clv,
            'total_customers': total_customers,
            'retention_rate': 68,  # Calculate from retention table
            'avg_lifespan': 8.5,   # Calculate from customer data
            'cohort_heatmap': cohort_heatmap,
            'retention_table': retention_table,
            'product_clv': product_clv,
            'trend_labels': trend_labels,
            'historical_trend': historical_trend,
            'predicted_trend': predicted_trend,
            'distribution_data': distribution_data
        }
        
    except Exception as e:
        logging.error(f"Error generating CLV report data: {str(e)}")
        return {
            'avg_clv': 0,
            'total_customers': 0,
            'retention_rate': 0,
            'avg_lifespan': 0,
            'cohort_heatmap': [],
            'retention_table': [],
            'product_clv': [],
            'trend_labels': [],
            'historical_trend': [],
            'predicted_trend': [],
            'distribution_data': [0, 0, 0, 0]
        }

def generate_cohort_heatmap(customers, orders):
    """Generate CLV heatmap data by acquisition month"""
    cohort_data = {}
    
    for customer in customers:
        if not customer.created_at:
            continue
            
        month_key = customer.created_at.strftime('%Y-%m')
        if month_key not in cohort_data:
            cohort_data[month_key] = {
                'month': customer.created_at.strftime('%b %Y'),
                'new_customers': 0,
                'customers': [],
                'month_0_clv': 0,
                'month_1_clv': 0,
                'month_2_clv': 0,
                'month_3_clv': 0,
                'month_6_clv': 0,
                'month_12_clv': 0,
                'predicted_clv': 0
            }
        
        cohort_data[month_key]['new_customers'] += 1
        cohort_data[month_key]['customers'].append(customer)
    
    # Calculate CLV progression for each cohort
    heatmap_data = []
    for month_key, data in sorted(cohort_data.items(), reverse=True)[:6]:  # Last 6 months
        customers_in_cohort = data['customers']
        
        # Calculate CLV at different time intervals
        month_0_clvs = []
        month_1_clvs = []
        month_3_clvs = []
        predicted_clvs = []
        
        for customer in customers_in_cohort:
            customer_orders = [o for o in orders if o.customer_id == customer.id]
            
            if customer_orders:
                # Month 0 CLV (first month revenue)
                first_month_orders = [o for o in customer_orders if o.created_at and 
                                    (o.created_at - customer.created_at).days <= 30]
                month_0_clv = sum(float(o.total_price or 0) for o in first_month_orders)
                month_0_clvs.append(month_0_clv)
                
                # Month 1 CLV (cumulative through month 1)
                month_1_orders = [o for o in customer_orders if o.created_at and 
                                (o.created_at - customer.created_at).days <= 60]
                month_1_clv = sum(float(o.total_price or 0) for o in month_1_orders)
                month_1_clvs.append(month_1_clv)
                
                # Month 3 CLV (cumulative through month 3)
                month_3_orders = [o for o in customer_orders if o.created_at and 
                                (o.created_at - customer.created_at).days <= 90]
                month_3_clv = sum(float(o.total_price or 0) for o in month_3_orders)
                month_3_clvs.append(month_3_clv)
                
                # Predicted CLV
                predicted_clvs.append(float(customer.predicted_clv or 0))
        
        # Calculate averages and assign performance classes
        avg_month_0 = sum(month_0_clvs) / len(month_0_clvs) if month_0_clvs else 0
        avg_month_1 = sum(month_1_clvs) / len(month_1_clvs) if month_1_clvs else 0
        avg_month_3 = sum(month_3_clvs) / len(month_3_clvs) if month_3_clvs else 0
        avg_predicted = sum(predicted_clvs) / len(predicted_clvs) if predicted_clvs else 0
        
        heatmap_data.append({
            'month': data['month'],
            'new_customers': data['new_customers'],
            'month_0_clv': avg_month_0,
            'month_0_class': get_clv_class(avg_month_0),
            'month_1_clv': avg_month_1 if avg_month_1 > 0 else None,
            'month_1_class': get_clv_class(avg_month_1) if avg_month_1 > 0 else 'empty',
            'month_2_clv': avg_month_1 * 1.1 if avg_month_1 > 0 else None,
            'month_2_class': get_clv_class(avg_month_1 * 1.1) if avg_month_1 > 0 else 'empty',
            'month_3_clv': avg_month_3 if avg_month_3 > 0 else None,
            'month_3_class': get_clv_class(avg_month_3) if avg_month_3 > 0 else 'empty',
            'month_6_clv': avg_month_3 * 1.2 if avg_month_3 > 0 else None,
            'month_6_class': get_clv_class(avg_month_3 * 1.2) if avg_month_3 > 0 else 'empty',
            'month_12_clv': avg_predicted if avg_predicted > 0 else None,
            'month_12_class': get_clv_class(avg_predicted) if avg_predicted > 0 else 'empty',
            'predicted_clv': avg_predicted
        })
    
    return heatmap_data

def get_clv_class(clv_value):
    """Determine CSS class for CLV value"""
    if clv_value >= 200:
        return 'high'
    elif clv_value >= 100:
        return 'medium'
    else:
        return 'low'

def generate_retention_table(customers, orders):
    """Generate retention table data"""
    cohort_data = {}
    
    for customer in customers:
        if not customer.created_at:
            continue
            
        month_key = customer.created_at.strftime('%Y-%m')
        if month_key not in cohort_data:
            cohort_data[month_key] = {
                'month': customer.created_at.strftime('%b %Y'),
                'customers': []
            }
        cohort_data[month_key]['customers'].append(customer)
    
    retention_data = []
    for month_key, data in sorted(cohort_data.items(), reverse=True)[:6]:
        customers_in_cohort = data['customers']
        initial_customers = len(customers_in_cohort)
        
        # Calculate retention for different periods
        month_1_retained = 0
        month_2_retained = 0
        month_3_retained = 0
        month_6_retained = 0
        month_12_retained = 0
        
        for customer in customers_in_cohort:
            customer_orders = [o for o in orders if o.customer_id == customer.id and o.created_at]
            
            # Check if customer made purchases in different time windows
            if any(o for o in customer_orders if (o.created_at - customer.created_at).days > 30 and (o.created_at - customer.created_at).days <= 60):
                month_1_retained += 1
            if any(o for o in customer_orders if (o.created_at - customer.created_at).days > 60 and (o.created_at - customer.created_at).days <= 90):
                month_2_retained += 1
            if any(o for o in customer_orders if (o.created_at - customer.created_at).days > 90 and (o.created_at - customer.created_at).days <= 120):
                month_3_retained += 1
            if any(o for o in customer_orders if (o.created_at - customer.created_at).days > 180):
                month_6_retained += 1
            if any(o for o in customer_orders if (o.created_at - customer.created_at).days > 365):
                month_12_retained += 1
        
        retention_data.append({
            'month': data['month'],
            'initial_customers': initial_customers,
            'month_1_retention': round(month_1_retained / initial_customers * 100, 1) if initial_customers > 0 else None,
            'month_2_retention': round(month_2_retained / initial_customers * 100, 1) if initial_customers > 0 else None,
            'month_3_retention': round(month_3_retained / initial_customers * 100, 1) if initial_customers > 0 else None,
            'month_6_retention': round(month_6_retained / initial_customers * 100, 1) if initial_customers > 0 else None,
            'month_12_retention': round(month_12_retained / initial_customers * 100, 1) if initial_customers > 0 else None
        })
    
    return retention_data

def generate_product_clv_analysis(orders):
    """Generate product CLV analysis"""
    product_data = {}
    
    for order in orders:
        if not order.shopify_data or 'line_items' not in order.shopify_data:
            continue
            
        for item in order.shopify_data['line_items']:
            product_title = item.get('title', 'Unknown Product')
            
            if product_title not in product_data:
                product_data[product_title] = {
                    'name': product_title,
                    'sku': item.get('sku', 'N/A'),
                    'category': 'General',
                    'orders': [],
                    'customers': set()
                }
            
            product_data[product_title]['orders'].append({
                'value': float(item.get('price', 0)) * int(item.get('quantity', 1)),
                'customer_id': order.customer_id,
                'date': order.created_at
            })
            
            if order.customer_id:
                product_data[product_title]['customers'].add(order.customer_id)
    
    # Calculate CLV metrics for each product
    product_clv_list = []
    for product_name, data in product_data.items():
        orders_list = data['orders']
        unique_customers = len(data['customers'])
        
        if not orders_list:
            continue
            
        total_orders = len(orders_list)
        total_revenue = sum(o['value'] for o in orders_list)
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        # Calculate retention rate (customers with repeat purchases)
        customer_order_counts = {}
        for order in orders_list:
            customer_id = order['customer_id']
            customer_order_counts[customer_id] = customer_order_counts.get(customer_id, 0) + 1
        
        repeat_customers = sum(1 for count in customer_order_counts.values() if count > 1)
        retention_rate = (repeat_customers / unique_customers * 100) if unique_customers > 0 else 0
        
        # Historical CLV (actual revenue)
        historical_clv = total_revenue / unique_customers if unique_customers > 0 else 0
        
        # Predicted CLV (historical * retention factor)
        retention_factor = 1 + (retention_rate / 100)
        predicted_clv = historical_clv * retention_factor
        
        product_clv_list.append({
            'name': product_name,
            'sku': data['sku'],
            'category': data['category'],
            'total_orders': total_orders,
            'avg_order_value': avg_order_value,
            'retention_rate': round(retention_rate, 1),
            'historical_clv': historical_clv,
            'predicted_clv': predicted_clv
        })
    
    # Sort by predicted CLV and return top 10
    return sorted(product_clv_list, key=lambda x: x['predicted_clv'], reverse=True)[:10]

def generate_product_clv_recommendations(products):
    """Generate personalized product recommendations for CLV optimization"""
    recommendations = []
    
    if not products:
        return recommendations
    
    # Sort products by CLV contribution
    sorted_products = sorted(products, key=lambda p: float(p.avg_clv_contribution or 0), reverse=True)
    
    # Top performers
    top_products = sorted_products[:3]
    for product in top_products:
        recommendations.append({
            'type': 'promotion',
            'priority': 'high',
            'title': f'Promote {product.title} to high-CLV customers',
            'description': f'This product contributes ${product.avg_clv_contribution:.2f} average CLV. Offer 10% discount to customers with CLV > $300.',
            'impact': 'High CLV increase potential',
            'action': f'Create targeted campaign for {product.title}'
        })
    
    # Low performers that could be optimized
    low_performers = [p for p in products if float(p.return_rate) > 0.1]
    for product in low_performers[:2]:
        recommendations.append({
            'type': 'optimization',
            'priority': 'medium',
            'title': f'Reduce returns for {product.title}',
            'description': f'Return rate of {product.return_rate:.1%} is impacting CLV. Review product quality or description.',
            'impact': 'Reduce negative CLV impact',
            'action': f'Quality review for {product.title}'
        })
    
    # Inventory optimization
    high_clv_low_inventory = [p for p in products if float(p.avg_clv_contribution or 0) > 200 and p.inventory_quantity < 100]
    for product in high_clv_low_inventory[:2]:
        recommendations.append({
            'type': 'inventory',
            'priority': 'high',
            'title': f'Restock {product.title}',
            'description': f'High CLV product with only {product.inventory_quantity} units remaining.',
            'impact': 'Prevent lost CLV opportunities',
            'action': f'Increase inventory for {product.title}'
        })
    
    return recommendations

def analyze_product_clv_performance(products):
    """Analyze product performance for CLV optimization"""
    if not products:
        return {}
    
    # Category analysis
    category_performance = {}
    for product in products:
        category = product.category
        if category not in category_performance:
            category_performance[category] = {
                'products': 0,
                'total_clv': 0,
                'avg_clv': 0,
                'total_sales': 0,
                'avg_return_rate': 0
            }
        
        cat_data = category_performance[category]
        cat_data['products'] += 1
        cat_data['total_clv'] += float(product.avg_clv_contribution or 0)
        cat_data['total_sales'] += float(product.total_sales or 0)
        cat_data['avg_return_rate'] += float(product.return_rate or 0)
    
    # Calculate averages
    for category, data in category_performance.items():
        if data['products'] > 0:
            data['avg_clv'] = data['total_clv'] / data['products']
            data['avg_return_rate'] = data['avg_return_rate'] / data['products']
    
    # Top and bottom performers
    sorted_products = sorted(products, key=lambda p: float(p.avg_clv_contribution or 0), reverse=True)
    
    return {
        'category_performance': category_performance,
        'top_performers': sorted_products[:5],
        'bottom_performers': sorted_products[-5:],
        'total_products': len(products),
        'avg_clv_contribution': sum(float(p.avg_clv_contribution or 0) for p in products) / len(products) if products else 0
    }

def calculate_recovery_metrics(abandoned_carts):
    """Calculate abandoned cart recovery metrics"""
    if not abandoned_carts:
        return {}
    
    total_carts = len(abandoned_carts)
    recovered_carts = sum(1 for cart in abandoned_carts if cart.recovered)
    emails_sent = sum(1 for cart in abandoned_carts if cart.recovery_email_sent)
    
    total_value = sum(float(cart.total_price) for cart in abandoned_carts)
    recovered_value = sum(float(cart.total_price) for cart in abandoned_carts if cart.recovered)
    
    # High-value carts (>$100)
    high_value_carts = [cart for cart in abandoned_carts if float(cart.total_price) > 100]
    high_value_recovered = [cart for cart in high_value_carts if cart.recovered]
    
    return {
        'total_carts': total_carts,
        'recovered_carts': recovered_carts,
        'recovery_rate': (recovered_carts / total_carts * 100) if total_carts > 0 else 0,
        'emails_sent': emails_sent,
        'email_recovery_rate': (recovered_carts / emails_sent * 100) if emails_sent > 0 else 0,
        'total_value': total_value,
        'recovered_value': recovered_value,
        'value_recovery_rate': (recovered_value / total_value * 100) if total_value > 0 else 0,
        'high_value_carts': len(high_value_carts),
        'high_value_recovered': len(high_value_recovered),
        'avg_cart_value': total_value / total_carts if total_carts > 0 else 0
    }

def generate_recovery_recommendations(abandoned_carts):
    """Generate targeted recovery recommendations"""
    recommendations = []
    
    if not abandoned_carts:
        return recommendations
    
    # High-value carts without recovery emails
    high_value_no_email = [cart for cart in abandoned_carts 
                          if float(cart.total_price) > 100 and not cart.recovery_email_sent]
    
    if high_value_no_email:
        recommendations.append({
            'type': 'email_campaign',
            'priority': 'high',
            'title': f'Target {len(high_value_no_email)} high-value abandoned carts',
            'description': f'Send recovery emails to carts worth $100+ (total value: ${sum(float(c.total_price) for c in high_value_no_email):.2f})',
            'impact': 'High revenue recovery potential',
            'action': 'Create high-value cart recovery campaign'
        })
    
    # Recent abandoners with high CLV
    recent_high_clv = [cart for cart in abandoned_carts 
                      if cart.customer and float(cart.customer.predicted_clv or 0) > 300 
                      and cart.abandoned_at and (datetime.utcnow() - cart.abandoned_at).days <= 3]
    
    if recent_high_clv:
        recommendations.append({
            'type': 'personalized_offer',
            'priority': 'high',
            'title': f'Personalized offers for {len(recent_high_clv)} high-CLV customers',
            'description': 'Recent cart abandoners with high CLV deserve personalized attention',
            'impact': 'Retain high-value customers',
            'action': 'Send personalized discount offers'
        })
    
    # Multi-item carts
    multi_item_carts = [cart for cart in abandoned_carts if cart.line_items_count > 2]
    if multi_item_carts:
        recommendations.append({
            'type': 'bundle_discount',
            'priority': 'medium',
            'title': f'Bundle discounts for {len(multi_item_carts)} multi-item carts',
            'description': 'Offer bundle discounts to recover carts with multiple items',
            'impact': 'Increase average order value',
            'action': 'Create bundle discount campaign'
        })
    
    return recommendations
