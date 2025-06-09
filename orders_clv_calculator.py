import hashlib
import logging
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
from sqlalchemy import func, text
from app import db
from models import Order, ShopifyStore


class OrdersCLVCalculator:
    """Calculate Customer Lifetime Value using orders-only analysis (no customer data stored)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def hash_customer_id(self, customer_id: str) -> str:
        """Create anonymous hash for customer grouping without storing personal data"""
        return hashlib.sha256(str(customer_id).encode()).hexdigest()
    
    def calculate_order_metrics(self, store) -> Dict:
        """Calculate comprehensive order-based CLV metrics"""
        orders = Order.query.filter_by(store_id=store.id).all()
        
        if not orders:
            return self._empty_metrics()
        
        # Group orders by customer hash
        customer_groups = defaultdict(list)
        for order in orders:
            if order.customer_hash:
                customer_groups[order.customer_hash].append(order)
        
        total_revenue = sum(float(order.total_price or 0) for order in orders)
        total_orders = len(orders)
        unique_customers = len(customer_groups)
        
        # Calculate CLV metrics
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        # Customer lifetime metrics
        customer_lifetimes = []
        customer_clvs = []
        repeat_customers = 0
        
        for customer_hash, customer_orders in customer_groups.items():
            customer_orders.sort(key=lambda x: x.created_at or datetime.min)
            
            if len(customer_orders) > 1:
                repeat_customers += 1
                first_order = customer_orders[0].created_at
                last_order = customer_orders[-1].created_at
                
                if first_order and last_order:
                    lifetime_days = (last_order - first_order).days
                    customer_lifetimes.append(lifetime_days)
            
            # Calculate CLV for this customer
            customer_total = sum(float(order.total_price or 0) for order in customer_orders)
            customer_frequency = len(customer_orders)
            customer_clvs.append(customer_total)
        
        avg_customer_lifetime = np.mean(customer_lifetimes) if customer_lifetimes else 30
        avg_clv = np.mean(customer_clvs) if customer_clvs else 0
        
        return {
            'total_revenue': total_revenue,
            'total_orders': total_orders,
            'unique_customers': unique_customers,
            'avg_order_value': avg_order_value,
            'avg_customer_lifetime_days': avg_customer_lifetime,
            'avg_clv': avg_clv,
            'repeat_customer_rate': (repeat_customers / unique_customers * 100) if unique_customers > 0 else 0,
            'customer_groups': len(customer_groups)
        }
    
    def get_clv_segments(self, store) -> Dict[str, List]:
        """Segment anonymous customers by CLV"""
        orders = Order.query.filter_by(store_id=store.id).all()
        
        if not orders:
            return {'high': [], 'medium': [], 'low': []}
        
        # Group by customer hash and calculate CLV
        customer_groups = defaultdict(list)
        for order in orders:
            if order.customer_hash:
                customer_groups[order.customer_hash].append(order)
        
        customer_clvs = []
        for customer_hash, customer_orders in customer_groups.items():
            total_spent = sum(float(order.total_price or 0) for order in customer_orders)
            order_count = len(customer_orders)
            
            customer_clvs.append({
                'customer_hash': customer_hash,
                'total_spent': total_spent,
                'order_count': order_count,
                'avg_order_value': total_spent / order_count if order_count > 0 else 0
            })
        
        # Sort by total spent and segment
        customer_clvs.sort(key=lambda x: x['total_spent'], reverse=True)
        
        total_customers = len(customer_clvs)
        high_threshold = int(total_customers * 0.2)  # Top 20%
        medium_threshold = int(total_customers * 0.6)  # Next 40%
        
        return {
            'high': len(customer_clvs[:high_threshold]),
            'medium': len(customer_clvs[high_threshold:high_threshold + medium_threshold]),
            'low': len(customer_clvs[high_threshold + medium_threshold:])
        }
    
    def get_order_cohort_analysis(self, store) -> Dict:
        """Analyze order patterns by acquisition cohorts"""
        orders = Order.query.filter_by(store_id=store.id).order_by(Order.created_at).all()
        
        if not orders:
            return {}
        
        # Group by customer and find first order date
        customer_groups = defaultdict(list)
        for order in orders:
            if order.customer_hash and order.created_at:
                customer_groups[order.customer_hash].append(order)
        
        cohort_data = defaultdict(lambda: defaultdict(int))
        
        for customer_hash, customer_orders in customer_groups.items():
            customer_orders.sort(key=lambda x: x.created_at)
            first_order_date = customer_orders[0].created_at
            cohort_month = first_order_date.strftime('%Y-%m')
            
            for order in customer_orders:
                months_since_first = ((order.created_at.year - first_order_date.year) * 12 + 
                                    order.created_at.month - first_order_date.month)
                cohort_data[cohort_month][months_since_first] += 1
        
        return dict(cohort_data)
    
    def predict_churn_risk(self, store) -> Dict:
        """Predict churn risk based on order patterns"""
        orders = Order.query.filter_by(store_id=store.id).all()
        
        if not orders:
            return {}
        
        customer_groups = defaultdict(list)
        for order in orders:
            if order.customer_hash:
                customer_groups[order.customer_hash].append(order)
        
        churn_analysis = []
        current_date = datetime.utcnow()
        
        for customer_hash, customer_orders in customer_groups.items():
            customer_orders.sort(key=lambda x: x.created_at or datetime.min)
            
            if len(customer_orders) >= 2:
                # Calculate average days between orders
                intervals = []
                for i in range(1, len(customer_orders)):
                    if customer_orders[i].created_at and customer_orders[i-1].created_at:
                        interval = (customer_orders[i].created_at - customer_orders[i-1].created_at).days
                        intervals.append(interval)
                
                if intervals:
                    avg_interval = np.mean(intervals)
                    last_order_date = customer_orders[-1].created_at
                    days_since_last = (current_date - last_order_date).days if last_order_date else 999
                    
                    # Simple churn risk calculation
                    expected_next_order = avg_interval * 1.5  # 50% buffer
                    churn_risk = min(days_since_last / expected_next_order, 1.0) * 100
                    
                    churn_analysis.append({
                        'customer_hash': customer_hash,
                        'churn_risk': churn_risk,
                        'days_since_last_order': days_since_last,
                        'avg_order_interval': avg_interval,
                        'total_orders': len(customer_orders),
                        'total_spent': sum(float(order.total_price or 0) for order in customer_orders)
                    })
        
        # Sort by churn risk
        churn_analysis.sort(key=lambda x: x['churn_risk'], reverse=True)
        
        high_risk = len([c for c in churn_analysis if c['churn_risk'] > 80])
        medium_risk = len([c for c in churn_analysis if 50 <= c['churn_risk'] <= 80])
        low_risk = len([c for c in churn_analysis if c['churn_risk'] < 50])
        total_at_risk = high_risk + medium_risk
        
        return {
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'total_at_risk': total_at_risk,
            'at_risk_percentage': round((total_at_risk / len(churn_analysis) * 100), 1) if churn_analysis else 0
        }
    
    def get_revenue_trends(self, store) -> Dict:
        """Calculate revenue trends and growth metrics"""
        orders = Order.query.filter_by(store_id=store.id).order_by(Order.created_at).all()
        
        if not orders:
            return {}
        
        # Group by month
        monthly_revenue = defaultdict(float)
        monthly_orders = defaultdict(int)
        
        for order in orders:
            if order.created_at and order.total_price:
                month_key = order.created_at.strftime('%Y-%m')
                monthly_revenue[month_key] += float(order.total_price)
                monthly_orders[month_key] += 1
        
        # Calculate trends
        sorted_months = sorted(monthly_revenue.keys())
        
        if len(sorted_months) >= 2:
            current_month = monthly_revenue[sorted_months[-1]]
            previous_month = monthly_revenue[sorted_months[-2]]
            growth_rate = ((current_month - previous_month) / previous_month * 100) if previous_month > 0 else 0
        else:
            growth_rate = 0
        
        return {
            'monthly_revenue': dict(monthly_revenue),
            'monthly_orders': dict(monthly_orders),
            'growth_rate': growth_rate,
            'total_months': len(sorted_months)
        }
    
    def generate_ai_recommendations(self, store) -> List[str]:
        """Generate AI-powered CLV optimization recommendations based on order patterns"""
        metrics = self.calculate_order_metrics(store)
        segments = self.get_clv_segments(store)
        churn_analysis = self.predict_churn_risk(store)
        
        recommendations = []
        
        # Revenue recommendations
        if metrics['avg_order_value'] < 50:
            recommendations.append("Implement upselling strategies to increase average order value (currently ${:.2f})".format(metrics['avg_order_value']))
        
        # Repeat customer recommendations
        if metrics['repeat_customer_rate'] < 30:
            recommendations.append("Focus on customer retention - only {:.1f}% of customers make repeat purchases".format(metrics['repeat_customer_rate']))
        
        # Churn prevention
        if churn_analysis and churn_analysis.get('high_risk', 0) > 0:
            high_risk_count = churn_analysis['high_risk']
            recommendations.append("Implement win-back campaigns for {} high-risk customers".format(high_risk_count))
        
        # Segmentation recommendations
        if segments.get('high', 0) > 0:
            high_value_count = segments['high']
            recommendations.append("Create VIP program for {} high-value customers to increase retention".format(high_value_count))
        
        return recommendations
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_revenue': 0,
            'total_orders': 0,
            'unique_customers': 0,
            'avg_order_value': 0,
            'avg_customer_lifetime_days': 0,
            'avg_clv': 0,
            'repeat_customer_rate': 0,
            'customer_groups': 0
        }