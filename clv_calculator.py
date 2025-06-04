import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
from app import db
from models import Customer, Order, CLVPrediction
import pandas as pd
from sqlalchemy import and_

class CLVCalculator:
    """Calculate Customer Lifetime Value using various methods"""
    
    def __init__(self):
        self.default_lifespan = 365  # Default customer lifespan in days
        
    def calculate_basic_clv(self, customer: Customer) -> Optional[Decimal]:
        """
        Calculate basic CLV using the formula:
        CLV = (avg_order_value * purchase_frequency) * customer_lifespan * (1 - return_rate)
        """
        try:
            if not customer.orders:
                return Decimal('0.00')
            
            # Calculate metrics
            avg_order_value = self.calculate_avg_order_value(customer)
            purchase_frequency = self.calculate_purchase_frequency(customer)
            customer_lifespan = self.calculate_customer_lifespan(customer)
            return_rate = self.calculate_return_rate(customer)
            
            if not all([avg_order_value, purchase_frequency, customer_lifespan is not None]):
                return Decimal('0.00')
            
            # Basic CLV formula
            clv = (avg_order_value * Decimal(str(purchase_frequency))) * Decimal(str(customer_lifespan)) * (Decimal('1.0') - Decimal(str(return_rate)))
            
            # Update customer record
            customer.avg_order_value = avg_order_value
            customer.purchase_frequency = purchase_frequency
            customer.customer_lifespan = customer_lifespan
            customer.return_rate = return_rate
            customer.predicted_clv = clv
            
            return clv
            
        except Exception as e:
            logging.error(f"Error calculating CLV for customer {customer.id}: {str(e)}")
            return Decimal('0.00')
    
    def calculate_avg_order_value(self, customer: Customer) -> Optional[Decimal]:
        """Calculate average order value for a customer"""
        try:
            orders = [order for order in customer.orders if order.total_price and order.total_price > 0]
            if not orders:
                return Decimal('0.00')
            
            total_value = sum(order.total_price for order in orders)
            return total_value / len(orders)
            
        except Exception as e:
            logging.error(f"Error calculating AOV for customer {customer.id}: {str(e)}")
            return Decimal('0.00')
    
    def calculate_purchase_frequency(self, customer: Customer) -> float:
        """Calculate purchase frequency (orders per month)"""
        try:
            orders = [order for order in customer.orders if order.created_at]
            if len(orders) < 2:
                return 1.0  # Default frequency for customers with few orders
            
            # Sort orders by date
            orders.sort(key=lambda x: x.created_at)
            
            # Calculate time span
            first_order = orders[0].created_at
            last_order = orders[-1].created_at
            time_span_days = (last_order - first_order).days
            
            if time_span_days == 0:
                return len(orders)  # All orders on same day
            
            # Convert to monthly frequency
            time_span_months = time_span_days / 30.44  # Average days per month
            frequency = len(orders) / time_span_months if time_span_months > 0 else 1.0
            
            return max(frequency, 0.1)  # Minimum frequency to avoid division by zero
            
        except Exception as e:
            logging.error(f"Error calculating purchase frequency for customer {customer.id}: {str(e)}")
            return 1.0
    
    def calculate_customer_lifespan(self, customer: Customer) -> float:
        """Calculate customer lifespan in months"""
        try:
            orders = [order for order in customer.orders if order.created_at]
            if not orders:
                return 12.0  # Default 12 months for new customers
            
            # Sort orders by date
            orders.sort(key=lambda x: x.created_at)
            
            first_order = orders[0].created_at
            last_order = orders[-1].created_at
            
            # Calculate actual lifespan
            actual_lifespan_days = (last_order - first_order).days
            actual_lifespan_months = actual_lifespan_days / 30.44
            
            # Predict future lifespan based on purchase frequency
            purchase_frequency = self.calculate_purchase_frequency(customer)
            
            # If customer is active (recent orders), predict extended lifespan
            days_since_last_order = (datetime.utcnow() - last_order).days
            if days_since_last_order < 90:  # Active within 3 months
                predicted_extension = max(12, purchase_frequency * 2)  # At least 12 months more
                return max(actual_lifespan_months + predicted_extension, 12.0)
            else:
                return max(actual_lifespan_months, 6.0)  # Minimum 6 months
            
        except Exception as e:
            logging.error(f"Error calculating customer lifespan for customer {customer.id}: {str(e)}")
            return 12.0
    
    def calculate_return_rate(self, customer: Customer) -> float:
        """Calculate return rate for a customer"""
        try:
            total_orders = len(customer.orders)
            if total_orders == 0:
                return 0.0
            
            returned_orders = len([order for order in customer.orders if order.is_returned])
            return returned_orders / total_orders
            
        except Exception as e:
            logging.error(f"Error calculating return rate for customer {customer.id}: {str(e)}")
            return 0.0
    
    def calculate_store_clv(self, store) -> int:
        """Calculate CLV for all customers in a store"""
        try:
            customers = Customer.query.filter_by(store_id=store.id).all()
            updates = 0
            
            for customer in customers:
                clv = self.calculate_basic_clv(customer)
                if clv and clv > 0:
                    updates += 1
            
            db.session.commit()
            logging.info(f"Updated CLV for {updates} customers in store {store.shop_name}")
            return updates
            
        except Exception as e:
            logging.error(f"Error calculating store CLV: {str(e)}")
            return 0
    
    def get_clv_segments(self, store) -> Dict[str, List]:
        """Segment customers by CLV"""
        try:
            customers = Customer.query.filter(
                and_(
                    Customer.store_id == store.id,
                    Customer.predicted_clv.isnot(None),
                    Customer.predicted_clv > 0
                )
            ).order_by(Customer.predicted_clv.desc()).all()
            
            if not customers:
                return {
                    'high_value': [],
                    'medium_value': [],
                    'low_value': []
                }
            
            # Calculate quartiles
            clv_values = [float(customer.predicted_clv) for customer in customers]
            clv_df = pd.DataFrame({'clv': clv_values})
            
            q75 = clv_df['clv'].quantile(0.75)
            q25 = clv_df['clv'].quantile(0.25)
            
            # Segment customers
            high_value = [c for c in customers if float(c.predicted_clv) >= q75]
            medium_value = [c for c in customers if q25 <= float(c.predicted_clv) < q75]
            low_value = [c for c in customers if float(c.predicted_clv) < q25]
            
            return {
                'high_value': high_value[:50],  # Limit to top 50
                'medium_value': medium_value[:50],
                'low_value': low_value[:50]
            }
            
        except Exception as e:
            logging.error(f"Error segmenting customers: {str(e)}")
            return {
                'high_value': [],
                'medium_value': [],
                'low_value': []
            }
    
    def predict_churn_risk(self, customer: Customer) -> float:
        """Predict churn risk based on order patterns"""
        try:
            orders = [order for order in customer.orders if order.created_at]
            if not orders:
                return 0.5  # Medium risk for customers with no orders
            
            # Sort orders by date
            orders.sort(key=lambda x: x.created_at)
            last_order = orders[-1].created_at
            
            # Calculate days since last order
            days_since_last_order = (datetime.utcnow() - last_order).days
            
            # Calculate average time between orders
            if len(orders) > 1:
                intervals = []
                for i in range(1, len(orders)):
                    interval = (orders[i].created_at - orders[i-1].created_at).days
                    intervals.append(interval)
                avg_interval = sum(intervals) / len(intervals)
            else:
                avg_interval = 30  # Default for single order customers
            
            # Risk increases with time since last order relative to normal interval
            risk_multiplier = days_since_last_order / max(avg_interval, 1)
            
            # Cap risk between 0 and 1
            churn_risk = min(risk_multiplier / 3, 1.0)  # Normalize to 0-1 scale
            
            return churn_risk
            
        except Exception as e:
            logging.error(f"Error predicting churn risk for customer {customer.id}: {str(e)}")
            return 0.5
    
    def get_clv_insights(self, store) -> Dict:
        """Get CLV insights and recommendations"""
        try:
            customers = Customer.query.filter_by(store_id=store.id).all()
            
            # Calculate aggregate metrics
            total_customers = len(customers)
            customers_with_clv = [c for c in customers if c.predicted_clv and c.predicted_clv > 0]
            
            if not customers_with_clv:
                return {
                    'total_customers': total_customers,
                    'avg_clv': 0,
                    'clv_distribution': {},
                    'recommendations': []
                }
            
            clv_values = [float(c.predicted_clv) for c in customers_with_clv]
            avg_clv = sum(clv_values) / len(clv_values)
            
            # CLV distribution
            clv_ranges = {
                '0-100': len([v for v in clv_values if 0 <= v < 100]),
                '100-500': len([v for v in clv_values if 100 <= v < 500]),
                '500-1000': len([v for v in clv_values if 500 <= v < 1000]),
                '1000+': len([v for v in clv_values if v >= 1000])
            }
            
            # Generate recommendations
            recommendations = self.generate_recommendations(customers_with_clv, avg_clv)
            
            return {
                'total_customers': total_customers,
                'customers_with_clv': len(customers_with_clv),
                'avg_clv': round(avg_clv, 2),
                'clv_distribution': clv_ranges,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logging.error(f"Error generating CLV insights: {str(e)}")
            return {}
    
    def generate_recommendations(self, customers: List[Customer], avg_clv: float) -> List[str]:
        """Generate actionable CLV recommendations"""
        recommendations = []
        
        try:
            # High-value customer retention
            high_clv_customers = [c for c in customers if float(c.predicted_clv) > avg_clv * 1.5]
            if high_clv_customers:
                recommendations.append(f"Focus on retaining {len(high_clv_customers)} high-value customers with personalized offers")
            
            # Churn risk analysis
            at_risk_customers = []
            for customer in customers:
                churn_risk = self.predict_churn_risk(customer)
                if churn_risk > 0.7:
                    at_risk_customers.append(customer)
            
            if at_risk_customers:
                recommendations.append(f"Re-engage {len(at_risk_customers)} customers at high churn risk with win-back campaigns")
            
            # Purchase frequency optimization
            low_frequency_customers = [c for c in customers if c.purchase_frequency and c.purchase_frequency < 1.0]
            if low_frequency_customers:
                recommendations.append(f"Increase purchase frequency for {len(low_frequency_customers)} customers through email marketing")
            
            # AOV improvement
            low_aov_customers = [c for c in customers if c.avg_order_value and float(c.avg_order_value) < avg_clv * 0.1]
            if low_aov_customers:
                recommendations.append(f"Increase average order value for {len(low_aov_customers)} customers with bundle offers")
            
            if not recommendations:
                recommendations.append("Continue monitoring customer behavior to identify optimization opportunities")
            
        except Exception as e:
            logging.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("Unable to generate specific recommendations at this time")
        
        return recommendations
