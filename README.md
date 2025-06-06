
# CustomerLifetimeAnalyzer
--- 
# Customer Lifetime Value (CLV) Optimization Platform

An AI-powered Customer Lifetime Value optimization platform for Shopify stores, providing advanced analytics, predictive modeling, and actionable insights for merchants.

## Features

### 🎯 Core Analytics
- **CLV Heatmaps**: Visual progression analysis by customer acquisition month
- **Retention Analysis**: Cohort-based retention tracking across multiple time periods
- **Product CLV Impact**: Historical vs predicted CLV analysis by product
- **Customer Segmentation**: High/Medium/Low value customer categorization

### 📊 Advanced Reports
- **Overview Dashboard**: Key metrics, revenue trends, and customer insights
- **Orders Report**: Comprehensive order analytics with filtering capabilities
- **CLV Report**: Detailed lifetime value analysis with interactive visualizations
- **Customer Segmentation**: AI-powered customer categorization and recommendations

### 🔗 Shopify Integration
- Secure OAuth 2.0 authentication
- Real-time data synchronization
- Customer and order data import
- Abandoned cart analysis

### 🤖 AI-Powered Features
- Churn risk prediction
- CLV optimization recommendations
- Revenue retention forecasting
- Product return rate analysis

## Technology Stack

- **Backend**: Flask with SQLAlchemy ORM
- **Database**: PostgreSQL (Supabase compatible)
- **Frontend**: Bootstrap 5 with Chart.js visualizations
- **Authentication**: Shopify OAuth 2.0
- **Deployment**: Replit with Gunicorn

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL database (Supabase recommended)
- Shopify Partner account with app credentials

### Environment Variables
```env
DATABASE_URL=postgresql://user:password@host:port/database
SESSION_SECRET=your-session-secret
SHOPIFY_API_KEY=your-shopify-api-key
SHOPIFY_API_SECRET=your-shopify-api-secret
```

### Installation
```bash
# Clone the repository
git clone https://github.com/jklardata/CustomerLifetimeAnalyzer.git
cd CustomerLifetimeAnalyzer

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run the application
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

### Shopify App Configuration
1. Create a Shopify Partner account
2. Create a new app with these settings:
   - App URL: `https://your-domain.com/auth`
   - Allowed redirection URLs: `https://your-domain.com/callback`
   - Required scopes: `read_orders,read_customers,read_analytics`

## Demo Mode

Try the platform without Shopify integration:
1. Visit `/auth`
2. Click "Try Demo Mode"
3. Explore with comprehensive sample data

## Key Components

### Models (`models.py`)
- **ShopifyStore**: Store configuration and OAuth tokens
- **Customer**: Customer data with CLV metrics
- **Order**: Order history and financial data
- **CLVPrediction**: AI-generated CLV forecasts

### CLV Calculator (`clv_calculator.py`)
- Basic CLV calculation using order history
- Advanced metrics: AOV, purchase frequency, customer lifespan
- Churn risk prediction algorithms
- Revenue retention analysis

### Shopify Client (`shopify_client.py`)
- OAuth 2.0 authentication flow
- Rate-limited API requests
- Customer and order synchronization
- Error handling and retries

## API Endpoints

- `GET /` - Landing page
- `GET /auth` - Shopify authentication
- `GET /callback` - OAuth callback handler
- `GET /dashboard` - Main analytics dashboard
- `GET /reports/orders` - Orders report
- `GET /reports/clv` - CLV analysis report
- `GET /reports/segmentation` - Customer segmentation
- `POST /sync` - Sync Shopify data
- `GET /logout` - Clear session

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is proprietary software developed for Klardata.

## Support

For technical support or feature requests, contact: justin@klardata.com
4dac753 (Set up initial project structure and provide setup instructions)
