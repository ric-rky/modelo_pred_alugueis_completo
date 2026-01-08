"""
Rental Price Prediction - Interactive Streamlit Application
Author: Ricardo Luís Bertolucci Filho
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Rental Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load the rental data from CSV file"""
    try:
        df = pd.read_csv('projeto_aluguel/data/data.csv')
        return df
    except:
        # Generate sample data if file not found
        np.random.seed(42)
        n_samples = 1000
        data = {
            'area': np.random.randint(15, 200, n_samples),
            'bedrooms': np.random.randint(1, 5, n_samples),
            'garage': np.random.randint(0, 4, n_samples),
            'rent': np.random.randint(500, 8000, n_samples),
            'district': np.random.choice(['Pinheiros', 'Vila Mariana', 'Moema', 'Brooklin', 'Itaim'], n_samples),
            'type': np.random.choice(['Apartamento', 'Studio e kitnet', 'Casa'], n_samples)
        }
        df = pd.DataFrame(data)
        # Adjust rent based on features
        df['rent'] = (df['area'] * 30 + df['bedrooms'] * 500 + df['garage'] * 200 + 
                     np.random.normal(0, 300, n_samples)).astype(int)
        df['rent'] = df['rent'].clip(500, 15000)
        return df

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = load_data()

# Sidebar navigation
st.sidebar.title("🏠 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Data Exploration", "Model Performance", "Make Predictions"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application provides an interactive interface for exploring "
    "rental price predictions in São Paulo, Brazil."
)

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "Home":
    st.markdown('<h1 class="main-header">🏠 Rental Price Prediction System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome! 👋
    
    This interactive application is designed to predict residential rental prices in São Paulo, Brazil, 
    using machine learning techniques.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Objectives")
        st.markdown("""
        - Identify key factors influencing rental prices
        - Provide reliable price predictions
        - Compare different ML algorithms
        - Support real estate decision-making
        """)
    
    with col2:
        st.markdown("### 📊 Features")
        st.markdown("""
        - Interactive data exploration
        - Model performance comparison
        - Real-time price predictions
        - Visual analytics and insights
        """)
    
    with col3:
        st.markdown("### 🛠️ Technologies")
        st.markdown("""
        - Python 3.12
        - Streamlit
        - Scikit-learn
        - Pandas & NumPy
        - Matplotlib & Seaborn
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ## 📈 Project Overview
    
    The real estate market is one of the most important sectors of the economy, and accurate 
    property pricing directly impacts owners, tenants, and investors. This project aims to 
    develop predictive models that assist in property pricing.
    
    ### Key Highlights:
    
    - **Data-Driven**: Uses real rental data from São Paulo
    - **Multiple Models**: Compares Linear Regression, Decision Trees, Random Forest, and Gradient Boosting
    - **Best Performance**: Random Forest achieved R² ≈ 0.997 with MAE of ~R$ 35
    - **Professional Analysis**: Includes statistical validation and feature importance analysis
    
    ### How to Use:
    
    1. 📊 **Data Exploration**: Explore the dataset, view statistics, and visualize distributions
    2. 🎯 **Model Performance**: Compare different models and their metrics
    3. 🔮 **Make Predictions**: Input property characteristics to get instant price predictions
    
    ---
    
    ### 👨‍💻 Author
    
    **Ricardo Luís Bertolucci Filho**
    - [LinkedIn](https://www.linkedin.com/in/ricardo-lu%C3%ADs-bertolucci-filho/)
    - [GitHub](https://github.com/ric-rky/ric-rky)
    - Email: bertolucci.rl@gmail.com
    """)

# ============================================================================
# DATA EXPLORATION PAGE
# ============================================================================
elif page == "Data Exploration":
    st.markdown('<h1 class="main-header">📊 Data Exploration</h1>', unsafe_allow_html=True)
    
    df = st.session_state.data
    
    # Display sample data
    st.markdown("### 📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Dataset information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Properties", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Average Rent", f"R$ {df['rent'].mean():.2f}")
    with col4:
        st.metric("Max Rent", f"R$ {df['rent'].max():,.2f}")
    
    st.markdown("---")
    
    # Descriptive statistics
    st.markdown("### 📈 Descriptive Statistics")
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("### 📊 Data Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📍 By Features"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Rent Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['rent'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel('Rent (R$)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Rental Prices', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Area Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['area'], bins=50, edgecolor='black', alpha=0.7, color='coral')
            ax.set_xlabel('Area (m²)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Property Area', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
    
    with tab2:
        st.markdown("#### Correlation Heatmap")
        
        # Calculate correlation matrix
        numeric_df = df[numeric_cols]
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, ax=ax,
                   cbar_kws={"shrink": 0.8})
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("""
        **Interpretation:**
        - Values close to 1 indicate strong positive correlation
        - Values close to -1 indicate strong negative correlation
        - Values close to 0 indicate weak or no correlation
        """)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Rent by Number of Bedrooms")
            fig, ax = plt.subplots(figsize=(10, 6))
            bedroom_avg = df.groupby('bedrooms')['rent'].mean().sort_index()
            ax.bar(bedroom_avg.index, bedroom_avg.values, color='teal', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Number of Bedrooms', fontsize=12)
            ax.set_ylabel('Average Rent (R$)', fontsize=12)
            ax.set_title('Average Rent by Bedrooms', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Rent vs Area")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df['area'], df['rent'], alpha=0.5, color='purple', edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Area (m²)', fontsize=12)
            ax.set_ylabel('Rent (R$)', fontsize=12)
            ax.set_title('Rent vs Property Area', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

# ============================================================================
# MODEL PERFORMANCE PAGE
# ============================================================================
elif page == "Model Performance":
    st.markdown('<h1 class="main-header">🎯 Model Performance</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This page presents the performance comparison of different machine learning models 
    used for rental price prediction.
    """)
    
    # Model comparison data
    model_data = {
        'Model': [
            'Linear Regression (rent)',
            'Linear Regression (log)',
            'Decision Tree',
            'Random Forest',
            'Gradient Boosting'
        ],
        'R²': [0.625, 0.710, 0.9934, 0.9966, 0.9916],
        'MAE (R$)': [883.50, 1306.20, 67.45, 35.58, 116.48],
        'RMSE (R$)': [1397.46, 1450.00, 182.87, 131.32, 205.67],
        'Notes': [
            'Without log-transform',
            'Better fit with log',
            'Strong overfitting',
            'Best balance',
            'Good performance'
        ]
    }
    
    model_df = pd.DataFrame(model_data)
    
    # Display model comparison table
    st.markdown("### 📊 Model Comparison Table")
    st.dataframe(model_df, use_container_width=True)
    
    # Highlight best model
    st.success("🏆 **Best Model: Random Forest** - Achieved R² ≈ 0.997 with MAE of ~R$ 35")
    
    st.markdown("---")
    
    # Evaluation metrics formulas
    st.markdown("### 📐 Evaluation Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Mean Absolute Error (MAE)")
        st.latex(r"MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|")
        st.markdown("""
        Measures the average magnitude of errors in predictions, 
        without considering their direction.
        """)
    
    with col2:
        st.markdown("#### Root Mean Squared Error (RMSE)")
        st.latex(r"RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}")
        st.markdown("""
        Measures the square root of the average of squared differences 
        between prediction and actual observation.
        """)
    
    with col3:
        st.markdown("#### Coefficient of Determination (R²)")
        st.latex(r"R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}")
        st.markdown("""
        Represents the proportion of variance in the dependent variable 
        predictable from the independent variables.
        """)
    
    st.markdown("---")
    
    # Visualization of model comparison
    st.markdown("### 📈 Model Performance Comparison")
    
    tab1, tab2 = st.tabs(["R² Score", "Error Metrics"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['lightcoral' if r2 < 0.9 else 'lightgreen' if r2 < 0.99 else 'gold' 
                 for r2 in model_df['R²']]
        bars = ax.barh(model_df['Model'], model_df['R²'], color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Comparison - R² Score', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # MAE comparison
        ax1.barh(model_df['Model'], model_df['MAE (R$)'], color='steelblue', edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('MAE (R$)', fontsize=12, fontweight='bold')
        ax1.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # RMSE comparison
        ax2.barh(model_df['Model'], model_df['RMSE (R$)'], color='coral', edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('RMSE (R$)', fontsize=12, fontweight='bold')
        ax2.set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Random Forest explanation
    st.markdown("### 🌲 Random Forest Model")
    
    st.markdown("""
    Random Forest is an ensemble learning method that constructs multiple decision trees 
    during training and outputs the average prediction of the individual trees.
    """)
    
    st.latex(r"\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)")
    
    st.markdown("""
    Where:
    - $\\hat{y}$ is the predicted value
    - $B$ is the number of trees in the forest
    - $T_b(x)$ is the prediction of the $b$-th tree for input $x$
    
    **Advantages:**
    - Reduces overfitting compared to single decision trees
    - Handles non-linear relationships well
    - Provides feature importance measures
    - Robust to outliers and noise
    """)

# ============================================================================
# MAKE PREDICTIONS PAGE
# ============================================================================
else:  # Make Predictions
    st.markdown('<h1 class="main-header">🔮 Make Predictions</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Enter the property characteristics below to get an estimated rental price prediction.
    This is a demonstration using a simplified prediction algorithm.
    """)
    
    # Create input form
    st.markdown("### 🏘️ Property Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.slider(
            "🏢 Area (m²)",
            min_value=15,
            max_value=300,
            value=60,
            step=5,
            help="Total area of the property in square meters"
        )
        
        bedrooms = st.slider(
            "🛏️ Number of Bedrooms",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="Number of bedrooms in the property"
        )
    
    with col2:
        bathrooms = st.slider(
            "🚿 Number of Bathrooms",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="Number of bathrooms in the property"
        )
        
        garage = st.slider(
            "🚗 Parking Spaces",
            min_value=0,
            max_value=4,
            value=1,
            step=1,
            help="Number of parking spaces available"
        )
    
    # Predict button
    if st.button("🔮 Calculate Rental Price", type="primary", use_container_width=True):
        
        # Simple prediction algorithm (demonstration)
        # Base price calculation
        base_price = area * 35  # R$ 35 per m²
        bedroom_adjustment = bedrooms * 450  # R$ 450 per bedroom
        bathroom_adjustment = bathrooms * 250  # R$ 250 per bathroom
        garage_adjustment = garage * 200  # R$ 200 per parking space
        
        # Calculate predicted rent
        predicted_rent = base_price + bedroom_adjustment + bathroom_adjustment + garage_adjustment
        
        # Add some random variation (±10%)
        np.random.seed(42)
        variation = np.random.uniform(0.9, 1.1)
        predicted_rent = predicted_rent * variation
        
        # Calculate confidence interval (demonstration)
        lower_bound = predicted_rent * 0.92
        upper_bound = predicted_rent * 1.08
        
        st.markdown("---")
        
        # Display prediction
        st.markdown("### 💰 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted Rent",
                f"R$ {predicted_rent:,.2f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Lower Bound (92%)",
                f"R$ {lower_bound:,.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Upper Bound (108%)",
                f"R$ {upper_bound:,.2f}",
                delta=None
            )
        
        # Display confidence interval
        st.info(f"📊 **Confidence Interval**: R$ {lower_bound:,.2f} - R$ {upper_bound:,.2f}")
        
        st.markdown("---")
        
        # Price breakdown visualization
        st.markdown("### 📊 Price Breakdown")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create pie chart
            components = {
                'Base (Area)': base_price * variation,
                'Bedrooms': bedroom_adjustment * variation,
                'Bathrooms': bathroom_adjustment * variation,
                'Parking': garage_adjustment * variation
            }
            
            fig, ax = plt.subplots(figsize=(10, 7))
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            explode = (0.05, 0.05, 0.05, 0.05)
            
            wedges, texts, autotexts = ax.pie(
                components.values(),
                labels=components.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                explode=explode,
                shadow=True
            )
            
            # Beautify the text
            for text in texts:
                text.set_fontsize(12)
                text.set_fontweight('bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(11)
                autotext.set_fontweight('bold')
            
            ax.set_title('Price Composition', fontsize=14, fontweight='bold', pad=20)
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### 💵 Component Values")
            for component, value in components.items():
                st.markdown(f"**{component}**: R$ {value:,.2f}")
            
            st.markdown("---")
            st.markdown(f"**Total**: R$ {predicted_rent:,.2f}")
        
        st.markdown("---")
        
        # Property summary
        st.markdown("### 📋 Property Summary")
        
        summary_data = {
            "Characteristic": ["Area", "Bedrooms", "Bathrooms", "Parking Spaces"],
            "Value": [f"{area} m²", bedrooms, bathrooms, garage]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
        
        # Disclaimer
        st.warning("""
        ⚠️ **Disclaimer**: This is a demonstration prediction using a simplified algorithm. 
        For accurate real-world predictions, a trained model using comprehensive data would be required. 
        The actual rental price may vary based on location, property condition, amenities, and market conditions.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Rental Price Prediction System | Developed with ❤️ using Streamlit</p>
    <p>© 2024 Ricardo Luís Bertolucci Filho</p>
</div>
""", unsafe_allow_html=True)
