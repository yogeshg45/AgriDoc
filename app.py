from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import joblib
import google.generativeai as genai
import os
import json
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# ------------------ WEATHER API CONFIG -------------------
OPENWEATHER_API_KEY = "04291de0601b991a13fde123960c41dc"

def get_weather_data(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def get_detailed_weather_advice(weather_data):
    temp = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    rain = weather_data.get('rain', {}).get('1h', 0) or 0
    wind_speed = weather_data.get('wind', {}).get('speed', 0)
    conditions = weather_data['weather'][0]['description']
    
    advice = {
        'general_condition': conditions.title(),
        'temperature_advice': '',
        'humidity_advice': '',
        'rainfall_advice': '',
        'fertilizer_timing': '',
        'crop_care': '',
        'irrigation_advice': ''
    }
    
    # Temperature advice
    if temp < 10:
        advice['temperature_advice'] = "Very cold conditions. Protect crops from frost. Avoid fertilizing."
        advice['fertilizer_timing'] = "❌ Not recommended - Too cold"
    elif 10 <= temp <= 15:
        advice['temperature_advice'] = "Cool weather. Monitor for cold stress in sensitive crops."
        advice['fertilizer_timing'] = "⚠️ Use with caution - Apply during warmer hours"
    elif 16 <= temp <= 30:
        advice['temperature_advice'] = "Ideal temperature range for most crops and fertilizer application."
        advice['fertilizer_timing'] = "✅ Excellent conditions for fertilizing"
    elif 31 <= temp <= 35:
        advice['temperature_advice'] = "Hot conditions. Ensure adequate irrigation."
        advice['fertilizer_timing'] = "⚠️ Apply early morning or evening only"
    else:
        advice['temperature_advice'] = "Extremely hot. Provide shade and extra water for crops."
        advice['fertilizer_timing'] = "❌ Avoid fertilizing - Risk of plant burn"
    
    # Humidity advice
    if humidity < 30:
        advice['humidity_advice'] = "Very dry air. Increase irrigation frequency."
        advice['irrigation_advice'] = "🚿 Increase watering - Low humidity detected"
    elif 30 <= humidity <= 60:
        advice['humidity_advice'] = "Optimal humidity levels for crop growth."
        advice['irrigation_advice'] = "💧 Normal irrigation schedule"
    elif 61 <= humidity <= 80:
        advice['humidity_advice'] = "High humidity. Monitor for fungal diseases."
        advice['irrigation_advice'] = "⚠️ Reduce watering - High humidity"
    else:
        advice['humidity_advice'] = "Very high humidity. Risk of fungal infections."
        advice['irrigation_advice'] = "❌ Minimal watering - Very high humidity"
    
    # Rainfall advice
    if rain > 10:
        advice['rainfall_advice'] = "Heavy rain detected. Ensure good drainage."
        advice['crop_care'] = "🌧️ Check drainage systems, avoid field operations"
    elif 1 <= rain <= 10:
        advice['rainfall_advice'] = "Light to moderate rain. Good for crops."
        advice['crop_care'] = "☔ Good natural irrigation, monitor soil moisture"
    else:
        advice['rainfall_advice'] = "No recent rainfall detected."
        advice['crop_care'] = "☀️ Monitor irrigation needs closely"
    
    return advice

# ------------------ ML MODEL CONFIG -------------------
try:
    PIPELINE_OUTPUT = "preprocessing_pipeline.joblib"
    MODEL_OUTPUT = "fertilizer_multioutput_rf.joblib"
    preprocessor = joblib.load(PIPELINE_OUTPUT)
    model_meta = joblib.load(MODEL_OUTPUT)
    model = model_meta["model"]
    top_indices = model_meta["top_indices"]
    top_feature_names = model_meta["top_feature_names"]
except:
    print("Warning: ML models not found. Some features may not work.")
    preprocessor = None
    model = None

feature_cols = [
    "Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity",
    "pH_Value", "Rainfall", "Crop", "Soil_Type", "Variety"
]

labels = ["N_deficit_kg_ha", "P_deficit_kg_ha", "K_deficit_kg_ha"]

FERTILIZERS = {
    "Urea (46-0-0)": {"N": 0.46, "color": "#e74c3c"},
    "DAP (18-46-0)": {"P": 0.46, "color": "#3498db"},
    "MOP (0-0-60)": {"K": 0.60, "color": "#2ecc71"},
}

# Sample data for dropdowns
CROP_OPTIONS = ["Rice", "Wheat", "Corn", "Sugarcane", "Cotton", "Tomato", "Potato", "Onion", "Cabbage", "Lettuce"]
SOIL_TYPE_OPTIONS = ["Clay", "Sandy", "Loam", "Black", "Red", "Alluvial", "Clayey", "Sandy Loam"]
VARIETY_OPTIONS = ["Hybrid", "Local", "Improved", "Traditional", "High Yield", "Drought Resistant"]

def generate_detailed_recommendation(row):
    recommendations = {
        'nitrogen': {},
        'phosphorus': {},
        'potassium': {},
        'overall_health': 'Good',
        'priority_actions': []
    }
    
    # Nitrogen analysis
    n_present = float(row["Nitrogen"])
    n_def = float(row["N_deficit_kg_ha"])
    recommendations['nitrogen'] = {
        'present': n_present,
        'deficit': n_def,
        'status': 'Sufficient' if n_def <= 0.01 else ('Low' if n_def < 50 else 'Very Low'),
        'fertilizer_needed': n_def / FERTILIZERS["Urea (46-0-0)"]["N"] if n_def > 0.01 else 0,
        'fertilizer_type': 'Urea (46-0-0)'
    }
    
    # Phosphorus analysis
    p_present = float(row["Phosphorus"])
    p_def = float(row["P_deficit_kg_ha"])
    recommendations['phosphorus'] = {
        'present': p_present,
        'deficit': p_def,
        'status': 'Sufficient' if p_def <= 0.01 else ('Low' if p_def < 30 else 'Very Low'),
        'fertilizer_needed': p_def / FERTILIZERS["DAP (18-46-0)"]["P"] if p_def > 0.01 else 0,
        'fertilizer_type': 'DAP (18-46-0)'
    }
    
    # Potassium analysis
    k_present = float(row["Potassium"])
    k_def = float(row["K_deficit_kg_ha"])
    recommendations['potassium'] = {
        'present': k_present,
        'deficit': k_def,
        'status': 'Sufficient' if k_def <= 0.01 else ('Low' if k_def < 40 else 'Very Low'),
        'fertilizer_needed': k_def / FERTILIZERS["MOP (0-0-60)"]["K"] if k_def > 0.01 else 0,
        'fertilizer_type': 'MOP (0-0-60)'
    }
    
    # Overall health assessment
    deficits = [n_def, p_def, k_def]
    if all(d <= 0.01 for d in deficits):
        recommendations['overall_health'] = 'Excellent'
    elif sum(1 for d in deficits if d > 0.01) == 1:
        recommendations['overall_health'] = 'Good'
    elif sum(1 for d in deficits if d > 0.01) == 2:
        recommendations['overall_health'] = 'Fair'
    else:
        recommendations['overall_health'] = 'Poor'
    
    # Priority actions
    if n_def > 0.01:
        recommendations['priority_actions'].append('Apply Nitrogen fertilizer urgently')
    if p_def > 0.01:
        recommendations['priority_actions'].append('Supplement with Phosphorus')
    if k_def > 0.01:
        recommendations['priority_actions'].append('Add Potassium fertilizer')
    
    return recommendations

# ------------------ GEMINI CHATBOT CONFIG -------------------
genai.configure(api_key="AIzaSyDh_q12etYVVvBmqqqZzfO5aGiWZ2Z-lB4")

try:
    chat_model = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ Gemini AI initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing Gemini AI: {e}")
    chat_model = None

conversation_history = {}

def get_enhanced_farming_prompt(user_message, user_id="default"):
    """Enhanced prompt with context and conversation history"""
    context = """You are Dr. AgriBot, an expert agricultural advisor and fertilizer specialist with 20+ years of experience in Indian farming conditions. 
    You provide practical, actionable farming advice to farmers worldwide, with special focus on Indian agriculture.
    
    Your expertise includes:
    - Soil analysis and nutrient management for Indian soil types
    - Crop selection and rotation strategies suitable for Indian climate
    - Fertilizer recommendations (both organic and synthetic) available in India
    - Pest and disease management for tropical/subtropical conditions
    - Weather-based farming decisions for monsoon patterns
    - Sustainable agriculture practices for small and medium farms
    - Cost-effective farming solutions for budget-conscious farmers
    - Modern farming technologies accessible to Indian farmers
    - Water management and irrigation techniques
    - Government schemes and subsidies for farmers
    
    Communication style:
    - Be friendly, encouraging, and professional like a caring agricultural officer
    - Use simple Hindi-English mixed language that Indian farmers understand
    - Provide specific, actionable advice with exact quantities
    - Include local market prices and availability when possible
    - Consider monsoon seasons, Rabi/Kharif crop cycles
    - Always prioritize sustainable and organic practices when possible
    - Give alternatives for expensive solutions
    - Include timing based on Indian agricultural calendar
    """
    
    # Get conversation history for this user
    history = conversation_history.get(user_id, [])
    
    # Build context from recent messages
    context_messages = ""
    if history:
        context_messages = "\n\nRecent conversation with this farmer:\n"
        for i, msg in enumerate(history[-3:]):  # Last 3 exchanges
            context_messages += f"Farmer: {msg['user']}\nDr. AgriBot: {msg['bot']}\n"
    
    full_prompt = f"""{context}{context_messages}
    
    Current farmer question: {user_message}
    
    Please provide detailed, practical advice in simple language. Include specific recommendations for:
    - What to do (step-by-step if needed)
    - When to do it (timing based on Indian seasons/months)
    - How much to use (exact quantities in kg/acre or as per Indian measurements)
    - Where to buy (local markets, cooperatives, online)
    - Approximate costs in Indian Rupees
    - Why it works (brief scientific explanation in simple terms)
    - Alternative cheaper options if available
    - Any government schemes or subsidies applicable
    
    Keep your response helpful and encouraging, like talking to a farmer friend. Aim for 250-400 words."""
    
    return full_prompt

# ------------------ SATELLITE DATA SIMULATION -------------------
def generate_satellite_data():
    """Generate simulated satellite data for demonstration"""
    current_time = datetime.now()
    return {
        "vegetation_health": round(random.uniform(75, 95), 1),
        "soil_moisture": round(random.uniform(45, 85), 1),
        "temperature": round(random.uniform(18, 35), 1),
        "area": round(random.uniform(20, 50), 2),
        "last_updated": f"{random.randint(1, 6)} hours ago",
        "ndvi_index": round(random.uniform(0.3, 0.8), 3),
        "crop_stress": "Low" if random.random() > 0.3 else "Medium",
        "irrigation_needed": random.choice(["Yes", "No", "Partial"]),
        "weather_alert": random.choice(["None", "High Temperature", "Low Humidity", "Storm Warning"])
    }

# ------------------ MARKETPLACE CONFIG (40 ITEMS) -------------------
marketplace_items = [
    # FERTILIZERS (10 items)
    {
        "id": 1, "name": "Urea (46-0-0)", "price": 550, 
        "desc": "High nitrogen content fertilizer ideal for leafy growth and chlorophyll production. Perfect for cereals and vegetables. Government subsidized rate.",
        "category": "fertilizer", "rating": 4.5, "stock": 150, "image": "💊"
    },
    {
        "id": 2, "name": "DAP (18-46-0)", "price": 1200, 
        "desc": "Diammonium phosphate rich in phosphorus for strong root development and early plant establishment. Essential for flowering crops.",
        "category": "fertilizer", "rating": 4.7, "stock": 89, "image": "🧪"
    },
    {
        "id": 3, "name": "MOP (0-0-60)", "price": 800, 
        "desc": "Muriate of potash providing essential potassium for fruit quality, disease resistance, and overall plant health. Great for fruit crops.",
        "category": "fertilizer", "rating": 4.3, "stock": 200, "image": "⚗️"
    },
    {
        "id": 4, "name": "NPK Complex (16-16-16)", "price": 900, 
        "desc": "Balanced fertilizer providing equal amounts of nitrogen, phosphorus, and potassium for general crop nutrition. One solution for all nutrients.",
        "category": "fertilizer", "rating": 4.4, "stock": 120, "image": "⚖️"
    },
    {
        "id": 5, "name": "Calcium Ammonium Nitrate", "price": 750, 
        "desc": "Quick-release nitrogen source with calcium for improved soil structure. Reduces acidity and provides immediate nutrient availability.",
        "category": "fertilizer", "rating": 4.6, "stock": 85, "image": "🧱"
    },
    {
        "id": 6, "name": "Single Super Phosphate", "price": 650, 
        "desc": "Phosphorus fertilizer with sulphur and calcium. Excellent for root development in acid soils. Long-lasting phosphorus source.",
        "category": "fertilizer", "rating": 4.2, "stock": 110, "image": "🏭"
    },
    {
        "id": 7, "name": "Potassium Sulphate", "price": 950, 
        "desc": "Premium potassium source with sulphur. Chloride-free formulation perfect for fruits, vegetables, and salt-sensitive crops.",
        "category": "fertilizer", "rating": 4.8, "stock": 75, "image": "💎"
    },
    {
        "id": 8, "name": "Magnesium Sulphate", "price": 420, 
        "desc": "Essential for chlorophyll production and enzyme activation. Prevents yellowing of leaves and improves photosynthesis efficiency.",
        "category": "fertilizer", "rating": 4.1, "stock": 140, "image": "🔬"
    },
    {
        "id": 9, "name": "Zinc Sulphate", "price": 380, 
        "desc": "Micronutrient fertilizer crucial for plant growth hormones. Prevents stunted growth and improves grain quality in cereals.",
        "category": "fertilizer", "rating": 4.3, "stock": 95, "image": "⚡"
    },
    {
        "id": 10, "name": "NPK 20-20-0", "price": 850, 
        "desc": "High nitrogen-phosphorus fertilizer for early crop stages. Promotes vigorous vegetative growth and strong root establishment.",
        "category": "fertilizer", "rating": 4.5, "stock": 105, "image": "🚀"
    },
    
    # SEEDS (10 items)
    {
        "id": 11, "name": "Hybrid Rice Seeds (IR64)", "price": 1500, 
        "desc": "High-yielding hybrid rice variety with excellent grain quality and disease resistance. Suitable for both Kharif and Rabi seasons.",
        "category": "seeds", "rating": 4.8, "stock": 45, "image": "🌾"
    },
    {
        "id": 12, "name": "Wheat Seeds (HD-2967)", "price": 850, 
        "desc": "Dwarf variety wheat seeds with high protein content. Drought-tolerant and suitable for late sowing conditions.",
        "category": "seeds", "rating": 4.6, "stock": 60, "image": "🌾"
    },
    {
        "id": 13, "name": "Maize Seeds (NK-6240)", "price": 1200, 
        "desc": "High-yielding hybrid maize with excellent cob filling. Resistant to fall armyworm and suitable for mechanized farming.",
        "category": "seeds", "rating": 4.7, "stock": 35, "image": "🌽"
    },
    {
        "id": 14, "name": "Cotton Seeds (BT-Cotton)", "price": 2200, 
        "desc": "Genetically modified cotton seeds with built-in pest resistance. Higher fiber quality and reduced pesticide requirement.",
        "category": "seeds", "rating": 4.9, "stock": 25, "image": "🌿"
    },
    {
        "id": 15, "name": "Tomato Seeds (Hybrid)", "price": 3500, 
        "desc": "Determinate hybrid tomato variety with uniform fruit size. High yield potential with disease resistance to bacterial wilt.",
        "category": "seeds", "rating": 4.4, "stock": 50, "image": "🍅"
    },
    {
        "id": 16, "name": "Onion Seeds (Nasik Red)", "price": 2800, 
        "desc": "Traditional red onion variety with long storage life. Suitable for export quality with strong pungency and good keeping quality.",
        "category": "seeds", "rating": 4.2, "stock": 40, "image": "🧅"
    },
    {
        "id": 17, "name": "Chili Seeds (Green Hot)", "price": 1800, 
        "desc": "High-yielding green chili variety with consistent fruit size. Tolerant to leaf curl virus and thrips damage.",
        "category": "seeds", "rating": 4.5, "stock": 55, "image": "🌶️"
    },
    {
        "id": 18, "name": "Soybean Seeds (JS-335)", "price": 950, 
        "desc": "Short-duration soybean variety suitable for late planting. High oil content and resistant to yellow mosaic virus.",
        "category": "seeds", "rating": 4.3, "stock": 70, "image": "🫘"
    },
    {
        "id": 19, "name": "Sunflower Seeds (Hybrid)", "price": 1100, 
        "desc": "High oil content sunflower with large flower heads. Drought-tolerant and suitable for rain-fed conditions.",
        "category": "seeds", "rating": 4.6, "stock": 30, "image": "🌻"
    },
    {
        "id": 20, "name": "Mustard Seeds (Pusa Bold)", "price": 680, 
        "desc": "High-yielding mustard variety with good oil content. Cold-tolerant and suitable for late Rabi sowing.",
        "category": "seeds", "rating": 4.1, "stock": 85, "image": "🌼"
    },
    
    # EQUIPMENT (10 items)
    {
        "id": 21, "name": "Knapsack Sprayer (16L)", "price": 2500, 
        "desc": "High-pressure manual sprayer for pesticide and fertilizer application. Ergonomic design with adjustable nozzle settings.",
        "category": "equipment", "rating": 4.3, "stock": 25, "image": "🎒"
    },
    {
        "id": 22, "name": "Seed Drill Machine", "price": 35000, 
        "desc": "Tractor-mounted seed drill for precise seed placement. Adjustable row spacing and depth control for multiple crops.",
        "category": "equipment", "rating": 4.7, "stock": 8, "image": "🚜"
    },
    {
        "id": 23, "name": "Rotavator (5ft)", "price": 45000, 
        "desc": "Heavy-duty rotavator for soil preparation and stubble management. Compatible with 35-50 HP tractors.",
        "category": "equipment", "rating": 4.6, "stock": 12, "image": "⚙️"
    },
    {
        "id": 24, "name": "Harvesting Sickle", "price": 180, 
        "desc": "Sharp stainless steel sickle for manual crop harvesting. Ergonomic wooden handle for comfortable grip during long use.",
        "category": "equipment", "rating": 4.1, "stock": 150, "image": "🔪"
    },
    {
        "id": 25, "name": "Irrigation Pump (5HP)", "price": 18500, 
        "desc": "Centrifugal water pump for irrigation systems. High efficiency motor with corrosion-resistant impeller design.",
        "category": "equipment", "rating": 4.5, "stock": 18, "image": "💧"
    },
    {
        "id": 26, "name": "Mulching Film (100m)", "price": 850, 
        "desc": "Black polyethylene mulching film for weed control and moisture retention. UV-stabilized for longer field life.",
        "category": "equipment", "rating": 4.2, "stock": 80, "image": "🎞️"
    },
    {
        "id": 27, "name": "Greenhouse Kit (1000sqft)", "price": 125000, 
        "desc": "Complete greenhouse setup with ventilation system. Includes shade nets, misting system, and climate control equipment.",
        "category": "equipment", "rating": 4.8, "stock": 5, "image": "🏠"
    },
    {
        "id": 28, "name": "Soil pH Meter", "price": 1200, 
        "desc": "Digital soil pH and moisture meter for field testing. Instant readings with temperature compensation feature.",
        "category": "equipment", "rating": 4.4, "stock": 35, "image": "📏"
    },
    {
        "id": 29, "name": "Threshing Machine", "price": 85000, 
        "desc": "Multi-crop threshing machine for wheat, rice, and other cereals. High efficiency with minimal grain damage.",
        "category": "equipment", "rating": 4.6, "stock": 6, "image": "🏭"
    },
    {
        "id": 30, "name": "Drip Irrigation Kit (1 Acre)", "price": 25000, 
        "desc": "Complete drip irrigation system for 1 acre. Includes filters, pressure regulators, and emitter lines with fittings.",
        "category": "equipment", "rating": 4.9, "stock": 15, "image": "💦"
    },
    
    # ORGANIC (10 items)
    {
        "id": 31, "name": "Organic Compost (Premium)", "price": 400, 
        "desc": "Premium organic compost made from cow dung and decomposed plant materials. Improves soil structure naturally and enhances microbial activity.",
        "category": "organic", "rating": 4.6, "stock": 300, "image": "🌱"
    },
    {
        "id": 32, "name": "Vermicompost", "price": 650, 
        "desc": "Earthworm processed organic fertilizer rich in nutrients. Improves soil fertility and water retention capacity significantly.",
        "category": "organic", "rating": 4.8, "stock": 180, "image": "🪱"
    },
    {
        "id": 33, "name": "Neem Oil Concentrate", "price": 280, 
        "desc": "Pure neem oil extract for organic pest control. Natural pesticide effective against aphids, thrips, and whiteflies.",
        "category": "organic", "rating": 4.5, "stock": 120, "image": "🌿"
    },
    {
        "id": 34, "name": "Seaweed Extract Fertilizer", "price": 850, 
        "desc": "Liquid seaweed extract rich in micronutrients and growth hormones. Enhances plant immunity and stress tolerance.",
        "category": "organic", "rating": 4.7, "stock": 85, "image": "🌊"
    },
    {
        "id": 35, "name": "Bone Meal Fertilizer", "price": 520, 
        "desc": "Slow-release phosphorus source from animal bones. Ideal for flowering plants and long-term soil nutrient supply.",
        "category": "organic", "rating": 4.3, "stock": 95, "image": "🦴"
    },
    {
        "id": 36, "name": "Bio-NPK Fertilizer", "price": 320, 
        "desc": "Microbial consortium for nitrogen, phosphorus, and potassium mobilization. Reduces chemical fertilizer requirement by 25-30%.",
        "category": "organic", "rating": 4.4, "stock": 140, "image": "🦠"
    },
    {
        "id": 37, "name": "Organic Pesticide (Karanj)", "price": 380, 
        "desc": "Karanj oil-based organic pesticide for soil and foliar application. Controls nematodes and soil-borne pests naturally.",
        "category": "organic", "rating": 4.2, "stock": 110, "image": "🌰"
    },
    {
        "id": 38, "name": "Trichoderma Viride", "price": 180, 
        "desc": "Beneficial fungus for soil health and disease suppression. Protects roots from fungal pathogens and improves nutrient uptake.",
        "category": "organic", "rating": 4.6, "stock": 200, "image": "🍄"
    },
    {
        "id": 39, "name": "Jeevamrit Concentrate", "price": 150, 
        "desc": "Traditional organic growth promoter made from cow products. Enhances soil microbial activity and plant growth naturally.",
        "category": "organic", "rating": 4.1, "stock": 250, "image": "🥛"
    },
    {
        "id": 40, "name": "Rock Phosphate", "price": 450, 
        "desc": "Natural phosphorus source for long-term soil fertility. Slow-release formula ideal for organic farming systems.",
        "category": "organic", "rating": 4.3, "stock": 90, "image": "🪨"
    }
]

# API endpoint to get marketplace products
@app.route("/api/marketplace/products", methods=["GET"])
def get_marketplace_products():
    """API endpoint to get marketplace products with AI-enhanced pricing"""
    try:
        category = request.args.get('category', 'all')
        
        # Filter products by category
        if category != 'all':
            filtered_items = [item for item in marketplace_items if item['category'] == category]
        else:
            filtered_items = marketplace_items
        
        # Simulate AI-enhanced pricing (in production, this would use Gemini API)
        for item in filtered_items:
            # Add slight price variation to simulate market fluctuations
            base_price = item['price']
            fluctuation = random.uniform(0.95, 1.05)  # ±5% variation
            ai_price = int(base_price * fluctuation)
            
            item['ai_price'] = ai_price
            item['price_change'] = ai_price - base_price
            item['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify({
            "status": "success",
            "products": filtered_items,
            "total_count": len(filtered_items)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ------------------ MAIN ROUTES -------------------
@app.route("/")
def index():
    return render_template("home.html", items=marketplace_items)

@app.route("/weather")
def weather():
    return render_template("weather.html")

@app.route("/weather", methods=["POST"])
def weather_result():
    city = request.form.get("city")
    if not city:
        return render_template("weather.html", error="Please enter a city name.")
    
    weather_data = get_weather_data(city)
    if weather_data:
        advice = get_detailed_weather_advice(weather_data)
        weather_info = {
            "temp": weather_data['main']['temp'],
            "humidity": weather_data['main']['humidity'],
            "pressure": weather_data['main']['pressure'],
            "wind_speed": weather_data.get('wind', {}).get('speed', 0),
            "rain": weather_data.get('rain', {}).get('1h', 0) or 0,
            "description": weather_data['weather'][0]['description'].title(),
            "city": city.title(),
            "advice": advice
        }
        return render_template("weather.html", weather_info=weather_info, city=city)
    else:
        return render_template("weather.html", error="Could not retrieve weather data. Please check the city name and try again.")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html", 
                         crop_options=CROP_OPTIONS,
                         soil_type_options=SOIL_TYPE_OPTIONS,
                         variety_options=VARIETY_OPTIONS)

@app.route("/prediction", methods=["POST"])
def predict_result():
    if not model:
        return render_template("prediction.html", 
                             error="ML models not loaded. Please check model files.",
                             crop_options=CROP_OPTIONS,
                             soil_type_options=SOIL_TYPE_OPTIONS,
                             variety_options=VARIETY_OPTIONS)
    
    try:
        # Handle file upload
        if "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            df_new = pd.read_csv(file)
        else:
            # Handle manual input
            data_dict = {}
            for col in feature_cols:
                value = request.form.get(col)
                if not value:
                    return render_template("prediction.html", 
                                         error=f"Please provide value for {col}",
                                         crop_options=CROP_OPTIONS,
                                         soil_type_options=SOIL_TYPE_OPTIONS,
                                         variety_options=VARIETY_OPTIONS)
                data_dict[col] = [value]
            df_new = pd.DataFrame.from_dict(data_dict)

        # Validate required columns
        missing_cols = [col for col in feature_cols if col not in df_new.columns]
        if missing_cols:
            return render_template("prediction.html", 
                                 error=f"Missing columns: {', '.join(missing_cols)}",
                                 crop_options=CROP_OPTIONS,
                                 soil_type_options=SOIL_TYPE_OPTIONS,
                                 variety_options=VARIETY_OPTIONS)

        # Process predictions
        Xraw = df_new[feature_cols].copy()
        Xtrans = preprocessor.transform(Xraw)
        Xtrans_top = Xtrans[:, top_indices]
        preds = model.predict(Xtrans_top)
        
        preds_df = pd.DataFrame(preds, columns=labels)
        result_df = pd.concat([df_new.reset_index(drop=True), preds_df], axis=1)

        results = []
        for _, row in result_df.iterrows():
            recommendation = generate_detailed_recommendation(row)
            results.append({
                "input_data": {
                    "crop": row["Crop"],
                    "soil_type": row["Soil_Type"],
                    "variety": row["Variety"],
                    "temperature": row["Temperature"],
                    "humidity": row["Humidity"],
                    "ph": row["pH_Value"],
                    "rainfall": row["Rainfall"],
                    "nitrogen": row["Nitrogen"],
                    "phosphorus": row["Phosphorus"],
                    "potassium": row["Potassium"]
                },
                "recommendation": recommendation
            })

        return render_template("prediction.html", 
                             results=results,
                             crop_options=CROP_OPTIONS,
                             soil_type_options=SOIL_TYPE_OPTIONS,
                             variety_options=VARIETY_OPTIONS)
        
    except Exception as e:
        return render_template("prediction.html", 
                             error=f"Error processing request: {str(e)}",
                             crop_options=CROP_OPTIONS,
                             soil_type_options=SOIL_TYPE_OPTIONS,
                             variety_options=VARIETY_OPTIONS)

@app.route("/marketplace")
def marketplace():
    category = request.args.get('category', 'all')
    if category != 'all':
        filtered_items = [item for item in marketplace_items if item['category'] == category]
    else:
        filtered_items = marketplace_items
    return render_template("marketplace.html", items=filtered_items)

@app.route("/satellite")
def satellite():
    return render_template("satellite.html")

@app.route("/api/satellite/data", methods=["GET"])
def get_satellite_data():
    """API endpoint to get satellite data"""
    try:
        data = generate_satellite_data()
        return jsonify({
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/analytics", methods=["POST"])
def analytics_result():
    try:
        if "file" not in request.files or not request.files["file"].filename:
            return render_template("analytics.html", error="Please upload a CSV file for analysis.")
        
        file = request.files["file"]
        df = pd.read_csv(file)
        
        # Check for required columns
        required_columns = ["N_deficit_kg_ha", "P_deficit_kg_ha", "K_deficit_kg_ha"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return render_template("analytics.html", 
                                 error=f"Missing required columns: {', '.join(missing_columns)}. Please ensure your CSV contains nutrient deficit data.")
        
        # Calculate comprehensive statistics
        stats = {}
        for col in required_columns:
            nutrient_name = col.replace('_deficit_kg_ha', '').replace('_', ' ').title()
            stats[nutrient_name] = {
                'mean': round(df[col].mean(), 3),
                'min': round(df[col].min(), 3),
                'max': round(df[col].max(), 3),
                'std': round(df[col].std(), 3),
                'count': len(df[col])
            }
        
        # Generate insights
        insights = []
        total_samples = len(df)
        
        for nutrient, values in stats.items():
            if values['mean'] > 50:
                severity = "High"
                icon = "🔴"
            elif values['mean'] > 20:
                severity = "Medium"
                icon = "🟡"
            else:
                severity = "Low" 
                icon = "🟢"
            
            insights.append({
                'nutrient': nutrient,
                'severity': severity,
                'icon': icon,
                'message': f"Average {nutrient.lower()} deficit is {values['mean']} kg/ha ({severity.lower()} level)",
                'recommendation': get_nutrient_recommendation(nutrient, values['mean'])
            })
        
        return render_template("analytics.html", stats=stats, insights=insights, total_samples=total_samples)
        
    except Exception as e:
        return render_template("analytics.html", error=f"Error processing file: {str(e)}")

# ------------------ CHATBOT ROUTES -------------------
@app.route("/chatbot")
def chatbot_page():
    """Render the chatbot interface with Gemini AI integration"""
    return render_template("chatbot.html")

@app.route("/api/chatbot", methods=["POST"])
def chatbot_api():
    """Handle chatbot API requests using Gemini AI"""
    try:
        # Check if Gemini AI is available
        if not chat_model:
            return jsonify({
                "error": "AI service is currently unavailable. Please try again later.",
                "status": "error"
            }), 503
        
        data = request.get_json()
        user_message = data.get("message", "").strip()
        user_id = data.get("user_id", "default")
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Get enhanced prompt with context and conversation history
        farming_prompt = get_enhanced_farming_prompt(user_message, user_id)
        
        # Generate response using Gemini AI
        try:
            response = chat_model.generate_content(farming_prompt)
            bot_reply = response.text
            
            # Store conversation history
            if user_id not in conversation_history:
                conversation_history[user_id] = []
            
            conversation_history[user_id].append({
                'user': user_message,
                'bot': bot_reply
            })
            
            # Keep only last 10 exchanges per user to manage memory
            if len(conversation_history[user_id]) > 10:
                conversation_history[user_id] = conversation_history[user_id][-10:]
            
            return jsonify({
                "reply": bot_reply,
                "status": "success"
            })
            
        except Exception as gemini_error:
            print(f"Gemini AI error: {str(gemini_error)}")
            return jsonify({
                "error": "I'm having trouble processing your question right now. Please try rephrasing or ask something else.",
                "status": "error"
            }), 500
        
    except Exception as e:
        print(f"Chatbot API error: {str(e)}")
        return jsonify({
            "error": "Sorry, I'm experiencing technical difficulties. Please try again in a moment.",
            "status": "error"
        }), 500

@app.route("/api/chatbot/suggestions", methods=["GET"])
def get_suggestions():
    """Get suggested questions for the chatbot"""
    suggestions = [
        "मेरे टमाटर के लिए कौन सा खाद अच्छा है?",
        "What fertilizer should I use for tomatoes?",
        "How to improve soil pH naturally?",
        "Best time to apply nitrogen fertilizer?",
        "धान की फसल के लिए कब पानी दें?",
        "Organic pest control methods for vegetables",
        "How to increase crop yield sustainably?",
        "मिट्टी में पोषक तत्वों की कमी के लक्षण",
        "Signs of nutrient deficiency in plants",
        "Crop rotation strategies for small farms",
        "सूखे के दौरान पानी का प्रबंधन कैसे करें?",
        "Water management during drought",
        "Choosing the right fertilizer NPK ratio",
        "अगले सीजन के लिए मिट्टी की तैयारी",
        "Soil preparation for the next season"
    ]
    return jsonify({"suggestions": suggestions})

@app.route("/api/chatbot/health", methods=["GET"])
def chatbot_health():
    """Check if the chatbot AI service is working"""
    try:
        if chat_model:
            # Test the model with a simple prompt
            test_response = chat_model.generate_content("Hello, are you working?")
            return jsonify({
                "status": "healthy",
                "ai_service": "operational",
                "model": "gemini-1.5-flash"
            })
        else:
            return jsonify({
                "status": "degraded",
                "ai_service": "unavailable",
                "error": "Gemini AI not initialized"
            }), 503
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "ai_service": "error",
            "error": str(e)
        }), 503

def get_nutrient_recommendation(nutrient, mean_deficit):
    """Generate specific recommendations based on nutrient deficits"""
    if nutrient == "N":
        if mean_deficit > 50:
            return "Apply Urea fertilizer immediately. Consider split application for better uptake."
        elif mean_deficit > 20:
            return "Moderate nitrogen application needed. Apply Urea as per soil test recommendations."
        else:
            return "Nitrogen levels are adequate. Monitor regularly."
    elif nutrient == "P":
        if mean_deficit > 30:
            return "Apply DAP or single superphosphate. Phosphorus is crucial for root development."
        elif mean_deficit > 10:
            return "Light phosphorus application recommended. Use DAP for quick results."
        else:
            return "Phosphorus levels are sufficient. No immediate action needed."
    else:  # Potassium
        if mean_deficit > 40:
            return "Apply MOP (Potash) fertilizer. Essential for fruit quality and disease resistance."
        elif mean_deficit > 15:
            return "Moderate potassium application needed. Use MOP as recommended."
        else:
            return "Potassium levels are good. Continue current management practices."

# Error handlers for better user experience
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    print("🚀 Starting Smart Farming Assistant...")
    print("🤖 Gemini AI Status:", "✅ Ready" if chat_model else "❌ Failed")
    print("🛰️ Satellite Monitoring: ✅ Ready")
    print(f"📦 Marketplace Items: {len(marketplace_items)} products loaded")
    print(f"   - Fertilizers: {len([i for i in marketplace_items if i['category'] == 'fertilizer'])}")
    print(f"   - Seeds: {len([i for i in marketplace_items if i['category'] == 'seeds'])}")
    print(f"   - Equipment: {len([i for i in marketplace_items if i['category'] == 'equipment'])}")
    print(f"   - Organic: {len([i for i in marketplace_items if i['category'] == 'organic'])}")
    app.run(debug=True, host='0.0.0.0', port=5000)
